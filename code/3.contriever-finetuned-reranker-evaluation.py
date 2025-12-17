import json
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

# =============================
# CONFIG
# =============================
MODEL_FINE = "contriever-finetuned-cosine"
INPUT_JSON = "contriever_base_full_rankings.json"
OUTPUT_JSON = "contriever_finetuned_reranked_full_rankings.json"

MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [1, 3, 5, 10, 50]

# =============================
# LOAD MODEL
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_FINE)
model = AutoModel.from_pretrained(MODEL_FINE).to(DEVICE)
model.eval()

# =============================
# EMBEDDING
# =============================
def embed(texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    emb = output.last_hidden_state[:, 0]  # CLS
    emb = F.normalize(emb, p=2, dim=1)
    return emb

# =============================
# LOAD BASE RANKINGS
# =============================
def load_rankings(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =============================
# RE-RANK TOP-50
# =============================
def rerank_all(rankings):

    reranked_results = []

    for ex in tqdm(rankings, desc="🔄 Re-ranking (Contriever fine-tuned)"):

        pmcid = ex.get("PMCID", "UNKNOWN")
        query = ex["query"]
        ranking = ex["top_50_ranking"]

        passages = [item["text"] for item in ranking]
        labels = [item["label"] for item in ranking]

        q_emb = embed([query])
        p_emb = embed(passages)

        scores = torch.matmul(q_emb, p_emb.T).squeeze(0).tolist()

        ranked = sorted(
            zip(passages, labels, scores),
            key=lambda x: x[2],
            reverse=True
        )

        new_ranking = []
        positive_rank = None

        for i, (text, label, score) in enumerate(ranked, start=1):
            new_ranking.append({
                "rank": i,
                "label": label,
                "score": score,
                "text": text
            })
            if label == "POS":
                positive_rank = i

        reranked_results.append({
            "PMCID": pmcid,
            "query": query,
            "positive_rank": positive_rank,
            "top_50_reranked": new_ranking
        })

    return reranked_results

# =============================
# EVALUATION
# =============================
def evaluate(rankings):

    recall_at_k = {k: 0 for k in K_VALUES}
    hit_at_1 = 0
    mrr_total = 0.0
    rank_distribution = Counter()

    for ex in rankings:
        r = ex["positive_rank"]

        rank_distribution[r] += 1
        mrr_total += 1.0 / r

        if r == 1:
            hit_at_1 += 1

        for k in K_VALUES:
            if r <= k:
                recall_at_k[k] += 1

    n = len(rankings)

    recall_at_k = {k: v / n for k, v in recall_at_k.items()}
    hit_at_1 /= n
    mrr = mrr_total / n

    return recall_at_k, hit_at_1, mrr, rank_distribution

# =============================
# RUN
# =============================
if __name__ == "__main__":

    rankings = load_rankings(INPUT_JSON)
    print(f"📦 Total de queries: {len(rankings)}")

    reranked = rerank_all(rankings)

    # salvar ranking re-ranqueado
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(reranked, f, indent=2, ensure_ascii=False)

    recall_at_k, hit_at_1, mrr, rank_dist = evaluate(reranked)

    print("\n==============================")
    print("📊 RESULTADOS – RE-RANK (CONTRIEVER FINE-TUNED)")
    print("==============================")

    print("\nRecall@K:")
    for k, v in recall_at_k.items():
        print(f"Recall@{k}: {v:.4f}")

    print(f"\nHit@1: {hit_at_1:.4f}")
    print(f"MRR:   {mrr:.4f}")

    print("\nDistribuição dos ranks do positivo:")
    for rank in sorted(rank_dist):
        print(f"Rank {rank}: {rank_dist[rank]}")

    print(f"\n📁 Rankings re-ranqueados salvos em: {OUTPUT_JSON}")


