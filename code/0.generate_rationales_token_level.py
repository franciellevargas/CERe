import json
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
import string

# --- Configuração ---
MODEL_NAME = "contriever-finetuned-clinicaltrials"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# --- spaCy: carregar idioma desejado ---
nlp = spacy.load("en_core_web_sm")  # mude para "pt_core_news_sm" se for PT-BR

# --- Caracteres de pontuação ---
punctuation_chars = set(string.punctuation)

def merge_subtokens(tokens, scores):
    merged_tokens, merged_scores = [], []
    current_word, current_scores = "", []

    for tok, score in zip(tokens, scores):
        if tok.startswith("##"):
            current_word += tok[2:]
            current_scores.append(score)
        else:
            if current_word:
                merged_tokens.append(current_word)
                merged_scores.append(sum(current_scores) / len(current_scores))
            current_word = tok
            current_scores = [score]

    if current_word:
        merged_tokens.append(current_word)
        merged_scores.append(sum(current_scores) / len(current_scores))

    return merged_tokens, merged_scores

def is_valid_token(tok):
    """
    Retorna True se o token não for stopword, não for pontuação pura e tiver mais de 1 letra
    """
    if len(tok) <= 1:
        return False
    if all(c in punctuation_chars for c in tok):
        return False
    doc = nlp(tok)
    if any(token.is_stop for token in doc):
        return False
    return True

def token_rationales(query, passage, top_k_tokens=5):
    with torch.no_grad():
        # Encode query (pooled)
        q_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        q_emb = model(**q_inputs).last_hidden_state.mean(dim=1)

        # Encode passage (token-level)
        p_inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        p_out = model(**p_inputs)
        token_embs = p_out.last_hidden_state.squeeze(0)

        # Dot product
        scores = torch.matmul(token_embs, q_emb.squeeze(0))

        # Converter IDs → tokens
        tokens = tokenizer.convert_ids_to_tokens(p_inputs['input_ids'][0])

        # Merge subtokens
        merged_tokens, merged_scores = merge_subtokens(tokens, scores.tolist())

        # Filtrar tokens
        filtered = [(tok, score) for tok, score in zip(merged_tokens, merged_scores) if is_valid_token(tok)]

        # Top-K
        topk = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k_tokens]
        return topk

# --- Processa o corpus ---
with open("contriever_valtest_candidates.jsonl", "r") as f:
    corpus = [json.loads(line) for line in f]

results = []

for item in corpus:
    query = item['query']
    positives = [c['text'] for c in item['candidates'] if c['label'] == 'positive']
    
    for passage in positives:
        rationales = token_rationales(query, passage, top_k_tokens=10)
        results.append({
            'query': query,
            'passage': passage,
            'rationales': rationales
        })

# --- Salva em JSONL ---
with open("tcer_rationales.jsonl", "w") as f_out:
    for r in results:
        f_out.write(json.dumps(r) + "\n")

print(f"Processamento concluído! Total de {len(results)} exemplos salvos em 'tcer_rationales.jsonl'")


