# features.py
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Danh sách từ nối cơ bản (cohesion markers)
CONNECTIVES = ["however", "therefore", "moreover", "furthermore", 
               "in addition", "for example", "in conclusion"]

def _sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def _tokens(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())

def lexical_features(essay: str):
    toks = _tokens(essay)
    n_tokens = len(toks)
    n_types = len(set(toks))
    return {
        "ttr": n_types / (n_tokens+1e-9),          # Type-Token Ratio
        "avg_word_len": np.mean([len(w) for w in toks]) if toks else 0
    }

def coherence_features(essay: str):
    sents = _sentences(essay)
    avg_len = np.mean([len(_tokens(s)) for s in sents]) if sents else 0
    connectives = sum(essay.lower().count(c) for c in CONNECTIVES)
    return {"n_sent": len(sents), "avg_sent_len": avg_len, "connectives": connectives}

def task_response_features(prompt: str, essay: str):
    vect = TfidfVectorizer()
    X = vect.fit_transform([prompt, essay])
    sim = cosine_similarity(X[0], X[1])[0,0]
    return {"prompt_essay_sim": sim}

def extract_all_features(prompt: str, essay: str):
    feats = {}
    feats.update(task_response_features(prompt, essay))
    feats.update(coherence_features(essay))
    feats.update(lexical_features(essay))
    return feats
