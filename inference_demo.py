import os, pickle, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ==== Config ====
MODEL_NAME = "microsoft/deberta-v3-base"   # backbone đã dùng lúc train
POOL = "cls"                               # pooling: "cls" hoặc "mean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load tokenizer & model (encoder) ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def encode_texts(texts, batch_size=8, pool="cls", max_len=512):
    """Trả về embedding cho list văn bản"""
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc).last_hidden_state  # [B, L, H]
        if pool == "cls":
            pooled = out[:, 0, :]  # [CLS] token
        else:  # mean pooling (mask aware)
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1)
        outs.append(pooled.cpu().numpy())
    return np.vstack(outs)


# ==== Cheap features giống lúc train ====
def cheap_feats(text: str):
    sents = re.split(r"[.!?]", text)
    sents = [s.strip() for s in sents if s.strip()]
    words = re.findall(r"\w+", text)
    n_words = len(words)
    n_sents = len(sents)
    ttr = len(set(words)) / (len(words) + 1e-6)
    avg_len = n_words / (n_sents+1e-6)
    return np.array([n_words, n_sents, avg_len, ttr], dtype=float)   # chỉ 4 feature

# ==== Các feature rubric proxy (bản đơn giản) ====
def tr_feats(text):  # Task Response
    return np.array([len(text), text.count("?"), text.count("!")], dtype=float)

def cc_feats(text):  # Coherence & Cohesion
    return np.array([text.count("because"), text.count("therefore"), text.count("although")], dtype=float)

def lr_feats(text):  # Lexical Resource
    words = re.findall(r"\w+", text.lower())
    return np.array([len(set(words)), np.mean([len(w) for w in words]) if words else 0], dtype=float)

def gr_feats(text):  # Grammar
    return np.array([text.count(","), text.count(";"), text.count("'")], dtype=float)


# ==== Hàm round band ====
def round_to_half(x: float) -> float:
    return float(np.round(x*2)/2.0)

def compute_overall_prep(tr, cc, lr, gr):
    raw = (tr+cc+lr+gr)/2.0
    return (int(np.floor(raw)))/2.0


# ==== Load 5 models đã train ====
save_dir = "/Users/vanhkhongpeo/Documents/Prep_Test/models_save"   # thư mục local chứa pkl
with open(os.path.join(save_dir, "models_overall.pkl"), "rb") as f:
    models_overall = pickle.load(f)
with open(os.path.join(save_dir, "head_tr.pkl"), "rb") as f:
    head_tr = pickle.load(f)
with open(os.path.join(save_dir, "head_cc.pkl"), "rb") as f:
    head_cc = pickle.load(f)
with open(os.path.join(save_dir, "head_lr.pkl"), "rb") as f:
    head_lr = pickle.load(f)
with open(os.path.join(save_dir, "head_gr.pkl"), "rb") as f:
    head_gr = pickle.load(f)

print("✅ Loaded all 5 models")


# ==== Hàm inference cho 1 essay ====
def predict_essay(text, alpha=0.7):
    # encode
    v = encode_texts([text], pool=POOL)
    f = cheap_feats(text).reshape(1, -1)
    x = np.hstack([v, f])

    # overall direct
    overall_direct_raw = np.mean([m.predict(x)[0] for m in models_overall])
    overall_direct = round_to_half(overall_direct_raw)

    # rubric heads
    Xt = np.hstack([v, tr_feats(text).reshape(1, -1)])
    Xc = np.hstack([v, cc_feats(text).reshape(1, -1)])
    Xl = np.hstack([v, lr_feats(text).reshape(1, -1)])
    Xg = np.hstack([v, gr_feats(text).reshape(1, -1)])

    tr_pred = int(np.clip(np.rint(np.mean([m.predict(Xt)[0] for m in head_tr])), 1, 9))
    cc_pred = int(np.clip(np.rint(np.mean([m.predict(Xc)[0] for m in head_cc])), 1, 9))
    lr_pred = int(np.clip(np.rint(np.mean([m.predict(Xl)[0] for m in head_lr])), 1, 9))
    gr_pred = int(np.clip(np.rint(np.mean([m.predict(Xg)[0] for m in head_gr])), 1, 9))

    overall_from_rubrics = compute_overall_prep(tr_pred, cc_pred, lr_pred, gr_pred)

    # blend
    overall_final = round_to_half(alpha*overall_from_rubrics + (1-alpha)*overall_direct)

    return {
        "TR": tr_pred, "CC": cc_pred, "LR": lr_pred, "GR": gr_pred,
        "overall_direct": overall_direct,
        "overall_from_rubrics": overall_from_rubrics,
        "overall_final": overall_final
    }


# ==== Ví dụ chạy thử ====
essay = """
Some people argue that governments should focus on improving public transportation
rather than building new roads. I agree because public transport reduces traffic
congestion and is better for the environment...
"""

print(predict_essay(essay))
