import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================
# CẤU HÌNH
# =========================
# Model IELTS_essay_scoring đã bị xoá weights -> sẽ thử load và fallback sang Engessay_grading_ML
PRIMARY_MODEL = "KevSun/IELTS_essay_scoring"
FALLBACK_MODEL = "KevSun/Engessay_grading_ML"

# Essay mẫu (bạn thay nội dung nếu muốn)
ESSAY = """
Some people believe that technology has made our lives more complicated, while others argue that it has simplified daily tasks.
In my opinion, although modern devices sometimes cause distraction, the overall impact of technology has been positive.
Firstly, communication has become faster and cheaper through emails, messaging apps, and video calls.
Secondly, automation in workplaces increases productivity and reduces human error.
Finally, access to information on the Internet allows people to learn new skills more easily than ever before.
Therefore, I strongly believe that technology has simplified our daily lives more than it has complicated them.
"""

def load_or_fallback():
    """
    Thử load IELTS_essay_scoring. Nếu thiếu weights -> fallback sang Engessay_grading_ML.
    Trả về (tokenizer, model, mode), trong đó mode = "ielts" hoặc "engessay".
    """
    try:
        tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(PRIMARY_MODEL)
        mdl.eval()
        return tok, mdl, "ielts"
    except Exception as e:
        print(f"[INFO] Không tải được {PRIMARY_MODEL} (có thể thiếu weights). Fallback sang {FALLBACK_MODEL}.")
        tok = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
        mdl.eval()
        return tok, mdl, "engessay"

def score_ielts(text, tokenizer, model, max_length=512):
    """
    Chuẩn hoá theo README IELTS: (logits / max(logits)) * 9, làm tròn 0.5 band.
    Trả về dict 5 tiêu chí.
    """
    items = ["Task Achievement", "Coherence and Cohesion", "Vocabulary", "Grammar", "Overall"]
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze().cpu().numpy()
    normalized = (logits / logits.max()) * 9.0
    rounded = np.round(normalized * 2) / 2
    return dict(zip(items, rounded.tolist()))

def score_engessay(text, tokenizer, model, max_length=512, scale="1to5"):
    """
    Theo model card Engessay:
    - scale="1to5": min-max theo vector logits từng bài, rồi map vào [1,5], làm tròn 0.5
    - scale="1to10": công thức tuyến tính 2.25 * logits - 1.25, làm tròn 0.5
    Trả về dict 6 tiêu chí.
    """
    items = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze().cpu().numpy()

    if scale == "1to10":
        scaled = 2.25 * logits - 1.25
        rounded = [round(x * 2) / 2 for x in scaled]
    else:
        # map vào [1,5]
        mn, mx = float(np.min(logits)), float(np.max(logits))
        if mx - mn < 1e-8:
            # phòng TH mọi logit bằng nhau
            scaled = np.full_like(logits, 3.0, dtype=float)
        else:
            scaled = 1 + 4 * (logits - mn) / (mx - mn)
        rounded = np.round(scaled * 2) / 2

    return dict(zip(items, [float(v) for v in rounded]))

if __name__ == "__main__":
    tokenizer, model, mode = load_or_fallback()

    if mode == "ielts":
        scores = score_ielts(ESSAY, tokenizer, model)
        print("— IELTS Essay Scores (0–9, round 0.5) —")
        for k, v in scores.items():
            print(f"{k}: {v:.1f}")
    else:
        scores = score_engessay(ESSAY, tokenizer, model, scale="1to5")
        print("— Engessay Scores (6 tiêu chí, 1–5, round 0.5) —")
        for k, v in scores.items():
            print(f"{k}: {v:.1f}")
