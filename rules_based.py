# -*- coding: utf-8 -*-
"""
IELTS Task 2 Heuristic Scoring (1 file)
- Tính TR/CC/LR/GR (integer 1..9) từ prompt + essay bằng đặc trưng NLP đơn giản
- Tính Overall theo Prep: floor((tr+cc+lr+gr)/2)/2  → .0 hoặc .5
- Ánh xạ điểm sang nhóm band (1–4, 4.5–5.5, 6–7.5, 8–9) + feedback ngắn

Lưu ý: Đây là bản proxy để demo/inference nhanh, không thay thế giám khảo IELTS.
Bạn có thể tinh chỉnh trọng số/ngưỡng ở các hàm score_* để hợp dữ liệu thực tế.
"""

from __future__ import annotations
import re, math
from typing import Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== Tiện ích cơ bản =====================
CONNECTIVES = [
    "however","therefore","moreover","furthermore","in addition","nevertheless",
    "on the other hand","for example","for instance","in contrast","as a result",
    "consequently","firstly","secondly","finally","in conclusion","to conclude"
]
STOPWORDS = set("""
a an the and or but if while of in on at for from by to with as that this those these
is are was were be been being it its they them he she we you your our their i me my
""".split())

def sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def tokens(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())

def n_words(text: str) -> int:
    return len(tokens(text))

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def map_0_1_to_band9(x: float) -> int:
    """Map linear 0..1 → 1..9 (integer)."""
    x = clip(x, 0.0, 1.0)
    return int(round(1 + x * 8))

def scale_to_0_1(x, lo, hi):
    if hi == lo: 
        return 0.0
    return clip((x - lo) / float(hi - lo), 0.0, 1.0)

# ===================== Đặc trưng theo rubric =====================
def feats_task_response(prompt: str, essay: str) -> Dict[str, float]:
    # TF-IDF similarity + keyword coverage (loại stopwords) + độ dài phù hợp (Task 2 ~ 250 từ)
    vect = TfidfVectorizer(min_df=1, ngram_range=(1,1))
    X = vect.fit_transform([prompt, essay])
    sim = float(cosine_similarity(X[0], X[1])[0, 0])

    p_kw = {t for t in tokens(prompt) if t not in STOPWORDS}
    e_toks = set(tokens(essay))
    keyword_cov = len(p_kw & e_toks) / (len(p_kw) + 1e-9)

    L = n_words(essay)
    len_score = scale_to_0_1(L, lo=160, hi=320)  # <160 phạt nhẹ, >320 không cộng thêm

    return {"sim": sim, "kw_cov": keyword_cov, "len_score": len_score, "len_words": float(L)}

def feats_coherence(essay: str) -> Dict[str, float]:
    sents = sentences(essay)
    ns = len(sents)
    avg_len = np.mean([len(tokens(s)) for s in sents]) if ns else 0.0
    paras = [p for p in essay.split("\n") if p.strip()]
    n_paras = len(paras) if paras else 1

    essay_l = essay.lower()
    conn_count = sum(essay_l.count(c) for c in CONNECTIVES)
    words = max(1, n_words(essay))
    conn_per_100 = (conn_count / words) * 100.0

    ns_s     = scale_to_0_1(ns,        lo=4,   hi=10)   # 4..10 câu
    avlen_s  = scale_to_0_1(avg_len,   lo=12,  hi=25)   # 12..25 từ/câu
    paras_s  = scale_to_0_1(n_paras,   lo=3,   hi=6)    # 3..6 đoạn
    conn_s   = scale_to_0_1(conn_per_100, lo=0.5, hi=3.0)

    return {"ns_s": ns_s, "avlen_s": avlen_s, "paras_s": paras_s, "conn_s": conn_s}

def feats_lexical(essay: str) -> Dict[str, float]:
    toks = tokens(essay)
    types = set(toks)
    ttr = len(types) / (len(toks) + 1e-9)
    long_ratio = sum(1 for w in toks if len(w) >= 7) / (len(toks) + 1e-9)
    bigrams = list(zip(toks, toks[1:]))
    uniq_bi = len(set(bigrams))
    bi_div = uniq_bi / (len(bigrams) + 1e-9)

    ttr_s   = scale_to_0_1(ttr,        lo=0.30, hi=0.60)
    long_s  = scale_to_0_1(long_ratio, lo=0.05, hi=0.25)
    bidiv_s = scale_to_0_1(bi_div,     lo=0.60, hi=0.90)

    return {"ttr_s": ttr_s, "long_s": long_s, "bidiv_s": bidiv_s}

def feats_grammar(essay: str) -> Dict[str, float]:
    sents = sentences(essay)
    long_ratio = np.mean([len(tokens(s)) >= 20 for s in sents]) if sents else 0.0
    cap_ok = np.mean([bool(re.match(r"\s*[A-Z]", s.strip())) for s in sents]) if sents else 0.0
    weird_repeat = len(re.findall(r"(.)\1\1+", essay.lower()))
    weird_penalty = 1.0 - scale_to_0_1(weird_repeat, lo=0, hi=5)

    L = max(1, len(essay))
    comma_ratio = essay.count(",") / L
    semicol_ratio = essay.count(";") / L
    comma_s   = 1.0 - scale_to_0_1(comma_ratio,   lo=0.010, hi=0.030)
    semicol_s = 1.0 - scale_to_0_1(semicol_ratio, lo=0.001, hi=0.010)

    long_s  = scale_to_0_1(long_ratio, lo=0.10, hi=0.60)
    cap_s   = scale_to_0_1(cap_ok,     lo=0.80, hi=1.00)

    return {
        "long_s": long_s,
        "cap_s": cap_s,
        "comma_s": clip(comma_s, 0.0, 1.0),
        "semicol_s": clip(semicol_s, 0.0, 1.0),
        "weird_s": clip(weird_penalty, 0.0, 1.0),
    }

# ===================== 4 hàm chấm band (1..9) =====================
def score_tr(prompt: str, essay: str) -> int:
    f = feats_task_response(prompt, essay)
    # Trọng số: similarity 0.5, keyword 0.3, length 0.2
    score_0_1 = (0.5*f["sim"] + 0.3*f["kw_cov"] + 0.2*f["len_score"])
    return map_0_1_to_band9(score_0_1)

def score_cc(prompt: str, essay: str) -> int:
    f = feats_coherence(essay)
    score_0_1 = (0.30*f["ns_s"] + 0.30*f["avlen_s"] + 0.20*f["paras_s"] + 0.20*f["conn_s"])
    return map_0_1_to_band9(score_0_1)

def score_lr(prompt: str, essay: str) -> int:
    f = feats_lexical(essay)
    score_0_1 = (0.40*f["ttr_s"] + 0.30*f["long_s"] + 0.30*f["bidiv_s"])
    return map_0_1_to_band9(score_0_1)

def score_gr(prompt: str, essay: str) -> int:
    f = feats_grammar(essay)
    score_0_1 = (0.30*f["long_s"] + 0.25*f["cap_s"] + 0.25*min(f["comma_s"], f["semicol_s"]) + 0.20*f["weird_s"])
    return map_0_1_to_band9(score_0_1)

# ===================== Overall (Prep) =====================
def compute_overall_prep(tr: int, cc: int, lr: int, gr: int) -> float:
    raw = (tr + cc + lr + gr) / 2.0
    return (int(math.floor(raw))) / 2.0  # .0 hoặc .5

# ===================== Nhóm band + feedback =====================
def band_group(score: int) -> str:
    if score <= 4:
        return "1–4"
    elif score <= 5:
        return "4.5–5.5"
    elif score <= 7:
        return "6–7.5"
    else:
        return "8–9"

FEEDBACK = {
    "TR": {
        "1–4":   "Chưa đáp ứng yêu cầu đề; ý rời rạc hoặc quá ngắn.",
        "4.5–5.5":"Trả lời một phần đề; phát triển ý còn hạn chế.",
        "6–7.5": "Trả lời đầy đủ; quan điểm rõ; có lý do/ví dụ nhưng chưa thật sâu.",
        "8–9":   "Hoàn toàn đáp ứng đề; lập luận sâu, ví dụ thuyết phục."
    },
    "CC": {
        "1–4":   "Mạch lạc yếu; thiếu/dùng sai từ nối; khó theo dõi.",
        "4.5–5.5":"Bố cục cơ bản; liên kết còn gượng/lặp.",
        "6–7.5": "Bố cục rõ; cohesive devices phù hợp; đôi chỗ lặp.",
        "8–9":   "Mạch lạc tự nhiên; chuyển đoạn linh hoạt; liên kết mượt."
    },
    "LR": {
        "1–4":   "Vốn từ hạn chế; nhiều lỗi dùng từ; lặp từ.",
        "4.5–5.5":"Từ vựng đủ dùng nhưng hạn chế; có lỗi gây hiểu sai.",
        "6–7.5": "Từ vựng khá đa dạng; có collocation/paraphrase; ít lỗi.",
        "8–9":   "Từ vựng rộng, tự nhiên, chính xác; collocation phong phú."
    },
    "GR": {
        "1–4":   "Cấu trúc rất đơn giản; nhiều lỗi nặng.",
        "4.5–5.5":"Có thêm cấu trúc nhưng lỗi còn nhiều.",
        "6–7.5": "Đa dạng cấu trúc cơ bản–phức; lỗi không làm khó hiểu.",
        "8–9":   "Cấu trúc đa dạng, chính xác; lỗi hiếm."
    }
}

def explain_rubrics(tr:int, cc:int, lr:int, gr:int):
    scores = {"TR": tr, "CC": cc, "LR": lr, "GR": gr}
    out = {}
    for k, v in scores.items():
        g = band_group(v)
        out[k] = {"score": v, "group": g, "feedback": FEEDBACK[k][g]}
    return out

# ===================== Demo chạy nhanh =====================
if __name__ == "__main__":
    prompt = ("Some people think governments should invest more in public transportation "
              "rather than road infrastructure. To what extent do you agree or disagree?")
    essay = ("In recent years, urban congestion has become a pressing issue. While some argue for "
             "expanding roads, I believe funding public transport is more sustainable. Firstly, "
             "buses and subways move more people efficiently; moreover, they reduce emissions. "
             "Secondly, better transit encourages citizens to leave cars at home. However, roads "
             "are still necessary for logistics. In conclusion, governments should prioritize "
             "public transportation.")

    tr = score_tr(prompt, essay)
    cc = score_cc(prompt, essay)
    lr = score_lr(prompt, essay)
    gr = score_gr(prompt, essay)
    overall = compute_overall_prep(tr, cc, lr, gr)
    explain = explain_rubrics(tr, cc, lr, gr)

    print({"tr": tr, "cc": cc, "lr": lr, "gr": gr, "overall": overall})
    print(explain)
