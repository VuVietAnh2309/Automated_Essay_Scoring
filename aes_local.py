# -*- coding: utf-8 -*-
"""
Local AES pipeline for train.csv (essay_id, full_text, score)
- Overall (holistic): DeBERTa/Transformer embedding + LightGBM (KFold) => overall_direct
- Inference demo: predict 1 sample
- Proxy rubric heads (TR/CC/LR/GR) using same features => overall_from_rubrics (Prep)
- Blend => overall_final = alpha * overall_from_rubrics + (1-alpha) * overall_direct_rounded

Usage:
  python aes_local.py --train_path train.csv \
                      --model_name microsoft/deberta-v3-base \
                      --n_splits 5 --alpha 0.7

If model too heavy, use: --model_name sentence-transformers/all-MiniLM-L6-v2
"""

import os, math, random, re, argparse, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb

import torch
from transformers import AutoTokenizer, AutoModel


# -------------------- Utils --------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

def round_to_half(x: float) -> float:
    return float(np.round(x*2)/2.0)

def compute_overall_prep(tr:int, cc:int, lr:int, gr:int) -> float:
    """overall = floor((tr + cc + lr + gr)/2) / 2  -> .0 or .5"""
    raw = (tr + cc + lr + gr) / 2.0
    return (int(math.floor(raw))) / 2.0

def to_band_1_9(x: float) -> int:
    """Clamp & round to integer band 1..9"""
    return int(np.clip(np.rint(x), 1, 9))


# -------------------- Lightweight text features for LGBM --------------------
CONNECTIVES = ["however","therefore","moreover","furthermore","in addition",
               "on the other hand","for example","in conclusion","consequently"]

def _tok(s: str):
    return re.findall(r"[A-Za-z']+", str(s).lower())

def cheap_feats(text: str):
    essay = str(text or "")
    sents = [x.strip() for x in re.split(r"[.!?]+", essay) if x.strip()]
    ws = len(_tok(essay))
    avg_len = np.mean([len(_tok(s)) for s in sents]) if sents else 0.0
    conn = sum(essay.lower().count(c) for c in CONNECTIVES)
    ttr = len(set(_tok(essay))) / (len(_tok(essay)) + 1e-9)
    # you can add more features here if you want
    return np.array([ws, avg_len, conn, ttr], dtype=float)


# -------------------- Encoder --------------------
class TextEncoder:
    """
    Encode only full_text (no prompt in your dataset).
    If you later add prompt, you can concat: f"[PROMPT] {prompt} [ESSAY] {text}"
    """
    def __init__(self, model_name: str, device: str = None, max_len: int = 512):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_len = max_len

    @torch.no_grad()
    def encode_batch(self, texts, batch_size=8, pool="cls"):
        out_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i+batch_size]]
            inputs = self.tok(batch, truncation=True, max_length=self.max_len,
                              padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs).last_hidden_state  # [B, T, H]
            if pool == "cls":
                vec = outputs[:, 0, :]               # [B, H]
            else:
                # mean-pooling (mask aware)
                attn = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                vec = (outputs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
            out_vecs.append(vec.detach().cpu().numpy())
        return np.vstack(out_vecs)


# -------------------- KFold LGBM wrapper --------------------
def kfold_lgbm(X, y, n_splits=5, verbose=True, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float); models=[]
    for i, (tr, va) in enumerate(kf.split(X), 1):
        m = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.02, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=seed
        )
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              eval_metric="rmse", verbose=False)
        oof[va] = m.predict(X[va])
        models.append(m)
        if verbose:
            rmse_f = mean_squared_error(y[va], oof[va], squared=False)
            print(f"Fold {i} RMSE: {rmse_f:.4f}")
    rmse = mean_squared_error(y, oof, squared=False)
    pr, sp = pearsonr(y, oof)[0], spearmanr(y, oof)[0]
    if verbose:
        print(f"OOF  RMSE={rmse:.4f}  Pearson={pr:.4f}  Spearman={sp:.4f}")
    return models, oof, {"rmse":rmse, "pearson":pr, "spearman":sp}


# -------------------- Main pipeline --------------------
def main(args):
    # 1) Load data
    df = pd.read_csv(args.train_path)
    for c in ["essay_id", "full_text", "score"]:
        assert c in df.columns, f"train.csv must contain column: {c}"
    texts = df["full_text"].astype(str).tolist()
    y = df["score"].astype(float).values

    # 2) Encode text + cheap features
    print("Loading encoder:", args.model_name)
    enc = TextEncoder(args.model_name, max_len=args.max_len)
    print("Encoding texts ...")
    E = enc.encode_batch(texts, batch_size=args.batch_size, pool=args.pool)  # (N, H)
    F = np.vstack([cheap_feats(t) for t in texts])                            # (N, 4)
    X = np.hstack([E, F])                                                    # (N, H+4)
    print("X shape:", X.shape, " y:", y.shape)

    # 3) Train holistic (overall_direct)
    print("\n=== Train overall_direct (DeBERTa + LGBM) ===")
    models_overall, oof_overall, metrics_overall = kfold_lgbm(
        X, y, n_splits=args.n_splits, verbose=True, seed=args.seed
    )

    # 4) Inference demo: pick one
    ridx = random.randrange(len(df))
    sample_text = df.loc[ridx, "full_text"]
    print("\n=== Inference sample (overall_direct) ===")
    v = enc.encode_batch([sample_text], batch_size=1, pool=args.pool)
    f = cheap_feats(sample_text).reshape(1, -1)
    x = np.hstack([v, f])
    preds = [m.predict(x)[0] for m in models_overall]
    overall_direct_raw = float(np.mean(preds))
    overall_direct = round_to_half(overall_direct_raw)
    print(f"Sample essay_id={int(df.loc[ridx,'essay_id'])}  GT={df.loc[ridx,'score']:.3f}  "
          f"overall_direct_raw={overall_direct_raw:.3f}  rounded={overall_direct:.1f}")

    # 5) Train 4 proxy rubric heads (TR/CC/LR/GR) using same (X,y)
    print("\n=== Train 4 proxy rubric heads (TR/CC/LR/GR) on holistic target ===")
    rubric_models = {}
    for name in ["tr","cc","lr","gr"]:
        print(f"[Head] {name}")
        ms, oof, _ = kfold_lgbm(X, y, n_splits=args.n_splits, verbose=False, seed=args.seed)
        rmse = mean_squared_error(y, oof, squared=False)
        pr, sp = pearsonr(y, oof)[0], spearmanr(y, oof)[0]
        print(f"  OOF RMSE={rmse:.4f}  Pearson={pr:.4f}  Spearman={sp:.4f}")
        rubric_models[name] = ms

    # 6) Inference sample for rubric heads
    def predict_avg(x, models): return float(np.mean([m.predict(x)[0] for m in models]))
    tr_hat = to_band_1_9(predict_avg(x, rubric_models["tr"]))
    cc_hat = to_band_1_9(predict_avg(x, rubric_models["cc"]))
    lr_hat = to_band_1_9(predict_avg(x, rubric_models["lr"]))
    gr_hat = to_band_1_9(predict_avg(x, rubric_models["gr"]))
    overall_from_rubrics = compute_overall_prep(tr_hat, cc_hat, lr_hat, gr_hat)

    print("\n=== Rubric heads (sample) ===")
    print({"tr": tr_hat, "cc": cc_hat, "lr": lr_hat, "gr": gr_hat,
           "overall_from_rubrics": overall_from_rubrics})

    # 7) Blend to overall_final
    overall_final = round_to_half(args.alpha * overall_from_rubrics + (1 - args.alpha) * overall_direct)
    print("\n=== overall_final (blend) ===")
    print({
        "overall_direct": overall_direct,
        "overall_from_rubrics": overall_from_rubrics,
        "alpha": args.alpha,
        "overall_final": overall_final
    })

    # 8) (Optional) Save models to disk
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        import pickle
        with open(os.path.join(args.save_dir, "overall_models.pkl"), "wb") as f:
            pickle.dump(models_overall, f)
        with open(os.path.join(args.save_dir, "rubric_models.pkl"), "wb") as f:
            pickle.dump(rubric_models, f)
        np.save(os.path.join(args.save_dir, "feature_meanvec_shape.npy"), np.array(X.shape[1]))
        print(f"\nSaved models to: {args.save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="train.csv")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base",
                   help="e.g., microsoft/deberta-v3-base or sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--pool", type=str, default="cls", choices=["cls","mean"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_arg_
