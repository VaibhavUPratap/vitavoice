"""
train_fast.py — Quick-training mode for development / demo purposes.

Uses ONLY the ParkinsonsDrawings dataset (204 images, 28 subjects),
3-fold CV, max 8 epochs — completes in ~5-10 minutes on CPU.

Run the full train.py overnight for production-quality models.

Output: same files as train.py
  backend/handwriting/spiral_model.pth
  backend/handwriting/wave_model.pth
  backend/handwriting/fusion_meta.pkl
  backend/handwriting/training_results.json
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root OR from backend/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_script_dir)
for p in [_script_dir, _backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from handwriting.train import (
    collect_all_images,
    print_summary,
    subject_level_split,
    train_model_with_cv,
    evaluate,
    compute_metrics,
    infer_scores,
    subject_avg_scores,
    build_model,
    HandwritingDataset,
    EVAL_TRANSFORM,
    MODEL_DIR,
    PARKINSONS_DRAWINGS_PD,
    PARKINSONS_DRAWINGS_HEALTHY,
    _classify_parkinsons_drawings_file,
)

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path


def collect_fast_images():
    """Only ParkinsonsDrawings — smallest, cleanest, fully labeled dataset."""
    spiral_records = []
    wave_records   = []

    for folder, label in [(PARKINSONS_DRAWINGS_PD, 1), (PARKINSONS_DRAWINGS_HEALTHY, 0)]:
        for fp in sorted(folder.iterdir()):
            if fp.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            dtype, lbl, sid = _classify_parkinsons_drawings_file(fp, label)
            if dtype == "spiral":
                spiral_records.append((fp, lbl, sid))
            elif dtype == "wave":
                wave_records.append((fp, lbl, sid))

    return spiral_records, wave_records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [FAST MODE] Using device: {device}")
    print("  Dataset: ParkinsonsDrawings only (204 images, 28 subjects)")
    print("  Config : 3-fold CV, max 8 epochs, early stop patience=3\n")

    spiral_records, wave_records = collect_fast_images()

    print(f"  Spiral pool: {len(spiral_records)} images, "
          f"{len({r[2] for r in spiral_records})} subjects")
    print(f"  Wave   pool: {len(wave_records)} images, "
          f"{len({r[2] for r in wave_records})} subjects\n")

    results = {}

    # Subject-level held-out test split
    spiral_trainval, spiral_test = subject_level_split(spiral_records, test_size=0.20, random_seed=42)
    wave_trainval,   wave_test   = subject_level_split(wave_records,   test_size=0.20, random_seed=42)

    print(f"  [SPLIT] Spiral: {len(spiral_trainval)} train/val + {len(spiral_test)} test")
    print(f"  [SPLIT] Wave  : {len(wave_trainval)} train/val + {len(wave_test)} test\n")

    # Train spiral
    spiral_model, spiral_cv = train_model_with_cv(
        spiral_trainval, "SPIRAL", device,
        n_splits=3, batch_size=16, max_epochs=8, early_stop_patience=3
    )
    torch.save(spiral_model.state_dict(), MODEL_DIR / "spiral_model.pth")
    print(f"\n  [SAVED] spiral_model.pth")

    criterion = nn.BCELoss()
    s_dl = DataLoader(HandwritingDataset(spiral_test, EVAL_TRANSFORM), batch_size=16, shuffle=False)
    _, s_labels, s_probs = evaluate(spiral_model, s_dl, criterion, device)
    s_metrics = compute_metrics(s_labels, s_probs)
    s_metrics["n_test_images"] = len(spiral_test)
    print("\n  [SPIRAL] Test metrics:", s_metrics)
    results["spiral"] = {"cv_folds": spiral_cv, "test_metrics": s_metrics}

    # Train wave
    wave_model, wave_cv = train_model_with_cv(
        wave_trainval, "WAVE", device,
        n_splits=3, batch_size=16, max_epochs=8, early_stop_patience=3
    )
    torch.save(wave_model.state_dict(), MODEL_DIR / "wave_model.pth")
    print(f"\n  [SAVED] wave_model.pth")

    w_dl = DataLoader(HandwritingDataset(wave_test, EVAL_TRANSFORM), batch_size=16, shuffle=False)
    _, w_labels, w_probs = evaluate(wave_model, w_dl, criterion, device)
    w_metrics = compute_metrics(w_labels, w_probs)
    w_metrics["n_test_images"] = len(wave_test)
    print("\n  [WAVE] Test metrics:", w_metrics)
    results["wave"] = {"cv_folds": wave_cv, "test_metrics": w_metrics}

    # Meta-model
    print("\n  [META] Training fusion logistic regression ...")
    s_tv_scores = subject_avg_scores(spiral_model, spiral_trainval, device)
    w_tv_scores = subject_avg_scores(wave_model,   wave_trainval,   device)

    common_sids = sorted(set(s_tv_scores) & set(w_tv_scores))
    X_meta, y_meta = [], []
    for sid in common_sids:
        X_meta.append([s_tv_scores[sid][0], w_tv_scores[sid][0]])
        y_meta.append(s_tv_scores[sid][1])

    meta = LogisticRegression(max_iter=200, random_state=42)
    meta.fit(np.array(X_meta), np.array(y_meta))
    joblib.dump(meta, MODEL_DIR / "fusion_meta.pkl")
    print("  [SAVED] fusion_meta.pkl")

    # Combined test eval
    s_test_scores = subject_avg_scores(spiral_model, spiral_test, device)
    w_test_scores = subject_avg_scores(wave_model,   wave_test,   device)
    test_sids = sorted(set(s_test_scores) | set(w_test_scores))
    X_test, y_test = [], []
    for sid in test_sids:
        s = s_test_scores.get(sid, (0.5, None))[0]
        w = w_test_scores.get(sid,   (0.5, None))[0]
        lbl = (s_test_scores.get(sid) or w_test_scores.get(sid))[1]
        X_test.append([s, w]); y_test.append(lbl)

    combined_probs = meta.predict_proba(np.array(X_test))[:, 1]
    c_metrics = compute_metrics(np.array(y_test), combined_probs)
    c_metrics["n_test_subjects"] = len(test_sids)
    print("\n  [COMBINED] Test metrics:", c_metrics)
    results["combined"] = {"test_metrics": c_metrics}
    results["mode"] = "fast (ParkinsonsDrawings only, 3-fold, 8 epochs)"

    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  [SAVED] training_results.json")
    print("\n  ✅ Fast training complete — all 3 model files saved.")
    print("     Run train.py overnight for full production-quality models.\n")

    return spiral_test, wave_test


if __name__ == "__main__":
    main()
