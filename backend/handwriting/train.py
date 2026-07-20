"""
train.py — Handwriting-based Parkinson's Disease screening module.

Trains TWO separate ResNet18 models (spiral and wave drawings) using
transfer learning, then fuses their outputs with a logistic-regression
meta-model. Subject-level stratified splits prevent data leakage.

Datasets consumed (paths are relative to repo root):
  datasets/handwriting/data/ParkinsonsDrawings/pd/       PD drawings
  datasets/handwriting/data/ParkinsonsDrawings/healthy/  Healthy drawings
  datasets/handwriting/data/HandPD/PD/                   PD (untyped, used in both)
  datasets/handwriting/data/HandPD/Healthy/              Healthy (untyped)
  datasets/handwriting/data/NewHandPD/PD/                PD (partially typed)
  datasets/handwriting/data/NewHandPD/Healthy/           Healthy (partially typed)

Output files (written to backend/handwriting/):
  spiral_model.pth        ResNet18 fine-tuned on spiral drawings
  wave_model.pth          ResNet18 fine-tuned on wave drawings
  fusion_meta.pkl         Logistic-regression meta-model
  training_results.json   Per-model and combined test-set metrics
"""

from __future__ import annotations

import os
import sys
import re
import ssl
import json
import copy
import random
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import joblib
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent           # backend/handwriting/
BACKEND_DIR  = SCRIPT_DIR.parent                         # backend/
REPO_ROOT    = BACKEND_DIR.parent                        # vitavoice/

DATA_ROOT = REPO_ROOT / "datasets" / "handwriting" / "data"
PARKINSONS_DRAWINGS_PD      = DATA_ROOT / "ParkinsonsDrawings" / "pd"
PARKINSONS_DRAWINGS_HEALTHY = DATA_ROOT / "ParkinsonsDrawings" / "healthy"
HANDPD_PD      = DATA_ROOT / "HandPD" / "PD"
HANDPD_HEALTHY = DATA_ROOT / "HandPD" / "Healthy"
NEWHANDPD_PD      = DATA_ROOT / "NewHandPD" / "PD"
NEWHANDPD_HEALTHY = DATA_ROOT / "NewHandPD" / "Healthy"

MODEL_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# SSL workaround — macOS Python 3.13 cert verification issue
# ---------------------------------------------------------------------------
_original_https_context = ssl._create_default_https_context  # noqa: SLF001

def _ssl_unverified_context(*args, **kwargs):
    return ssl._create_unverified_context(*args, **kwargs)  # noqa: SLF001

def _patch_ssl():
    ssl._create_default_https_context = _ssl_unverified_context  # noqa: SLF001

def _restore_ssl():
    ssl._create_default_https_context = _original_https_context  # noqa: SLF001


# ---------------------------------------------------------------------------
# Image type detection helpers
# ---------------------------------------------------------------------------
def _classify_parkinsons_drawings_file(
    filepath: Path, label: int
) -> Tuple[str, int, str]:
    """
    Returns (drawing_type, label, subject_id) for ParkinsonsDrawings.
    drawing_type ∈ {'spiral', 'wave', 'unknown'}
    """
    stem = filepath.stem.upper()
    # PD files: V{id}PE{seq} or V{id}PO{seq}
    # Healthy:  V{id}HE{seq} or V{id}HO{seq}
    m = re.match(r"V(\d+)([A-Z])([EO])", stem)
    if m:
        subject_id = m.group(1).zfill(3)
        code_char = m.group(2)
        char = m.group(3)
        dtype = "spiral" if char == "E" else "wave"
        prefix = "PD" if code_char == "P" else "H"
        return dtype, label, f"{prefix}_{subject_id}"
    # Fallback: try HE/HO pattern with single-digit seq (V03HE1)
    m2 = re.match(r"V(\d+)([A-Z]{2})\d", stem)
    if m2:
        subject_id = m2.group(1).zfill(3)
        code = m2.group(2)
        dtype = "spiral" if code[1] == "E" else "wave"
        prefix = "PD" if code[0] in ("P",) else "H"
        return dtype, label, f"{prefix}_{subject_id}"
    return "unknown", label, f"UNK_{stem}"


def _classify_handpd_file(
    filepath: Path, label: int, dataset_prefix: str
) -> Tuple[str, int, str]:
    """
    HandPD: {subject_id}-{seq}.jpg  — no type tag → both pools.
    Returns ('both', label, subject_id)
    """
    stem = filepath.stem
    m = re.match(r"^(\d+)-\d+$", stem)
    if m:
        subject_id = m.group(1)
        return "both", label, f"{dataset_prefix}_{subject_id}"
    return "both", label, f"{dataset_prefix}_UNK_{stem}"


def _classify_newhandpd_file(
    filepath: Path, label: int
) -> Tuple[str, int, str]:
    """
    NewHandPD naming conventions:
      {id}-{seq}.jpg        → both pools, subject = id
      sp{N}-{H/P}{seq}.jpg  → spiral
      mea{N}-{H/P}{seq}.jpg → wave
      circA-{H/P}{seq}.jpg  → spiral
    """
    stem = filepath.stem

    # Numeric: 0002-1 etc.
    if re.match(r"^\d{4}-\d+$", stem):
        m = re.match(r"^(\d+)-\d+$", stem)
        return "both", label, f"NHPD_{m.group(1)}"

    # sp{N}-P{seq} or sp{N}-H{seq}
    m = re.match(r"^(sp\d+)-[HP](\d+)$", stem)
    if m:
        batch = m.group(1)
        seq   = m.group(2)
        return "spiral", label, f"NHPD_{batch}_{seq}"

    # mea{N}-P{seq} or mea{N}-H{seq}
    m = re.match(r"^(mea\d+)-[HP](\d+)$", stem)
    if m:
        batch = m.group(1)
        seq   = m.group(2)
        return "wave", label, f"NHPD_{batch}_{seq}"

    # circA-P{seq} or circA-H{seq} (circle → treated as spiral)
    m = re.match(r"^(circA)-[HP](\d+)$", stem, re.IGNORECASE)
    if m:
        seq = m.group(2)
        return "spiral", label, f"NHPD_circA_{seq}"

    # Fallback
    return "both", label, f"NHPD_UNK_{stem}"


# ---------------------------------------------------------------------------
# Collect all images into records
# ---------------------------------------------------------------------------
ImageRecord = Tuple[Path, int, str]   # (path, label, subject_id)

def collect_all_images() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """
    Returns (spiral_records, wave_records).
    Each record is (image_path, label, subject_id).
    label = 1 (PD), 0 (Healthy).
    """
    spiral_records: List[ImageRecord] = []
    wave_records:   List[ImageRecord] = []

    def add(dtype: str, label: int, subject_id: str, path: Path):
        if dtype == "spiral":
            spiral_records.append((path, label, subject_id))
        elif dtype == "wave":
            wave_records.append((path, label, subject_id))
        elif dtype == "both":
            spiral_records.append((path, label, subject_id))
            wave_records.append((path, label, subject_id))

    # ── ParkinsonsDrawings ──────────────────────────────────────────────
    for folder, label in [(PARKINSONS_DRAWINGS_PD, 1), (PARKINSONS_DRAWINGS_HEALTHY, 0)]:
        if not folder.exists():
            print(f"  [WARN] Folder not found: {folder}")
            continue
        for fp in sorted(folder.iterdir()):
            if fp.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            dtype, lbl, sid = _classify_parkinsons_drawings_file(fp, label)
            add(dtype, lbl, sid, fp)

    # ── HandPD ──────────────────────────────────────────────────────────
    for folder, label in [(HANDPD_PD, 1), (HANDPD_HEALTHY, 0)]:
        if not folder.exists():
            print(f"  [WARN] Folder not found: {folder}")
            continue
        for fp in sorted(folder.iterdir()):
            if fp.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            dtype, lbl, sid = _classify_handpd_file(fp, label, "HPD")
            add(dtype, lbl, sid, fp)

    # ── NewHandPD ────────────────────────────────────────────────────────
    for folder, label in [(NEWHANDPD_PD, 1), (NEWHANDPD_HEALTHY, 0)]:
        if not folder.exists():
            print(f"  [WARN] Folder not found: {folder}")
            continue
        for fp in sorted(folder.iterdir()):
            if fp.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            dtype, lbl, sid = _classify_newhandpd_file(fp, label)
            add(dtype, lbl, sid, fp)

    return spiral_records, wave_records


def print_summary(spiral_records: List[ImageRecord], wave_records: List[ImageRecord]):
    """Step 7 — Print pre-training data summary."""
    print("\n" + "=" * 70)
    print("  DATA SUMMARY — Handwriting Parkinson's Screening Module")
    print("=" * 70)

    for name, records in [("SPIRAL", spiral_records), ("WAVE", wave_records)]:
        labels     = [r[1] for r in records]
        subjects   = set(r[2] for r in records)
        n_total    = len(records)
        n_pd       = sum(labels)
        n_healthy  = n_total - n_pd
        n_subjects = len(subjects)
        pd_subjs   = len({r[2] for r in records if r[1] == 1})
        hy_subjs   = len({r[2] for r in records if r[1] == 0})

        print(f"\n  {name} model pool:")
        print(f"    Total images   : {n_total}")
        print(f"    PD images      : {n_pd}")
        print(f"    Healthy images : {n_healthy}")
        print(f"    Unique subjects: {n_subjects}  (PD={pd_subjs}, Healthy={hy_subjs})")

        if n_pd > 0 and n_healthy > 0:
            ratio = n_pd / n_healthy
            print(f"    PD:Healthy ratio: {ratio:.2f}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# ImageNet stats for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class HandwritingDataset(Dataset):
    """Loads a list of ImageRecord tuples with the given transform."""

    def __init__(self, records: List[ImageRecord], transform=None):
        self.records   = records
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label, _ = self.records[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(device: torch.device) -> nn.Module:
    """
    ResNet18 pretrained on ImageNet.
    Freezes all layers except layer4 + replaces FC with custom head.
    Downloads weights with SSL verification disabled (macOS workaround).
    """
    _patch_ssl()
    try:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    finally:
        _restore_ssl()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    return model.to(device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (val_loss, true_labels, pred_probs)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        probs = model(imgs)
        loss  = criterion(probs, labels)
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().squeeze(1).numpy())
        all_probs.extend(probs.cpu().squeeze(1).numpy())
    return (
        total_loss / len(loader.dataset),
        np.array(all_labels),
        np.array(all_probs),
    )


def compute_metrics(true_labels: np.ndarray, pred_probs: np.ndarray) -> Dict:
    pred_bin = (pred_probs >= 0.5).astype(int)
    return {
        "accuracy":  float(accuracy_score(true_labels, pred_bin)),
        "precision": float(precision_score(true_labels, pred_bin, zero_division=0)),
        "recall":    float(recall_score(true_labels, pred_bin, zero_division=0)),
        "f1":        float(f1_score(true_labels, pred_bin, zero_division=0)),
        "auc":       float(roc_auc_score(true_labels, pred_probs))
                     if len(np.unique(true_labels)) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Subject-level split helpers
# ---------------------------------------------------------------------------
def subject_level_split(
    records: List[ImageRecord],
    test_size: float = 0.20,
    random_seed: int = 42,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """
    Splits records into train/val pool and held-out test by SUBJECT ID.
    Stratified by label (PD/healthy) at the subject level.
    """
    subjects_pd      = sorted({r[2] for r in records if r[1] == 1})
    subjects_healthy = sorted({r[2] for r in records if r[1] == 0})

    n_test_pd  = max(1, int(len(subjects_pd) * test_size))
    n_test_hy  = max(1, int(len(subjects_healthy) * test_size))

    rng = random.Random(random_seed)
    test_pd_subs  = set(rng.sample(subjects_pd, n_test_pd))
    test_hy_subs  = set(rng.sample(subjects_healthy, n_test_hy))
    test_subs = test_pd_subs | test_hy_subs

    train_records = [r for r in records if r[2] not in test_subs]
    test_records  = [r for r in records if r[2] in test_subs]
    return train_records, test_records


# ---------------------------------------------------------------------------
# Cross-validation training loop
# ---------------------------------------------------------------------------
def train_model_with_cv(
    records: List[ImageRecord],
    model_name: str,
    device: torch.device,
    n_splits: int = 5,
    batch_size: int = 32,
    max_epochs: int = 40,
    lr: float = 1e-4,
    early_stop_patience: int = 5,
    seed: int = 42,
) -> Tuple[nn.Module, List[Dict]]:
    """
    Runs StratifiedGroupKFold CV on train_records.
    Returns the best model (lowest fold val loss) and per-fold metrics.
    """
    print(f"\n  [{model_name}] Starting 5-fold cross-validation ...")

    paths    = np.array([str(r[0]) for r in records])
    labels   = np.array([r[1]     for r in records])
    groups   = np.array([r[2]     for r in records])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    criterion = nn.BCELoss()
    fold_results  = []
    best_val_loss = float("inf")
    best_model_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(paths, labels, groups), start=1
    ):
        print(f"\n    Fold {fold_idx}/{n_splits}  "
              f"(train={len(train_idx)}, val={len(val_idx)} images)")

        train_recs = [records[i] for i in train_idx]
        val_recs   = [records[i] for i in val_idx]

        train_ds = HandwritingDataset(train_recs, TRAIN_TRANSFORM)
        val_ds   = HandwritingDataset(val_recs,   EVAL_TRANSFORM)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

        model     = build_model(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_fold_val = float("inf")
        patience_ctr  = 0
        best_fold_state = None

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
            val_loss, val_labels, val_probs = evaluate(model, val_dl, criterion, device)
            scheduler.step(val_loss)

            if val_loss < best_fold_val:
                best_fold_val   = val_loss
                best_fold_state = copy.deepcopy(model.state_dict())
                patience_ctr    = 0
            else:
                patience_ctr += 1

            print(f"      Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f}  patience={patience_ctr}/{early_stop_patience}")

            if patience_ctr >= early_stop_patience:
                print(f"      → Early stop at epoch {epoch}")
                break

        model.load_state_dict(best_fold_state)
        _, val_labels, val_probs = evaluate(model, val_dl, criterion, device)
        fold_metrics = compute_metrics(val_labels, val_probs)
        fold_metrics["fold"]     = fold_idx
        fold_metrics["val_loss"] = best_fold_val
        fold_results.append(fold_metrics)

        print(f"    Fold {fold_idx} best val_loss={best_fold_val:.4f}  "
              f"AUC={fold_metrics['auc']:.3f}  F1={fold_metrics['f1']:.3f}")

        if best_fold_val < best_val_loss:
            best_val_loss    = best_fold_val
            best_model_state = best_fold_state

    # Load best fold's weights into a fresh model instance
    final_model = build_model(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    avg = {
        k: float(np.mean([f[k] for f in fold_results]))
        for k in ("accuracy", "precision", "recall", "f1", "auc")
    }
    print(f"\n  [{model_name}] CV averages: "
          f"Acc={avg['accuracy']:.3f}  AUC={avg['auc']:.3f}  F1={avg['f1']:.3f}")

    return final_model, fold_results


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_scores(
    model: nn.Module,
    records: List[ImageRecord],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Returns predicted probability array for each record."""
    ds = HandwritingDataset(records, EVAL_TRANSFORM)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs = []
    for imgs, _ in dl:
        imgs = imgs.to(device)
        probs = model(imgs).cpu().squeeze(1).numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


# ---------------------------------------------------------------------------
# Subject-level score aggregation (module-level — importable)
# ---------------------------------------------------------------------------
def subject_avg_scores(
    model: nn.Module,
    records: List[ImageRecord],
    device: torch.device,
) -> Dict[str, Tuple[float, int]]:
    """
    Runs inference over all records and returns a dict mapping
    subject_id -> (mean_prob, label).
    """
    all_probs = infer_scores(model, records, device)
    subject_probs: Dict[str, List[float]] = {}
    subject_labels: Dict[str, int] = {}
    for (_, label, sid), prob in zip(records, all_probs):
        subject_probs.setdefault(sid, []).append(float(prob))
        subject_labels[sid] = label
    return {
        sid: (float(np.mean(probs)), subject_labels[sid])
        for sid, probs in subject_probs.items()
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")

    # ── Collect data ──────────────────────────────────────────────────────
    spiral_records, wave_records = collect_all_images()

    # ── Step 7: Pre-training summary ─────────────────────────────────────
    print_summary(spiral_records, wave_records)

    results: Dict = {}

    # ── Subject-level held-out test split ─────────────────────────────────
    spiral_trainval, spiral_test = subject_level_split(spiral_records, test_size=0.20)
    wave_trainval,   wave_test   = subject_level_split(wave_records,   test_size=0.20)

    print(f"  [SPLIT] Spiral: {len(spiral_trainval)} train/val + {len(spiral_test)} test images")
    print(f"  [SPLIT] Wave:   {len(wave_trainval)} train/val + {len(wave_test)} test images")

    # ── Train spiral model ────────────────────────────────────────────────
    spiral_model, spiral_cv_results = train_model_with_cv(
        spiral_trainval, "SPIRAL", device
    )
    spiral_model_path = MODEL_DIR / "spiral_model.pth"
    torch.save(spiral_model.state_dict(), spiral_model_path)
    print(f"\n  [SAVED] Spiral model → {spiral_model_path}")

    # Test-set evaluation
    criterion = nn.BCELoss()
    spiral_test_ds = HandwritingDataset(spiral_test, EVAL_TRANSFORM)
    spiral_test_dl = DataLoader(spiral_test_ds, batch_size=32, shuffle=False, num_workers=0)
    _, s_test_labels, s_test_probs = evaluate(spiral_model, spiral_test_dl, criterion, device)
    spiral_test_metrics = compute_metrics(s_test_labels, s_test_probs)
    spiral_test_metrics["n_test_images"] = len(spiral_test)
    print("\n  [SPIRAL] Held-out test metrics:")
    for k, v in spiral_test_metrics.items():
        print(f"    {k:20s}: {v}")
    results["spiral"] = {
        "cv_folds":    spiral_cv_results,
        "test_metrics": spiral_test_metrics,
    }

    # ── Train wave model ──────────────────────────────────────────────────
    wave_model, wave_cv_results = train_model_with_cv(
        wave_trainval, "WAVE", device
    )
    wave_model_path = MODEL_DIR / "wave_model.pth"
    torch.save(wave_model.state_dict(), wave_model_path)
    print(f"\n  [SAVED] Wave model → {wave_model_path}")

    wave_test_ds = HandwritingDataset(wave_test, EVAL_TRANSFORM)
    wave_test_dl = DataLoader(wave_test_ds, batch_size=32, shuffle=False, num_workers=0)
    _, w_test_labels, w_test_probs = evaluate(wave_model, wave_test_dl, criterion, device)
    wave_test_metrics = compute_metrics(w_test_labels, w_test_probs)
    wave_test_metrics["n_test_images"] = len(wave_test)
    print("\n  [WAVE] Held-out test metrics:")
    for k, v in wave_test_metrics.items():
        print(f"    {k:20s}: {v}")
    results["wave"] = {
        "cv_folds":    wave_cv_results,
        "test_metrics": wave_test_metrics,
    }

    # ── Train meta-model (fusion) ─────────────────────────────────────────
    print("\n  [META] Building fusion meta-model ...")

    # Use subjects from the INTERSECTION of spiral_trainval & wave_trainval for meta training.
    # Strategy: find subjects that appear in BOTH pools → run inference → train LR.
    spiral_tv_subjects = {r[2] for r in spiral_trainval}
    wave_tv_subjects   = {r[2] for r in wave_trainval}
    shared_subjects    = spiral_tv_subjects & wave_tv_subjects

    if len(shared_subjects) < 5:
        print("  [WARN] Very few shared subjects for meta-model; using all train/val records.")
        # Fall back to using all, pairing by index (may have slight mismatch in records)
        # We just want one score per subject — use average over each subject's images
        shared_subjects = spiral_tv_subjects | wave_tv_subjects


    # Collect per-subject averaged scores using module-level helper
    spiral_tv_scores = subject_avg_scores(spiral_model, spiral_trainval, device)
    wave_tv_scores   = subject_avg_scores(wave_model,   wave_trainval,   device)

    # Build feature matrix over subjects present in both
    common_sids = sorted(set(spiral_tv_scores) & set(wave_tv_scores))
    if not common_sids:
        # Use union, filling missing with 0.5 (neutral)
        common_sids = sorted(set(spiral_tv_scores) | set(wave_tv_scores))

    X_meta, y_meta = [], []
    for sid in common_sids:
        s_score = spiral_tv_scores.get(sid, (0.5, None))[0]
        w_score = wave_tv_scores.get(sid,   (0.5, None))[0]
        label   = (spiral_tv_scores.get(sid) or wave_tv_scores.get(sid))[1]
        X_meta.append([s_score, w_score])
        y_meta.append(label)

    X_meta = np.array(X_meta)
    y_meta = np.array(y_meta)

    meta_model = LogisticRegression(max_iter=500, random_state=42)
    meta_model.fit(X_meta, y_meta)

    meta_model_path = MODEL_DIR / "fusion_meta.pkl"
    joblib.dump(meta_model, meta_model_path)
    print(f"  [SAVED] Fusion meta-model → {meta_model_path}")

    # ── Combined test-set evaluation ──────────────────────────────────────
    print("\n  [META] Evaluating combined model on held-out test sets ...")

    # Get per-subject scores on test sets
    spiral_test_scores = subject_avg_scores(spiral_model, spiral_test, device)
    wave_test_scores   = subject_avg_scores(wave_model,   wave_test,   device)

    # Use union of test subjects
    test_sids = sorted(set(spiral_test_scores) | set(wave_test_scores))
    X_test, y_test = [], []
    for sid in test_sids:
        s_score = spiral_test_scores.get(sid, (0.5, None))[0]
        w_score = wave_test_scores.get(sid,   (0.5, None))[0]
        label   = (spiral_test_scores.get(sid) or wave_test_scores.get(sid))[1]
        X_test.append([s_score, w_score])
        y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    combined_probs = meta_model.predict_proba(X_test)[:, 1]
    combined_metrics = compute_metrics(np.array(y_test), combined_probs)
    combined_metrics["n_test_subjects"] = len(test_sids)
    print("\n  [COMBINED] Held-out test metrics (subject-level fusion):")
    for k, v in combined_metrics.items():
        print(f"    {k:20s}: {v}")
    results["combined"] = {"test_metrics": combined_metrics}

    # ── Save training_results.json ────────────────────────────────────────
    results_path = MODEL_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] Training results → {results_path}")

    print("\n" + "=" * 70)
    print("  Training complete. Artifacts in backend/handwriting/:")
    print("    spiral_model.pth")
    print("    wave_model.pth")
    print("    fusion_meta.pkl")
    print("    training_results.json")
    print("=" * 70 + "\n")

    # Return test set records for smoke test (Step 9)
    return spiral_test, wave_test


if __name__ == "__main__":
    main()
