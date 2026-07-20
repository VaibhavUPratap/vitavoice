"""
predict.py — Inference endpoint for handwriting-based Parkinson's screening.

Exposes:
    predict_handwriting(spiral_input, wave_input) -> float

Accepts either:
  - File path strings / Path objects
  - FastAPI UploadFile objects (async reads automatically)

Models are loaded lazily and cached at module level.
"""

from __future__ import annotations

import os
import ssl
import asyncio
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import joblib
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HANDWRITING_DIR = Path(__file__).resolve().parent   # backend/handwriting/
SPIRAL_MODEL_PATH = HANDWRITING_DIR / "spiral_model.pth"
WAVE_MODEL_PATH   = HANDWRITING_DIR / "wave_model.pth"
META_MODEL_PATH   = HANDWRITING_DIR / "fusion_meta.pkl"

# ---------------------------------------------------------------------------
# Preprocessing transform (identical to EVAL_TRANSFORM in train.py)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_spiral_model: nn.Module | None = None
_wave_model:   nn.Module | None = None
_meta_model    = None
_device: torch.device | None = None


# ---------------------------------------------------------------------------
# SSL workaround (macOS Python 3.13)
# ---------------------------------------------------------------------------
_original_https_context = ssl._create_default_https_context  # noqa: SLF001

def _patch_ssl():
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

def _restore_ssl():
    ssl._create_default_https_context = _original_https_context  # noqa: SLF001


# ---------------------------------------------------------------------------
# Model architecture (must match train.py exactly)
# ---------------------------------------------------------------------------
def _build_resnet18_shell(device: torch.device) -> nn.Module:
    """Builds the model architecture without downloading weights."""
    _patch_ssl()
    try:
        model = models.resnet18(weights=None)
    finally:
        _restore_ssl()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
def _load_models():
    global _spiral_model, _wave_model, _meta_model, _device

    if _spiral_model is not None:
        return  # Already loaded

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not SPIRAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Spiral model not found at {SPIRAL_MODEL_PATH}. "
            "Run backend/handwriting/train.py first."
        )
    if not WAVE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Wave model not found at {WAVE_MODEL_PATH}. "
            "Run backend/handwriting/train.py first."
        )
    if not META_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fusion meta-model not found at {META_MODEL_PATH}. "
            "Run backend/handwriting/train.py first."
        )

    _spiral_model = _build_resnet18_shell(_device)
    _spiral_model.load_state_dict(
        torch.load(SPIRAL_MODEL_PATH, map_location=_device, weights_only=True)
    )
    _spiral_model.eval()

    _wave_model = _build_resnet18_shell(_device)
    _wave_model.load_state_dict(
        torch.load(WAVE_MODEL_PATH, map_location=_device, weights_only=True)
    )
    _wave_model.eval()

    _meta_model = joblib.load(META_MODEL_PATH)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def _preprocess_image(path: Union[str, Path]) -> torch.Tensor:
    """Loads an image from disk and returns a (1, 3, 224, 224) tensor."""
    img = Image.open(path).convert("RGB")
    tensor = EVAL_TRANSFORM(img)
    return tensor.unsqueeze(0)  # add batch dim


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def _score_image(model: nn.Module, img_tensor: torch.Tensor) -> float:
    """Forward pass through model, returns scalar probability."""
    img_tensor = img_tensor.to(_device)
    prob = model(img_tensor)
    return float(prob.squeeze().cpu().item())


async def _resolve_upload_file(upload_file) -> Path:
    """Saves an UploadFile to a temp file and returns its path."""
    suffix = Path(upload_file.filename).suffix if upload_file.filename else ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await upload_file.read()
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def predict_handwriting(
    spiral_input,
    wave_input,
) -> float:
    """
    Parameters
    ----------
    spiral_input : str | Path | FastAPI UploadFile
        The spiral drawing image.
    wave_input : str | Path | FastAPI UploadFile
        The wave drawing image.

    Returns
    -------
    float
        Combined Parkinson's probability in [0, 1].
        Higher = greater likelihood of Parkinson's.
    """
    _load_models()

    # Resolve inputs to paths
    spiral_tmp = None
    wave_tmp   = None
    try:
        # UploadFile objects have a .read coroutine
        if hasattr(spiral_input, "read") and asyncio.iscoroutinefunction(spiral_input.read):
            spiral_path = await _resolve_upload_file(spiral_input)
            spiral_tmp  = spiral_path
        else:
            spiral_path = Path(spiral_input)

        if hasattr(wave_input, "read") and asyncio.iscoroutinefunction(wave_input.read):
            wave_path = await _resolve_upload_file(wave_input)
            wave_tmp  = wave_path
        else:
            wave_path = Path(wave_input)

        # Preprocess
        spiral_tensor = _preprocess_image(spiral_path)
        wave_tensor   = _preprocess_image(wave_path)

        # Individual model scores
        spiral_score = _score_image(_spiral_model, spiral_tensor)
        wave_score   = _score_image(_wave_model,   wave_tensor)

        # Meta-model fusion
        X = np.array([[spiral_score, wave_score]])
        combined_prob = float(_meta_model.predict_proba(X)[0, 1])

        return combined_prob

    finally:
        # Clean up temp files
        for tmp in (spiral_tmp, wave_tmp):
            if tmp is not None and Path(tmp).exists():
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Synchronous wrapper for direct script usage / smoke tests
# ---------------------------------------------------------------------------
def predict_handwriting_sync(
    spiral_path: Union[str, Path],
    wave_path:   Union[str, Path],
) -> float:
    """
    Synchronous version of predict_handwriting for use in scripts.
    Accepts file paths only (not UploadFile).
    """
    _load_models()

    spiral_tensor = _preprocess_image(spiral_path)
    wave_tensor   = _preprocess_image(wave_path)

    spiral_score = _score_image(_spiral_model, spiral_tensor)
    wave_score   = _score_image(_wave_model,   wave_tensor)

    X = np.array([[spiral_score, wave_score]])
    return float(_meta_model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Quick smoke test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python predict.py <spiral_image_path> <wave_image_path>")
        sys.exit(1)

    spiral_p = sys.argv[1]
    wave_p   = sys.argv[2]

    print(f"Spiral image : {spiral_p}")
    print(f"Wave image   : {wave_p}")

    score = predict_handwriting_sync(spiral_p, wave_p)
    print(f"\nHandwriting PD probability : {score:.4f}")
    print(f"Interpretation             : {'HIGH risk (PD)' if score >= 0.5 else 'LOW risk (Healthy)'}")
