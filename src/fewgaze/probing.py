"""Study 2: layer-wise probing of frozen vision foundation models for gaze (docs/methodology.md, Study 2).

For every layer (embedding output + each transformer block) two representations are cached:
  global  CLS token (SigLIP: mean of all patch tokens, it has no CLS)
  eye     mean of the patch tokens over the right eye || mean over the left eye
Eye boxes come from MediaPipe landmarks re-detected on the cached 448x448 face crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fewgaze.geometry import EYES, IRIS_CENTRES

CROP = 448
INPUT = 224


@dataclass(frozen=True)
class ProbeModel:
    key: str
    hf_id: str
    kind: str  # dinov2 | clip | siglip | mae
    patch: int
    has_cls: bool
    label: str


MODELS = [
    ProbeModel("dinov2_s", "facebook/dinov2-small", "dinov2", 14, True, "DINOv2-S/14"),
    ProbeModel("dinov2_b", "facebook/dinov2-base", "dinov2", 14, True, "DINOv2-B/14"),
    ProbeModel("clip_b16", "openai/clip-vit-base-patch16", "clip", 16, True, "CLIP ViT-B/16"),
    ProbeModel("siglip_b16", "google/siglip-base-patch16-224", "siglip", 16, False, "SigLIP-B/16"),
    ProbeModel("mae_b16", "facebook/vit-mae-base", "mae", 16, True, "MAE ViT-B/16"),
]


def eye_boxes_from_landmarks(pts: np.ndarray, pad: float = 0.25) -> np.ndarray:
    """(2, 4) boxes [x0, y0, x1, y1] for the right then left eye, padded by `pad` x eye width."""
    boxes = []
    irises = pts[list(IRIS_CENTRES)]
    for eye in ("right", "left"):
        idx = [*EYES[eye]["corners"], *EYES[eye]["lids"]]
        p = pts[idx]
        mid = p[:2].mean(axis=0)
        iris = irises[np.argmin(np.linalg.norm(irises - mid, axis=1))]
        p = np.vstack([p, iris])
        width = float(np.linalg.norm(pts[EYES[eye]["corners"][0]] - pts[EYES[eye]["corners"][1]]))
        x0, y0 = p.min(axis=0) - pad * width
        x1, y1 = p.max(axis=0) + pad * width
        boxes.append([x0, y0, x1, y1])
    return np.array(boxes)


def eye_patch_masks(boxes_crop: np.ndarray, grid: int, patch: int) -> list[np.ndarray]:
    """For each eye: indices (row-major) of patches whose centre lies in the box, in INPUT-pixel space.
    Falls back to the single nearest patch when none qualifies."""
    s = INPUT / CROP
    centres = (np.arange(grid) + 0.5) * patch
    cy, cx = np.meshgrid(centres, centres, indexing="ij")
    out = []
    for x0, y0, x1, y1 in boxes_crop * s:
        inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        idx = np.flatnonzero(inside.ravel())
        if idx.size == 0:
            d = (cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2
            idx = np.array([int(np.argmin(d.ravel()))])
        out.append(idx)
    return out


class LayerExtractor:
    def __init__(self, spec: ProbeModel):
        import torch
        from transformers import AutoImageProcessor

        self.torch, self.spec = torch, spec
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(spec.hf_id)
        self.mean = np.array(proc.image_mean, dtype=np.float32)
        self.std = np.array(proc.image_std, dtype=np.float32)
        if spec.kind == "dinov2":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(spec.hf_id)
        elif spec.kind == "clip":
            from transformers import CLIPVisionModel

            model = CLIPVisionModel.from_pretrained(spec.hf_id)
        elif spec.kind == "siglip":
            from transformers import SiglipVisionModel

            model = SiglipVisionModel.from_pretrained(spec.hf_id)
        elif spec.kind == "mae":
            from transformers import ViTMAEModel

            model = ViTMAEModel.from_pretrained(spec.hf_id, mask_ratio=0.0)
        else:
            raise ValueError(spec.kind)
        self.model = model.to(self.device).eval()
        self.grid = INPUT // spec.patch
        self.dim = model.config.hidden_size
        self.params = int(sum(p.numel() for p in model.parameters()))

    def __call__(self, images: list[Image.Image], masks: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """Returns global (B, L, D) and eye (B, L, 2D) float32 arrays."""
        t = self.torch
        arr = np.stack(
            [np.asarray(im.convert("RGB").resize((INPUT, INPUT), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0 for im in images]
        )
        x = t.from_numpy(((arr - self.mean) / self.std).transpose(0, 3, 1, 2).copy()).to(self.device)
        n_patch = self.grid * self.grid
        with t.inference_mode():
            if self.spec.kind == "mae":
                noise = t.arange(n_patch, device=self.device, dtype=t.float32).expand(x.shape[0], n_patch)  # identity order
                out = self.model(pixel_values=x, noise=noise, output_hidden_states=True)
            else:
                out = self.model(pixel_values=x, output_hidden_states=True)
        hs = t.stack(out.hidden_states, dim=1).float().cpu().numpy()  # (B, L, T, D)
        off = 1 if self.spec.has_cls else 0
        assert hs.shape[2] == n_patch + off, (hs.shape, n_patch, off)
        patches = hs[:, :, off:, :]
        glob = hs[:, :, 0, :] if self.spec.has_cls else patches.mean(axis=2)
        eyes = np.stack(
            [
                np.concatenate([patches[b][:, m[0], :].mean(axis=1), patches[b][:, m[1], :].mean(axis=1)], axis=-1)
                for b, m in enumerate(masks)
            ]
        )
        return glob, eyes


def cache_dir(root: Path) -> Path:
    d = root / "study2"
    d.mkdir(parents=True, exist_ok=True)
    return d
