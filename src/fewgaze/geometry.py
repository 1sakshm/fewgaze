"""Method A: interpretable facial / iris geometry from MediaPipe Face Landmarker.

This stage also produces the square face crops shared by every appearance-based method
(DINOv2, L2CS-Net, CLIP), so all methods are evaluated on the same detected faces.

Feature definitions (all in image pixels before normalisation, per eye):
  iris_x   offset of the iris centre from the eye-corner midpoint, projected on the
           corner-to-corner axis (oriented towards image-right) and divided by eye width
  iris_y   same offset projected on the perpendicular axis (towards image-down) / eye width
  aperture distance between upper and lower lid landmarks / eye width
Head orientation: Euler angles (deg, scipy 'xyz' convention) of the rotation part of MediaPipe's
facial transformation matrix. These are features, so only consistency matters, not a convention.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "right_iris_x",
    "right_iris_y",
    "right_aperture",
    "left_iris_x",
    "left_iris_y",
    "left_aperture",
    "head_rx",
    "head_ry",
    "head_rz",
]

# MediaPipe Face Mesh topology (478 landmarks with iris refinement).
# "right"/"left" refer to the subject's eyes. Iris assignment is done by proximity, not index.
EYES = {
    "right": {"corners": (33, 133), "lids": (159, 145)},
    "left": {"corners": (362, 263), "lids": (386, 374)},
}
IRIS_CENTRES = (468, 473)


@dataclass
class GeometryResult:
    status: str  # ok | no_face | read_error
    features: np.ndarray  # (len(FEATURE_NAMES),) float, NaN on failure
    crop: Image.Image | None
    bbox: tuple[int, int, int, int] | None  # in decoded-image pixels


def ensure_model(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL from config
        tmp.replace(dest)
    return dest


def load_image(path: Path, decode_scale: float) -> Image.Image:
    img = Image.open(path)
    if decode_scale < 1.0 and img.format == "JPEG":
        w, h = img.size
        img.draft("RGB", (int(w * decode_scale), int(h * decode_scale)))
    return img.convert("RGB")


def eye_features(pts: np.ndarray) -> np.ndarray:
    """Compute the six per-eye features from (478, 2) pixel landmarks."""
    feats = []
    irises = pts[list(IRIS_CENTRES)]
    for eye in ("right", "left"):
        a, b = pts[EYES[eye]["corners"][0]], pts[EYES[eye]["corners"][1]]
        if a[0] > b[0]:  # orient axis towards image-right for both eyes
            a, b = b, a
        mid = (a + b) / 2
        width = float(np.linalg.norm(b - a))
        u = (b - a) / width
        v = np.array([-u[1], u[0]])  # perpendicular, points image-down when u points right
        iris = irises[np.argmin(np.linalg.norm(irises - mid, axis=1))]
        d = iris - mid
        up, lo = pts[EYES[eye]["lids"][0]], pts[EYES[eye]["lids"][1]]
        feats += [float(d @ u) / width, float(d @ v) / width, float(np.linalg.norm(up - lo)) / width]
    return np.array(feats)


def head_rotation(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    r = np.asarray(matrix, dtype=float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0, keepdims=True)  # strip scale
    u, _, vt = np.linalg.svd(r)  # nearest orthonormal matrix
    return Rotation.from_matrix(u @ vt).as_euler("xyz", degrees=True)


def square_crop(img: Image.Image, pts: np.ndarray, margin: float, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    box = (int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(cx + side / 2)), int(round(cy + side / 2)))
    return img.crop(box).resize((size, size), Image.Resampling.BICUBIC), box


class GeometryExtractor:
    def __init__(self, model_path: Path, pre_cfg: dict[str, Any]):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        self.mp = mp
        self.pre = pre_cfg
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(opts)

    def __call__(self, path: Path) -> GeometryResult:
        nan = np.full(len(FEATURE_NAMES), np.nan)
        try:
            img = load_image(path, float(self.pre["decode_scale"]))
        except (OSError, ValueError):
            return GeometryResult("read_error", nan, None, None)
        max_side = int(self.pre["landmark_max_side"])
        scale = min(1.0, max_side / max(img.size))
        small = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BILINEAR) if scale < 1 else img
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(small)))
        res = self.landmarker.detect(mp_img)
        if not res.face_landmarks:
            return GeometryResult("no_face", nan, None, None)
        lm = res.face_landmarks[0]
        # landmarks are normalised to the resized image; map to decoded-image pixels
        pts = np.array([[p.x * img.width, p.y * img.height] for p in lm])
        feats = np.concatenate([eye_features(pts), head_rotation(res.facial_transformation_matrixes[0])])
        crop, box = square_crop(img, pts, float(self.pre["face_crop_margin"]), int(self.pre["face_crop_size"]))
        return GeometryResult("ok", feats, crop, box)

    def close(self) -> None:
        self.landmarker.close()


def eye_strip(img: Image.Image, pts: np.ndarray, width_factor: float, out_size: tuple[int, int]) -> Image.Image:
    """Post-hoc P2: an axis-aligned strip around both eyes (see docs/methodology.md)."""
    idx = [EYES["right"]["corners"][0], EYES["right"]["corners"][1], EYES["left"]["corners"][0], EYES["left"]["corners"][1]]
    cx, cy = pts[idx].mean(axis=0)
    outer = float(np.linalg.norm(pts[EYES["right"]["corners"][0]] - pts[EYES["left"]["corners"][1]]))
    w = outer * width_factor
    h = w * out_size[1] / out_size[0]
    box = (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2)))
    return img.crop(box).resize(out_size, Image.Resampling.BICUBIC)


class EyeStripExtractor(GeometryExtractor):
    """Runs the landmarker on a cached face crop and returns the eye strip (or None)."""

    def strip(self, crop: Image.Image, width_factor: float, out_size: tuple[int, int]) -> Image.Image | None:
        arr = np.ascontiguousarray(np.asarray(crop.convert("RGB")))
        res = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=arr))
        if not res.face_landmarks:
            return None
        pts = np.array([[q.x * crop.width, q.y * crop.height] for q in res.face_landmarks[0]])
        return eye_strip(crop, pts, width_factor, out_size)
