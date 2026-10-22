"""Reference outputs for the browser demo parity test (site/demo/parity.html).

Writes 24 Columbia images (resized exactly as the Python pipeline does) plus the Python features and
probabilities to site/demo/parity/, which is gitignored: the images must never be published.
Then serve site/demo over HTTP and open parity.html in a browser.
"""

import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from fewgaze.config import REPO_ROOT, load_config, resolve_path
from fewgaze.dataset import load_manifest
from fewgaze.geometry import GeometryExtractor, ensure_model, eye_features, head_rotation, load_image

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from export_demo_model import predict_exported  # noqa: E402

OUT = REPO_ROOT / "site" / "demo" / "parity"  # local only, gitignored (contains dataset images)
OUT.mkdir(exist_ok=True)


# 1) Euler formula check: extrinsic 'xyz' == R = Rz(c) Ry(b) Rx(a)
rng = np.random.default_rng(0)
for _ in range(1000):
    R = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
    a, b, c = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    b2 = -np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    a2 = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    c2 = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    assert np.allclose([a, b, c], [a2, b2, c2], atol=1e-6), (a, b, c, a2, b2, c2)
print("euler formula matches scipy 'xyz'")

cfg = load_config(REPO_ROOT / "configs" / "experiment.yaml")
man = load_manifest(cfg).records
pick = pd.concat([man[(man.gaze_v == 0) & (man.gaze_h == 0)].sample(10, random_state=1), man.sample(14, random_state=2)])
model = json.loads((REPO_ROOT / "site/demo/model.json").read_text())
ext = GeometryExtractor(
    ensure_model(cfg["methods"]["geometry"]["landmarker_model_url"], resolve_path("cache/weights/face_landmarker.task")),
    cfg["preprocessing"],
)
root = resolve_path(cfg["dataset"]["root"])
refs = []
for r in pick.itertuples():
    img = load_image(root / r.relpath, 0.5)
    scale = min(1.0, 1600 / max(img.size))
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BILINEAR)
    small.save(OUT / f"{r.sample_id}.png")
    res = ext.landmarker.detect(ext.mp.Image(image_format=ext.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(small))))
    lm = res.face_landmarks[0]
    pts = np.array([[p.x * small.width, p.y * small.height] for p in lm])
    M = np.asarray(res.facial_transformation_matrixes[0])
    feats = np.concatenate([eye_features(pts), head_rotation(M)])
    prob = float(predict_exported(model, feats[None, :])[0])
    refs.append(
        {
            "id": r.sample_id,
            "w": small.width,
            "h": small.height,
            "features": feats.tolist(),
            "prob": prob,
            "matrix_rowmajor": M.ravel().tolist(),
            "lm0": [lm[0].x, lm[0].y],
            "lm468": [lm[468].x, lm[468].y],
        }
    )
(OUT / "ref.json").write_text(json.dumps(refs))
print("wrote", len(refs), "references")
