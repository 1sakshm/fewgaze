"""Method C: frozen L2CS-Net gaze estimator (Abdelrahman et al., 2023), Gaze360 weights.

The architecture is re-declared here (ResNet-50 trunk + two 90-bin classification heads, as in
the official repository's ``l2cs/model.py``) so that the official checkpoint loads with strict key
matching and without pulling the repository's face-detector dependency. The face crop comes from
the shared MediaPipe crop instead of RetinaFace; this is a documented deviation.

Decoding follows the official demo: softmax over bins, expectation over bin indices,
``angle_deg = E[bin] * bin_width - offset``. Outputs are pitch and yaw in degrees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fewgaze.embeddings import device_name, to_tensor_batch


def build_l2cs(num_bins: int):
    import torch.nn as nn
    import torchvision

    class L2CS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            r = torchvision.models.resnet50(weights=None)
            self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_yaw_gaze = nn.Linear(2048, num_bins)
            self.fc_pitch_gaze = nn.Linear(2048, num_bins)
            self.fc_finetune = nn.Linear(2048 + 3, 3)  # vestigial in the official checkpoint

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            feat = self.avgpool(x).flatten(1)
            return self.fc_yaw_gaze(feat), self.fc_pitch_gaze(feat), feat

    return L2CS()


class L2CSExtractor:
    def __init__(self, weights_path: Path, num_bins: int, bin_width: float, offset: float, input_size: int):
        import torch

        self.torch = torch
        self.device = device_name()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"L2CS-Net weights not found at {weights_path}. See docs/model_card.md for the official download link.")
        model = build_l2cs(num_bins)
        if str(weights_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(str(weights_path))
        else:  # original .pkl state dict; weights_only avoids arbitrary unpickling
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.idx = torch.arange(num_bins, dtype=torch.float32, device=self.device)
        self.bin_width, self.offset, self.size = bin_width, offset, input_size

    def __call__(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        t = self.torch
        with t.inference_mode():
            x = to_tensor_batch(images, self.size).to(self.device)
            yaw_logits, pitch_logits, feat = self.model(x)
            yaw = (yaw_logits.softmax(1) * self.idx).sum(1) * self.bin_width - self.offset
            pitch = (pitch_logits.softmax(1) * self.idx).sum(1) * self.bin_width - self.offset
        angles = t.stack([pitch, yaw], dim=1).float().cpu().numpy()
        return {"angles": angles, "embedding": feat.float().cpu().numpy()}


def angular_distance_from_camera(pitch_deg: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    """Angle (deg) between the predicted gaze vector and the direction towards the camera.

    Uses the L2CS/Gaze360 spherical convention g = (-cos p sin y, -sin p, -cos p cos y), for
    which (pitch, yaw) = (0, 0) points at the camera. Only |angle| matters, so the sign
    convention of the axes does not affect this score.
    """
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    cos_angle = np.clip(np.cos(p) * np.cos(y), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))
