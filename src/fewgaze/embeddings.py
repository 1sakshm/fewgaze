"""Method B: frozen DINOv2 representations (and CLIP image embeddings for the VLM probe).

Preprocessing for DINOv2 is deterministic and documented: the cached 448x448 face crop is resized
to ``input_size`` x ``input_size`` (bicubic) and normalised with ImageNet statistics. No centre
crop, no augmentation. The backbone is never updated (eval mode, no gradients).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor_batch(images: list[Image.Image], size: int | tuple[int, int]):
    import torch

    wh = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    arr = np.stack([np.asarray(im.convert("RGB").resize(wh, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0 for im in images])
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


class DinoV2Extractor:
    """Returns two representations per image: pooled CLS token and mean of patch tokens."""

    def __init__(self, model_id: str, input_size: int | tuple[int, int]):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device_name()
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.size = input_size
        self.dim = self.model.config.hidden_size

    def __call__(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        with self.torch.inference_mode():
            x = to_tensor_batch(images, self.size).to(self.device)
            out = self.model(pixel_values=x)
            cls = out.pooler_output  # layer-normed CLS token
            patch = out.last_hidden_state[:, 1:, :].mean(dim=1)
        return {"cls": cls.float().cpu().numpy(), "patchmean": patch.float().cpu().numpy()}


class ClipExtractor:
    """CLIP image embeddings (L2-normalised) plus fixed-prompt zero-shot scores."""

    def __init__(self, model_id: str, positive_prompt: str, negative_prompt: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device_name()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        with torch.inference_mode():
            tok = self.processor(text=[negative_prompt, positive_prompt], return_tensors="pt", padding=True).to(self.device)
            t = self.model.get_text_features(**tok)
            self.text = t / t.norm(dim=-1, keepdim=True)
        self.logit_scale = float(self.model.logit_scale.exp())
        self.dim = self.model.config.projection_dim

    def __call__(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        with self.torch.inference_mode():
            px = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**px)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * f @ self.text.T  # columns: [negative, positive]
            p_pos = logits.softmax(dim=-1)[:, 1]
        return {"embedding": f.float().cpu().numpy(), "zeroshot": p_pos.float().cpu().numpy()[:, None]}


def load_crop(crop_dir: Path, sample_id: str) -> Image.Image:
    return Image.open(crop_dir / f"{sample_id}.jpg").convert("RGB")
