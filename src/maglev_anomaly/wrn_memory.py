"""WideResNet50-2 Layer2+Layer3 feature extraction and normal memory scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_channels(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-8)


class WideResNetFeatureExtractor:
    def __init__(
        self,
        *,
        device: str = "cuda",
        model_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        max_long_side: int = 2366,
    ):
        import timm
        import torch

        self.torch = torch
        self.device = device
        self.max_long_side = int(max_long_side)
        self.model = timm.create_model(model_name, pretrained=pretrained).eval().to(device)
        modules = dict(self.model.named_modules())
        self.outputs = {}
        self.handles = []
        for name in ("layer2", "layer3"):
            if name not in modules:
                raise RuntimeError(f"{model_name} lacks required layer {name}")
            self.handles.append(
                modules[name].register_forward_hook(
                    lambda _m, _i, output, key=name: self.outputs.__setitem__(key, output)
                )
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _prepare(self, image_rgb: np.ndarray):
        import cv2

        h, w = image_rgb.shape[:2]
        scale = min(1.0, self.max_long_side / max(h, w))
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        x = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return self.torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0), (out_h, out_w)

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch.nn.functional as functional

        x, _input_shape = self._prepare(np.asarray(image_rgb, dtype=np.uint8))
        self.outputs.clear()
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", enabled=str(self.device).startswith("cuda")
        ):
            self.model(x.to(self.device))
        features = []
        spatial = []
        for name in ("layer2", "layer3"):
            value = self.outputs[name]
            if isinstance(value, (tuple, list)):
                value = value[0]
            if value.ndim != 4:
                raise RuntimeError(f"unsupported {name} output shape {tuple(value.shape)}")
            spatial.append(tuple(value.shape[-2:]))
        target_h = max(x[0] for x in spatial)
        target_w = max(x[1] for x in spatial)
        for name in ("layer2", "layer3"):
            value = self.outputs[name]
            if isinstance(value, (tuple, list)):
                value = value[0]
            if tuple(value.shape[-2:]) != (target_h, target_w):
                value = functional.interpolate(value, size=(target_h, target_w), mode="bilinear", align_corners=False)
            array = value[0].permute(1, 2, 0).float().cpu().numpy()
            features.append(normalize_channels(array))
        return normalize_channels(np.concatenate(features, axis=-1))


class WRNNormalMemoryScorer:
    def __init__(self, extractor: WideResNetFeatureExtractor, artifact: str | Path, *, device: str = "cuda"):
        self.extractor = extractor
        self.device = device
        z = np.load(artifact)
        required = {"memory", "calibration_median", "calibration_q995"}
        if not required.issubset(z.files):
            raise RuntimeError(f"WRN artifact lacks {sorted(required - set(z.files))}")
        self.median = z["calibration_median"].astype(np.float32)
        self.q995 = z["calibration_q995"].astype(np.float32)
        torch = extractor.torch
        self.memory = torch.from_numpy(z["memory"]).to(device=device, dtype=torch.float16)

    def score(self, canonical_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.extractor.torch
        query = self.extractor.extract(canonical_rgb)
        if tuple(query.shape[:2]) != tuple(self.median.shape):
            raise RuntimeError(f"feature grid {query.shape[:2]} != calibration grid {self.median.shape}")
        q = torch.from_numpy(query).to(device=self.device, dtype=self.memory.dtype)
        with torch.inference_mode():
            similarity = torch.einsum("hwc,nhwc->nhw", q, self.memory)
            raw = (1.0 - similarity.max(dim=0).values).float().cpu().numpy()
        calibrated = (raw - self.median) / (self.q995 - self.median + 1e-6)
        return raw.astype(np.float32), calibrated.astype(np.float32)


def build_wrn_memory_artifact(
    canonical_images,
    extractor: WideResNetFeatureExtractor,
    output: str | Path,
    *,
    device: str = "cuda",
):
    """Build exact nearest-normal memory and leave-one-out calibration."""
    torch = extractor.torch
    features = [extractor.extract(image).astype(np.float16) for image in canonical_images]
    memory_np = np.stack(features)
    memory = torch.from_numpy(memory_np).to(device=device, dtype=torch.float16)
    references = []
    for index, feature in enumerate(features):
        query = torch.from_numpy(feature).to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            similarity = torch.einsum("hwc,nhwc->nhw", query, memory)
            similarity[index].fill_(-float("inf"))
            references.append((1.0 - similarity.max(dim=0).values).float().cpu().numpy())
    references = np.stack(references).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        memory=memory_np,
        calibration_median=np.median(references, axis=0).astype(np.float32),
        calibration_q995=np.quantile(references, 0.995, axis=0).astype(np.float32),
    )
    return output
