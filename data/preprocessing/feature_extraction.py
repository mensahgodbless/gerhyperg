from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# Default paths for pretrained weights
_DEFAULT_HSEMOTION_PATH = Path("weights/enet_b2_8.pt")


class PersonFeatureExtractor:
    
    FACE_DIM: int = 1408
    BODY_DIM: int = 512
    BBOX_DIM: int = 64
    TOTAL_DIM: int = FACE_DIM + BODY_DIM + BBOX_DIM  # 1984

    def __init__(
        self,
        device: str = "cuda",
        face_model_name: str = "enet_b2_8",
        face_weights_path: Optional[str] = None,
        clip_model_name: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        clip_weights_path: Optional[str] = None,
        bbox_embed_dim: int = 64,
        face_size: int = 260,
        body_size: int = 224,
    ) -> None:
        
        self.device = torch.device(device)
        self.bbox_embed_dim = bbox_embed_dim
        self.face_size = face_size
        self.body_size = body_size

        assert bbox_embed_dim % 8 == 0, (
            f"bbox_embed_dim must be divisible by 8, got {bbox_embed_dim}"
        )

        # Face model (HSEmotion EfficientNet-B2)
        self.face_model = self._build_face_model(face_model_name, face_weights_path)
        self.face_model.to(self.device)
        self.face_model.eval()
        self._freeze(self.face_model)

        # Body model (CLIP ViT-B/16) 
        self.clip_model, self.body_transform = self._build_clip_model(
            clip_model_name, clip_pretrained, clip_weights_path
        )
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self._freeze(self.clip_model)

        # Face transform (HSEmotion) 
        # HSEmotion uses ImageNet normalisation with 260×260 input
        self.face_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((face_size, face_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info(
            "PersonFeatureExtractor ready — face: %d-dim, body: %d-dim, "
            "bbox: %d-dim, total: %d-dim",
            self.FACE_DIM, self.BODY_DIM, self.bbox_embed_dim, self.total_dim,
        )

    @property
    def total_dim(self) -> int:
        """Total per-person feature dimensionality."""
        return self.FACE_DIM + self.BODY_DIM + self.bbox_embed_dim

    

    def _build_face_model(
        self,
        model_name: str,
        weights_path: Optional[str],
    ) -> nn.Module:
        
        import timm

        # Resolve weight path
        if weights_path is None:
            weights_path = str(Path("weights") / f"{model_name}.pt")
        weights_file = Path(weights_path)

        if weights_file.exists():
            logger.info("Loading HSEmotion weights from %s", weights_file)
            # Create architecture without classification head
            model = timm.create_model(
                "tf_efficientnet_b2", pretrained=False, num_classes=0
            )
            state = torch.load(str(weights_file), map_location="cpu", weights_only=False)

            # HSEmotion checkpoints come in multiple formats:
            #   1. Full model object (torch.save(model)) → nn.Module
            #   2. Wrapped state dict: {"state_dict": OrderedDict(...)}
            #   3. Raw state dict: OrderedDict(...)
            if isinstance(state, nn.Module):
                # Saved as full model — extract its state dict
                logger.info("HSEmotion checkpoint is a full model object")
                state = state.state_dict()
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            # Filter to only keys present in the headless model
            model_keys = set(model.state_dict().keys())
            compatible = {
                k: v for k, v in state.items() if k in model_keys
            }
            missing = model_keys - set(compatible.keys())
            if missing:
                logger.warning(
                    "HSEmotion checkpoint missing %d keys (likely classifier "
                    "head); using random init for those layers.",
                    len(missing),
                )
            model.load_state_dict(compatible, strict=False)
        else:
            logger.warning(
                "HSEmotion weights not found at %s — using ImageNet-pretrained "
                "EfficientNet-B2 as fallback.",
                weights_file,
            )
            model = timm.create_model(
                "tf_efficientnet_b2", pretrained=True, num_classes=0
            )

        return model

    def _build_clip_model(
        self,
        model_name: str,
        pretrained: str,
        weights_path: Optional[str] = None,
    ) -> tuple[nn.Module, T.Compose]:
        
        import open_clip        

        # Try explicit local path first 
        if weights_path is not None and Path(weights_path).exists():
            logger.info("Loading CLIP weights from local path: %s", weights_path)
            full_model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=weights_path
            )
            visual_encoder = full_model.visual
            return visual_encoder, preprocess

        # Try normal HuggingFace download
        try:
            full_model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            visual_encoder = full_model.visual
            return visual_encoder, preprocess
        except (FileNotFoundError, OSError, RuntimeError) as e:
            logger.warning("Online CLIP download failed: %s", e)

        # Fallback: search local weights/ directory and HF cache
        local_candidates = [
            Path("weights") / "open_clip_pytorch_model.bin",
            Path("weights") / f"{model_name.replace('/', '_')}_{pretrained}.bin",
            Path("weights") / "ViT-B-16_openai.pt",
        ]

        # Also check the HuggingFace cache in case a prior run succeeded
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for candidate in hf_cache.rglob("open_clip_pytorch_model.bin"):
                local_candidates.append(candidate)
            for candidate in hf_cache.rglob("open_clip_model.safetensors"):
                local_candidates.append(candidate)

        for candidate in local_candidates:
            if candidate.exists():
                logger.info("Found cached CLIP weights at %s", candidate)
                full_model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=str(candidate)
                )
                visual_encoder = full_model.visual
                return visual_encoder, preprocess

        raise FileNotFoundError(
            f"Cannot load CLIP {model_name}/{pretrained}: no internet and no "
            f"local weights found. Download the weights on a machine with "
            f"internet access:\n"
            f"  python -c \"import open_clip; "
            f"open_clip.create_model_and_transforms('{model_name}', "
            f"pretrained='{pretrained}')\"\n"
            f"Then copy the cached file from "
            f"~/.cache/huggingface/hub/ to this machine's weights/ directory."
        )

    @staticmethod
    def _freeze(model: nn.Module) -> None:
        """Freeze all parameters and set to eval mode."""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    
    # Feature extraction
    @torch.no_grad()
    def extract_face_features(self, face_crops: torch.Tensor) -> torch.Tensor:
        
        if face_crops.shape[0] == 0:
            return torch.zeros(0, self.FACE_DIM, device=self.device)

        face_crops = face_crops.to(self.device)  # [N, 3, 260, 260]
        features = self.face_model(face_crops)    # [N, 1408]
        assert features.shape[-1] == self.FACE_DIM, (
            f"Expected face dim {self.FACE_DIM}, got {features.shape[-1]}"
        )
        return features

    @torch.no_grad()
    def extract_body_features(self, body_crops: torch.Tensor) -> torch.Tensor:
        
        if body_crops.shape[0] == 0:
            return torch.zeros(0, self.BODY_DIM, device=self.device)

        body_crops = body_crops.to(self.device)  # [N, 3, 224, 224]
        features = self.clip_model(body_crops)    # [N, 512]
        assert features.shape[-1] == self.BODY_DIM, (
            f"Expected body dim {self.BODY_DIM}, got {features.shape[-1]}"
        )
        return features

    @torch.no_grad()
    def encode_bbox(
        self,
        bboxes: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        
        if bboxes.shape[0] == 0:
            return torch.zeros(0, self.bbox_embed_dim, device=self.device)

        bboxes = bboxes.to(self.device).float()  # [N, 4]

        # Convert [x1, y1, x2, y2] → normalised [cx, cy, w, h]
        eps = 1e-6
        cx = ((bboxes[:, 0] + bboxes[:, 2]) / 2.0) / max(img_w, eps)
        cy = ((bboxes[:, 1] + bboxes[:, 3]) / 2.0) / max(img_h, eps)
        w = (bboxes[:, 2] - bboxes[:, 0]) / max(img_w, eps)
        h = (bboxes[:, 3] - bboxes[:, 1]) / max(img_h, eps)

        # Clamp to [0, 1] for safety
        coords = torch.stack([cx, cy, w, h], dim=-1)  # [N, 4]
        coords = coords.clamp(0.0, 1.0)

        # Build sinusoidal encoding
        num_bands = self.bbox_embed_dim // 8  # per-coordinate: sin + cos
        freq_bands = torch.arange(num_bands, device=self.device, dtype=torch.float32)
        freq_bands = 2.0 * math.pi * (2.0 ** freq_bands)  # [num_bands]

        # [N, 4, 1] * [num_bands] → [N, 4, num_bands]
        angles = coords.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)

        sin_enc = torch.sin(angles)  # [N, 4, num_bands]
        cos_enc = torch.cos(angles)  # [N, 4, num_bands]

        # Interleave sin/cos and flatten: [N, 4, 2*num_bands] → [N, bbox_embed_dim]
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [N, 4, num_bands, 2]
        encoding = encoding.reshape(bboxes.shape[0], -1)     # [N, 4*num_bands*2 = bbox_embed_dim]

        assert encoding.shape[-1] == self.bbox_embed_dim, (
            f"Expected bbox encoding dim {self.bbox_embed_dim}, "
            f"got {encoding.shape[-1]}"
        )
        return encoding

    @torch.no_grad()
    def extract_all(
        self,
        face_crops: torch.Tensor,
        body_crops: torch.Tensor,
        bboxes: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        
        n = face_crops.shape[0]
        if n == 0:
            return torch.zeros(0, self.total_dim, device=self.device)

        assert face_crops.shape[0] == body_crops.shape[0] == bboxes.shape[0], (
            f"Mismatched batch sizes: face={face_crops.shape[0]}, "
            f"body={body_crops.shape[0]}, bbox={bboxes.shape[0]}"
        )

        face_feat = self.extract_face_features(face_crops)   # [N, 1408]
        body_feat = self.extract_body_features(body_crops)    # [N, 512]
        bbox_feat = self.encode_bbox(bboxes, img_w, img_h)    # [N, 64]

        combined = torch.cat([face_feat, body_feat, bbox_feat], dim=-1)  # [N, 1984]
        assert combined.shape == (n, self.total_dim)
        return combined


    # Preprocessing convenience
    def preprocess_face_crop(self, bgr_crop: np.ndarray) -> torch.Tensor:
        rgb_crop = bgr_crop[:, :, ::-1].copy()  # BGR → RGB
        return self.face_transform(rgb_crop)

    def preprocess_body_crop(self, bgr_crop: np.ndarray) -> torch.Tensor:
        from PIL import Image

        rgb_crop = bgr_crop[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb_crop)
        return self.body_transform(pil_img)

    def preprocess_detections(
        self,
        detections: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if not detections:
            return (
                torch.zeros(0, 3, self.face_size, self.face_size),
                torch.zeros(0, 3, self.body_size, self.body_size),
                torch.zeros(0, 4),
            )

        face_tensors: list[torch.Tensor] = []
        body_tensors: list[torch.Tensor] = []
        bbox_list: list[list[float]] = []

        for det in detections:
            face_tensors.append(self.preprocess_face_crop(det["face_crop"]))
            body_tensors.append(self.preprocess_body_crop(det["body_crop"]))
            bbox_list.append(det["bbox"])

        face_batch = torch.stack(face_tensors, dim=0)              # [N, 3, 260, 260]
        body_batch = torch.stack(body_tensors, dim=0)              # [N, 3, 224, 224]
        bbox_batch = torch.tensor(bbox_list, dtype=torch.float32)  # [N, 4]

        return face_batch, body_batch, bbox_batch