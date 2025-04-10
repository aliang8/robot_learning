from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.models._utils import IntermediateLayerGetter

from robot_learning.models.utils.utils import make_conv_net
from robot_learning.utils.logger import log

try:
    import r3m

    R3M_AVAILABLE = True
except ImportError:
    R3M_AVAILABLE = False


EMBEDDING_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "r3m": 2048,
    "radio-g": 1536,
    "radio-h": 1536,
    "radio-l": 1536,
    "radio-b": 1536,
    "e-radio": 1536,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class ImageEmbedder(nn.Module):
    """Wrapper for pretrained image embedding models."""

    SUPPORTED_MODELS = [
        "resnet18",
        "resnet50",
        "r3m",
        "radio-g",
        "radio-h",
        "radio-l",
        "radio-b",
        "e-radio",
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
    ]
    RADIO_VERSIONS = {
        "radio-g": "radio_v2.5-g",
        "radio-h": "radio_v2.5-h",
        "radio-l": "radio_v2.5-l",
        "radio-b": "radio_v2.5-b",
        "e-radio": "e-radio_v2",
    }

    RESNET_CONFIGS = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    }

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        feature_fmt: str = "NCHW",  # For RADIO models
        use_spatial_features: bool = False,  # For RADIO models
        feature_map_layer: str = "avgpool",
    ):
        super().__init__()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Choose from {self.SUPPORTED_MODELS}"
            )

        self.model_name = model_name
        self.device = device
        self.feature_fmt = feature_fmt
        self.use_spatial_features = use_spatial_features

        # Initialize transforms based on model type
        if "radio" in model_name:
            # RADIO expects values between 0 and 1
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                ]
            )
        elif model_name == "r3m":
            self.transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]
            )
        elif "dinov2" in model_name:
            self.transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),  # TODO: check this
                ]
            )
        else:
            self.transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        # Initialize model
        if model_name in self.RESNET_CONFIGS:
            model_fn, weights = self.RESNET_CONFIGS[model_name]
            self.model = model_fn(weights=weights)
            self.transforms = weights.transforms()

            # get the feature map
            self.model = IntermediateLayerGetter(
                self.model, return_layers={feature_map_layer: "feature_map"}
            )
            log(f"Using feature map layer {feature_map_layer}")

            if feature_map_layer == "avgpool":
                self.output_dim = EMBEDDING_DIMS[model_name]
            elif feature_map_layer == "layer4":
                final_channels = EMBEDDING_DIMS[model_name]
                self.output_dim = [final_channels, 7, 7]
            else:
                raise ValueError(f"Feature map layer {feature_map_layer} not supported")

            log(f"Output dimension: {self.output_dim}")
            self.feature_map_layer = feature_map_layer

        elif model_name == "r3m":
            if not R3M_AVAILABLE:
                raise ImportError(
                    "R3M is not installed. Install it with: pip install r3m"
                )
            self.model = r3m.load_r3m("resnet50")  # Can also use resnet34 or resnet18
            self.output_dim = 2048

        elif "radio" in model_name:
            version = self.RADIO_VERSIONS[model_name]

            # Set custom download path
            import os

            os.environ["TORCH_HOME"] = "/scr/aliang80/.cache/torch/hub"

            default_cache_dir = os.path.expanduser("/scr/aliang80/.cache/torch/hub")
            cache_dir = os.getenv("TORCH_HOME", default_cache_dir)

            # Load model
            self.model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=version,
                progress=True,
                skip_validation=True,
                force_reload=False,
                cache_dir=cache_dir,
            )

            # Store useful model properties
            self.patch_size = self.model.patch_size
            self.max_resolution = self.model.max_resolution
            self.preferred_resolution = self.model.preferred_resolution
            self.min_resolution_step = self.model.min_resolution_step

            # Get feature dimensions from a test forward pass
            with torch.no_grad():
                test_input = torch.zeros(1, 3, 224, 224)
                summary, spatial = self.model(test_input)
                self.output_dim = (
                    summary.shape[-1]
                    if not self.use_spatial_features
                    else spatial.shape[-1]
                )

        elif model_name.startswith("dinov2"):
            # DINOv2 variants: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.model.eval()
            self.model.to(device)
            self.output_dim = EMBEDDING_DIMS[model_name]

        self.model.to(device)
        self.model.eval()

    def _process_radio_input(self, x: torch.Tensor) -> torch.Tensor:
        """Special processing for RADIO models."""
        # Get nearest supported resolution
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

        # Set optimal window size for e-radio
        if self.model_name == "e-radio":
            self.model.model.set_optimal_window_size(x.shape[2:])

        return x

    @torch.no_grad()
    def forward(
        self,
        images: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute embeddings for a batch of images.

        Args:
            images: Input images in one of these formats:
                   - torch.Tensor of shape [B, C, H, W] or [C, H, W]
                   - numpy array of shape [B, H, W, C] or [H, W, C]
                   - List of numpy arrays, each of shape [H, W, C]
            return_numpy: If True, return numpy array instead of torch tensor

        Returns:
            Embeddings as tensor or numpy array. Shape depends on model:
            - Regular models: [B, output_dim]
            - RADIO with spatial_features=True: [B, C, H, W] or [B, L, C] based on feature_fmt
        """

        # NOTES: PIL image expects channel last
        # PIL also expects uint8 so [0-255]

        # Convert input to PIL Images and apply transforms
        if isinstance(images, list):
            assert images[0].ndim == 3, "Images must be 3D numpy array"
            assert images[0].shape[2] == 3, "Images must have 3 channels"
            # assert between [0, 255]
            assert images[0].max() > 1.0 and images[0].min() >= 0.0, (
                "Images must be normalized to [0, 255]"
            )

            # List of numpy arrays
            processed = torch.stack(
                [
                    self.transforms(Image.fromarray(img.astype(np.uint8)))
                    for img in images
                ]
            )
        elif isinstance(images, np.ndarray):
            assert images.ndim in [3, 4], "Images must be 3D or 4D numpy array"
            if images.ndim == 4:
                assert images.shape[-1] == 3, "Images must have 3 channels"
            elif images.ndim == 3:
                assert images.shape[2] == 3, "Images must have 3 channels"

            # assert between [0, 255]
            assert images.max() > 1.0 and images.min() >= 0.0, (
                "Images must be normalized to [0, 255]"
            )

            if images.ndim == 3:
                # Single numpy image
                processed = self.transforms(
                    Image.fromarray(images.astype(np.uint8))
                ).unsqueeze(0)
            else:
                # Batch of numpy images
                processed = torch.stack(
                    [
                        self.transforms(Image.fromarray(img.astype(np.uint8)))
                        for img in images
                    ]
                )
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)

            # addd some sanity checks to input
            assert images.ndim == 4, "Images must be 4D tensor"
            assert images.max() <= 1.0 and images.min() >= 0.0, (
                "Images must be normalized to [0, 1]"
            )
            assert images.shape[1] == 3, (
                "Images must have 3 channels and be in CHW format"
            )

            processed = torch.stack(
                [
                    self.transforms(
                        Image.fromarray(
                            (img.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                        )
                    )
                    for img in images
                ]
            )
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")

        # Move to device and compute embeddings
        processed = processed.to(self.device)

        # Model-specific processing and forward pass
        if "radio" in self.model_name:
            processed = self._process_radio_input(processed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                summary, spatial_features = self.model(
                    processed, feature_fmt=self.feature_fmt
                )

                # Convert spatial features to NCHW format if requested
                if self.use_spatial_features and self.feature_fmt == "NCHW":
                    if (
                        self.model_name != "e-radio"
                    ):  # E-RADIO already returns BHWD format
                        from einops import rearrange

                        h = processed.shape[-2] // self.patch_size
                        w = processed.shape[-1] // self.patch_size
                        spatial_features = rearrange(
                            spatial_features, "b (h w) d -> b d h w", h=h, w=w
                        )

                embeddings = spatial_features if self.use_spatial_features else summary
        elif self.model_name in self.RESNET_CONFIGS:
            embeddings = self.model(processed)["feature_map"]
            if self.feature_map_layer == "avgpool":
                embeddings = embeddings.flatten(1)
        elif self.model_name == "r3m":
            # R3M expects values between 0 and 255
            assert processed.max() <= 1.0 and processed.min() >= 0.0, (
                "Images must be normalized to [0, 1]"
            )
            embeddings = self.model(processed * 255.0)
        elif self.model_name.startswith("dinov2"):
            # DINOv2 returns CLS token by default
            embeddings = self.model(processed)

        if return_numpy:
            embeddings = embeddings.cpu().numpy()

        return embeddings


class MultiInputEmbedder(nn.Module):
    """Handles embedding of multiple input modalities (e.g. images and states)"""

    def __init__(
        self,
        cfg: DictConfig,
        input_modalities: List[str],
        state_dim: int = 0,
        image_shape: Tuple[int, int] = None,
        seq_len: int = 1,
    ):
        super().__init__()
        self.input_modalities = input_modalities
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.seq_len = seq_len

        self.embedders = {}

        input_dim = 0

        embed_modalities = []
        image_modalities = []
        for modality in input_modalities:
            if "embed" in modality:
                embed_modalities.append(modality)
            elif "image" in modality:
                image_modalities.append(modality)

        if "states" in input_modalities:
            state_embedder = nn.Sequential(
                nn.Linear(state_dim * seq_len, cfg.embedding_dim),
                nn.GELU(),
                nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            )
            self.embedders["states"] = state_embedder
            input_dim += cfg.embedding_dim

        for modality in embed_modalities:
            # self.embedders[modality] = nn.Sequential(
            #     nn.Linear(
            #         EMBEDDING_DIMS[cfg.embedding_model] * seq_len, cfg.embedding_dim
            #     ),
            #     nn.GELU(),
            #     nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            # )
            self.embedders[modality] = nn.Identity()
            # input_dim += cfg.embedding_dim
            input_dim += EMBEDDING_DIMS[cfg.embedding_model] * seq_len

        for modality in image_modalities:
            if cfg.use_custom_cnn:
                # Create custom conv network to embed the images
                encoder_kwargs = None
                if "encoder" not in cfg:
                    # TODO: check this
                    encoder_kwargs = {
                        "out_channels": [64, 128, 128, 256, 256, cfg.input_embed_dim],
                        "kernel_size": [3] * 6,  # 3x3 kernel for all layers
                        "stride": [1] * 6,  # stride of 1 for all layers
                        "padding": [1] * 6,  # padding of 1 for all layers
                        "batch_norm": True,
                        "residual_layer": False,
                        "dropout": 0.1,
                    }
                    encoder_kwargs = OmegaConf.create(encoder_kwargs)
                else:
                    encoder_kwargs = cfg.encoder

                # TODO: add impala cnn back
                image_embedder, image_embedding_dim = make_conv_net(
                    image_shape,
                    output_embedding_dim=cfg.embedding_dim,
                    net_kwargs=encoder_kwargs,
                    apply_output_head=True,
                )
                input_dim += cfg.embedding_dim
            else:
                image_embedder = ImageEmbedder(
                    model_name=cfg.embedding_model,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                cfg.embedding_dim = image_embedder.output_dim
            self.embedders[modality] = image_embedder

        self.embedders = nn.ModuleDict(self.embedders)

        # if only one modality, no need for fusion
        if len(self.input_modalities) == 1:
            self.fusion_network = nn.Identity()
        else:
            self.fusion_network = nn.Sequential(
                nn.Linear(input_dim, cfg.embedding_dim),
                nn.GELU(),
                nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            )

        self.output_dim = cfg.embedding_dim

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the multimodal embedder.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of input tensors

        Returns:
            Fused embedding tensor of shape [B, fusion_dim]
        """
        embeds = []

        for modality in self.input_modalities:
            if modality in inputs:
                embeds.append(self.embedders[modality](inputs[modality]))

        # Concatenate and fuse embeddings
        combined = torch.cat(embeds, dim=-1)
        return self.fusion_network(combined)


if __name__ == "__main__":
    # Initialize RADIO embedder
    embedder = ImageEmbedder(
        model_name="radio-h",  # or 'radio-g', 'radio-l', 'radio-b', 'e-radio'
        device="cuda",
        feature_fmt="NCHW",
        use_spatial_features=False,  # Set to True to get spatial features instead of summary
    )

    # Single image
    image = np.random.rand(480, 640, 3)
    embedding = embedder(image)  # Get summary features

    # For spatial features
    embedder_spatial = ImageEmbedder(
        model_name="radio-h", use_spatial_features=False, feature_fmt="NCHW"
    )
    spatial_features = embedder_spatial(image)  # Get spatial features
    print(spatial_features.shape)

    # embedder = ImageEmbedder(
    #     model_name="r3m", device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    # img = np.random.rand(480, 640, 3)
    # embedding = embedder(img)
    # print(embedding.shape)
