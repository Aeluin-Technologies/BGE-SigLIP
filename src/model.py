import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel


class SiglipBgeAligner(nn.Module):
    """Aligns SigLip2 vision features with BGE-M3 text embeddings."""

    def __init__(
        self, model_name="google/siglip2-so400m-patch16-naflex", target_dim=1024
    ):
        super().__init__()
        full_model = AutoModel.from_pretrained(model_name)
        self.vision_model = full_model.vision_model

        hidden_dim = self.vision_model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.logit_bias = nn.Parameter(torch.tensor([-10.0]))

    def forward(self, **vision_kwargs):
        """Passes input through vision encoder and projector."""
        outputs = self.vision_model(**vision_kwargs)
        projected = self.projector(outputs.pooler_output)
        return F.normalize(projected, p=2, dim=-1)


class SiglipONNXWrapper(nn.Module):
    """Wrapper for ONNX export compatibility."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values, attention_mask, spatial_shapes):
        return self.base_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
        )
