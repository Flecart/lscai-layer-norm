from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormType(str, Enum):
    RMS = "rms"                 # non-learnable
    LRMS = "learnable_rms"      # learnable RMSNorm
    LAYER = "layernorm"
    NONE = "none"


class NormStrategy(nn.Module):
    """Base class for different normalization strategies."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class RMSNormStrategy(NormStrategy):
    """
    Parameter-free RMSNorm (no scale).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Strictly functional RMSNorm (no learnable parameters)
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class LearnableRMSNormStrategy(NormStrategy):
    """
    RMSNorm with a learnable scale parameter (gamma).
    Equivalent to LLaMA-style RMSNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim)
        self.eps = eps
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(x, (x.size(-1),), self.gamma, eps=self.eps)
        return out                                                    


class LayerNormStrategy(NormStrategy):
    """Standard LayerNorm wrapper."""
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__(dim)
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class IdentityNormStrategy(NormStrategy):
    """No-op normalization (for ablations: 'no norm')."""
    def __init__(self, dim: int):
        super().__init__(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_norm_strategy(norm_type: NormType, dim: int, **kwargs):
    if norm_type == NormType.RMS:
        return RMSNormStrategy(dim, **kwargs)
    elif norm_type == NormType.LRMS:
        return LearnableRMSNormStrategy(dim, **kwargs)
    elif norm_type == NormType.LAYER:
        return LayerNormStrategy(dim, **kwargs)
    elif norm_type == NormType.NONE:
        return IdentityNormStrategy(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
