from enum import Enum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormType(str, Enum):
    RMS = "rms"                 # non-learnable
    LRMS = "learnable_rms"      # learnable RMSNorm
    LAYER = "layernorm"
    TORUS = "torus"             # torus normalization
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
        out = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        return out * self.gamma


class LayerNormStrategy(NormStrategy):
    """Standard LayerNorm wrapper."""
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__(dim)
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)
    

class TorusNormStrategy(NormStrategy):
    """
    Project embeddings onto a product of spheres (generalized torus-like manifold).

    Implementation:
    - Choose a group_size G (>= 2).
    - Split last dim into groups: (..., D) -> (..., num_groups, G)
      where num_groups = D // G.
    - Each group is projected to a sphere of radius r_group in R^G.
    - Default r_group is chosen so that the overall L2 norm matches
      LayerNorm's sphere radius ≈ sqrt(D), which gives r_group = sqrt(G).

    Special cases:
    - group_size = 2  -> product of circles.
    - group_size = D  -> one big sphere (essentially classic sphere projection).
    """

    def __init__(
        self,
        dim: int,
        group_size_torus_norm: int = 2,
        radius: float | None = None,   # per-group radius in R^group_size
        eps: float = 1e-6,
        learnable_radius: bool = False,
    ):
        super().__init__(dim)

        if group_size_torus_norm <= 1:
            raise ValueError(f"group_size_torus_norm must be >= 2, got {group_size_torus_norm}.")
        if dim % group_size_torus_norm != 0:
            raise ValueError(
                f"TorusNormStrategy requires dim % group_size_torus_norm == 0, "
                f"got dim={dim}, group_size_torus_norm={group_size_torus_norm}."
            )

        self.eps = eps
        self.group_size = group_size_torus_norm
        num_groups = dim // group_size_torus_norm

        if learnable_radius:
            raise NotImplementedError("learnable_radius not supported with meta-init yet")

        # Pick radius s.t. total ||x|| ≈ sqrt(dim): r_group = sqrt(group_size)
        if radius is None:
            radius = math.sqrt(float(group_size_torus_norm))
        # Just store as a Python float; no buffer, no Parameter
        self.radius = float(radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) with D % group_size == 0.

        Steps:
        1) reshape to (..., num_groups, group_size)
        2) compute norm of each group
        3) rescale each group to the target radius
        4) reshape back to (..., D)
        """
        orig_shape = x.shape
        D = x.size(-1)
        group_size = self.group_size
        num_groups = D // group_size

        x_groups = x.reshape(-1, num_groups, group_size)      # (-1, G, group_size)
        norms = x_groups.norm(dim=-1, keepdim=True)           # (-1, G, 1)

        # radius as a scalar; PyTorch will put it on the right device/dtype in arithmetic
        scale = self.radius / (norms + self.eps)
        x_proj = x_groups * scale

        out = x_proj.reshape(orig_shape)
        return out


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
    elif norm_type == NormType.TORUS:
        return TorusNormStrategy(dim, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
