from enum import Enum
import torch.nn as nn
import torch

class MLPType(str, Enum):
    DEFAULT = "default"                 # just outputting linear, default functionality.
    COLUMN = "column"      # column learnable format
    ROW = "row"      # row learnable format
    FULL = "full"   # full learnable format
    PATCHED = "patched"   # patched format (not implemented here, need to implement when we have first results of column and row.)


class ColumnMLPFusedStrategy(nn.Module):
    """
    MLP with column-wise fused weights.
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.norm_parameters = nn.Parameter(torch.empty(input_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.norm_parameters.T

def build_norm_strategy(mlp_type: MLPType, input_dim: int, output_dim: int, bias: bool = False):
    if mlp_type == MLPType.DEFAULT:
        return nn.Linear(input_dim, output_dim, bias=bias)
    elif mlp_type == MLPType.COLUMN:
        return LearnableRMSNormStrategy(dim, **kwargs)
    elif mlp_type == MLPType.ROW:
        return LayerNormStrategy(dim, **kwargs)
    elif mlp_type == MLPType.FULL:
        return IdentityNormStrategy(dim)
    else:
        raise ValueError(f"Unknown norm type: {mlp_type}")
