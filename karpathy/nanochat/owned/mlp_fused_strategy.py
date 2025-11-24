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
    Meaning if operation is y = Wx, we learn a scaling per input dimension (column of W).
    This should be EQUIVALENT TO CLASSICAL LRMS layer!!!!!
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.norm_parameters = nn.Parameter(torch.empty(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.norm_parameters) 
    
class RowMLPFusedStrategy(nn.Module):
    """
    MLP with row-wise fused weights.
    Meaning if operation is y = Wx, we learn a scaling per output dimension (row of W).
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.norm_parameters = nn.Parameter(torch.empty(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.norm_parameters
    
class FullMLPFusedStrategy(nn.Module):
    """
    MLP with full fused weights.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.column_parameters = nn.Parameter(torch.empty(input_dim))
        self.row_parameters = nn.Parameter(torch.empty(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.column_parameters) * self.row_parameters

def build_mlp_strategy(mlp_type: str, input_dim: int, output_dim: int, bias: bool = False):
    mlp_type = MLPType(mlp_type)
    if mlp_type == MLPType.DEFAULT:
        return nn.Linear(input_dim, output_dim, bias=bias)
    elif mlp_type == MLPType.COLUMN:
        return ColumnMLPFusedStrategy(input_dim, output_dim, bias=bias)
    elif mlp_type == MLPType.ROW:
        return RowMLPFusedStrategy(input_dim, output_dim, bias=bias)
    elif mlp_type == MLPType.FULL:
        return FullMLPFusedStrategy(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")
