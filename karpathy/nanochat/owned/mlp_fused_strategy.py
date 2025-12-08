from enum import Enum
import torch.nn as nn
import torch
import torch.nn.functional as F

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
        self.norm_parameters = nn.Parameter(torch.ones(input_dim))

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
        self.norm_parameters = nn.Parameter(torch.ones(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.norm_parameters
    
class FullMLPFusedStrategy(nn.Module):
    """
    MLP with full fused weights.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.column_parameters = nn.Parameter(torch.ones(input_dim))
        self.row_parameters = nn.Parameter(torch.ones(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.column_parameters) * self.row_parameters

class BlockMLPStrategy(nn.Module):
    """
    NOTE: Block scaling requires calculating W_eff = W * S.
    We cannot fuse this into input/output vectors.
    
    We use reshaping/broadcasting here to avoid creating a separate 'mask' tensor,
    saving ~33% memory compared to the original ScaledLinear implementation.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        bias: bool = False,
        out_block_size: int = 16,
        in_block_size: int = 16
    ):
        super().__init__()
        
        if output_dim % out_block_size != 0 or input_dim % in_block_size != 0:
            raise ValueError(
                f"Dimensions must be divisible by block size. "
                f"In: {input_dim}%{in_block_size}, Out: {output_dim}%{out_block_size}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_block_size = out_block_size
        self.in_block_size = in_block_size
        
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        n_out_blocks = output_dim // out_block_size
        n_in_blocks = input_dim // in_block_size
        
        # Initialize to 1.0
        self.block_scale = nn.Parameter(torch.ones(n_out_blocks, n_in_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Access Weight
        W = self.linear.weight # Shape: (Out, In)
        
        # 2. Reshape W to isolate blocks: (N_out_blks, Blk_out, N_in_blks, Blk_in)
        W_view = W.view(
            self.block_scale.shape[0], 
            self.out_block_size, 
            self.block_scale.shape[1], 
            self.in_block_size
        )
        
        # 3. Reshape Scale to broadcast: (N_out_blks, 1, N_in_blks, 1)
        scale_view = self.block_scale.view(
            self.block_scale.shape[0], 1, self.block_scale.shape[1], 1
        )
        
        # 4. Apply scale (Broadcasting handles the expansion, no mask tensor created)
        W_scaled = W_view * scale_view
        
        # 5. Flatten back to (Out, In) and run linear
        # usage of .reshape() instead of .view() ensures contiguity if needed
        return F.linear(x, W_scaled.reshape(self.output_dim, self.input_dim), self.linear.bias)


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
    elif mlp_type == MLPType.PATCHED:
        return BlockMLPStrategy(input_dim, output_dim, bias=bias)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")
