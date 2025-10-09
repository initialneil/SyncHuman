from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        mv_dim: int = 1024,
        mv_condition_mode:Literal["4_view", "front","front&back","no"] = "4_view",
        use_faceinfo:bool=False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        self.mv_condition_mode= mv_condition_mode
        self.use_faceinfo= use_faceinfo
        if self.mv_condition_mode!="no":
            if self.use_faceinfo==True:
                if self.mv_condition_mode=='4_view':
                    self.mv_condition= nn.Sequential(
                        nn.Linear(channels+mv_dim*10, channels*4),
                        nn.GELU(approximate="tanh"),
                        nn.Linear(channels*4, channels),
                        )
                elif self.mv_condition_mode=='front&back':
                    self.mv_condition= nn.Sequential(
                        nn.Linear(channels+mv_dim*6, channels*4),
                        nn.GELU(approximate="tanh"),
                        nn.Linear(channels*4, channels),
                        )
                elif self.mv_condition_mode=='front':
                    self.mv_condition= nn.Sequential(
                        nn.Linear(channels+mv_dim*4, channels*4),
                        nn.GELU(approximate="tanh"),
                        nn.Linear(channels*4, channels),
                        )
            else:
                if self.mv_condition_mode=='4_view':
                    self.mv_condition= nn.Sequential(
                    nn.Linear(channels+mv_dim*8, channels*4),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(channels*4, channels),
                    )
                elif self.mv_condition_mode=='front&back':
                    self.mv_condition= nn.Sequential(
                    nn.Linear(channels+mv_dim*4, channels*4),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(channels*4, channels),
                    )
                elif self.mv_condition_mode=='front':
                    self.mv_condition= nn.Sequential(
                    nn.Linear(channels+mv_dim*2, channels*4),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(channels*4, channels),
                    )
            self.initialize_weights()
            last_layer = list(self.mv_condition.children())[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def _forward(self, x: SparseTensor,mv_normal=None,mv_color=None) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h)
        x = x + h
        if self.mv_condition_mode!="no":
            h = x.replace(self.norm3(x.feats))
            h =  h.replace(self.mv_condition(torch.concat([h.feats,mv_normal,mv_color],dim=1)))
            x = x + h
        
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor,mv_normal=None,mv_color=None) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mv_normal,mv_color,use_reentrant=False)
        else:
            return self._forward(x,mv_normal,mv_color)


class SparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor):
        h = x.replace(self.norm1(x.feats))
        h = self.self_attn(h)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)
