from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ....modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ....modules import sparse as sp
from .base import SparseTransformerBase
from ....representations import MeshExtractResult
from ....representations.mesh import SparseFeatures2Mesh



class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h


class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        mv_dim:int=1024,
        mv_condition_mode:Literal["4_view", "front","front&back","front_global_cross"] ="4_view",
        use_multiscale:bool=False,
        use_faceinfo:bool=False
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            mv_condition_mode=mv_condition_mode,
            use_faceinfo=use_faceinfo
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeatures2Mesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)
        if use_multiscale==True:
            self.input_layer_color=nn.Linear(4*mv_dim,mv_dim)
            self.input_layer_normal=nn.Linear(4*mv_dim,mv_dim)
        else:
            self.input_layer_color=nn.Linear(mv_dim,mv_dim)
            self.input_layer_normal=nn.Linear(mv_dim,mv_dim)
        if use_faceinfo==True:
            self.input_layer_color_face=nn.Linear(4*mv_dim,mv_dim)
            self.input_layer_normal_face=nn.Linear(4*mv_dim,mv_dim)
        self.mv_condition_mode=mv_condition_mode
        self.use_faceinfo=use_faceinfo
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
    
    def init_upsample_2(self,module):
        for name, submodule in module.named_modules():
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                if submodule.weight.dim() >= 2:
                # 只对维度 >= 2 的权重做 xavier_uniform_
                    nn.init.xavier_uniform_(submodule.weight)
                else:
                # 通常是 norm 层的 weight，置为1（可选）
                    nn.init.constant_(submodule.weight, 1.0)
            if hasattr(submodule, 'bias') and submodule.bias is not None:
                nn.init.constant_(submodule.bias, 0.0)
                
    

    def initialize_weights(self) -> None:
        # super().initialize_weights()
        # Zero-out output layers:
        # self.init_upsample_2(self.upsample[2])
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
        nn.init.xavier_uniform_(self.input_layer_color.weight)
        nn.init.constant_(self.input_layer_color.bias, 0)
        nn.init.xavier_uniform_(self.input_layer_normal.weight)
        nn.init.constant_(self.input_layer_normal.bias, 0)
        if self.use_faceinfo==True:
            nn.init.constant_(self.input_layer_color_face.weight, 0)
            nn.init.constant_(self.input_layer_color_face.bias, 0)
            nn.init.constant_(self.input_layer_normal_face.weight, 0)
            nn.init.constant_(self.input_layer_normal_face.bias, 0)
            
        

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret
    import torch

    def condition_mv(self,sparse_tensor, image_tensor,face_info):

        coords = sparse_tensor.coords  
        feats = sparse_tensor.feats    
        bs= coords[:, 0].to(torch.int64)
        x = coords[:, 1].to(torch.int64)
        y = coords[:, 2].to(torch.int64)
        z = coords[:, 3].to(torch.int64)
        row = 63 - z
       
        pixel0 = image_tensor[bs,0,  row, x,:]  # view 0
        pixel1 = image_tensor[bs,1,  row, y,:]  # view 1
        pixel2 = image_tensor[bs,2,  row, x,:]  # view 2
        pixel3 = image_tensor[bs,3, row, y,:]  # view 3
        
        if self.mv_condition_mode=="4_view":
            new_feats_mv = torch.concat([pixel0, pixel1, pixel2, pixel3], dim=1)
        elif self.mv_condition_mode=="front":
            new_feats_mv = pixel0
        elif self.mv_condition_mode=="front&back":
            new_feats_mv = torch.concat([pixel0, pixel2], dim=1)
        
        if self.use_faceinfo==True:
            pixel_face = face_info[bs,  row, x,:]  # view 0
            new_feats_mv= torch.concat([new_feats_mv,pixel_face], dim=1)
            
       
        return new_feats_mv


    def forward(self, x: sp.SparseTensor,mv_color,mv_normal,mv_color_face=None,mv_normal_face=None) -> List[MeshExtractResult]:
        #mv input layer
        mv_color=self.input_layer_color(mv_color)
        mv_normal=self.input_layer_normal(mv_normal)
        mv_normal_face_in=None
        mv_color_face_in=None
        if self.use_faceinfo==True:
            mv_color_face=self.input_layer_color_face(mv_color_face)
            mv_normal_face=self.input_layer_normal_face(mv_normal_face)
            mv_color_face_in=torch.zeros_like(mv_color[:,0]).to(device=mv_color.device)
            mv_color_face_in[:,:20,22:42,:]=mv_color_face
            mv_normal_face_in=torch.zeros_like(mv_normal[:,0]).to(device=mv_normal.device)
            mv_normal_face_in[:,:20,22:42,:]=mv_normal_face
        mv_normal=self.condition_mv(x,mv_normal,mv_normal_face_in)
        mv_color=self.condition_mv(x,mv_color,mv_color_face_in)
        
        h = super().forward(x,mv_normal,mv_color)
        for block in self.upsample:
            h = block(h)
        h = h.type(x.dtype)
        h = self.out_layer(h)
    
        return self.to_representation(h)
    

