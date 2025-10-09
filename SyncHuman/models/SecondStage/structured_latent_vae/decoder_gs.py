from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....modules import sparse as sp
from ....utils.random_utils import hammersley_sequence
from .base import SparseTransformerBase
from ....representations import Gaussian



class SLatGaussianDecoder(SparseTransformerBase):
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
        self._calc_layout()
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)
        
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
        
        
        self._build_perturbation()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        # super().initialize_weights()
        # Zero-out output layers:
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

    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config['voxel_size']
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer('offset_perturbation', perturbation)

    def _calc_layout(self) -> None:
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_features_dc' : {'shape': (self.rep_config['num_gaussians'], 1, 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.out_channels = start
    
    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            for k, v in self.layout.items():
                if k == '_xyz':
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    offset = offset * self.rep_config['lr'][k]
                    if self.rep_config['perturb_offset']:
                        offset = offset + self.offset_perturbation
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret
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


    def forward(self, x: sp.SparseTensor,mv_color,mv_normal,mv_color_face=None,mv_normal_face=None) -> List[Gaussian]:
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
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)
    


