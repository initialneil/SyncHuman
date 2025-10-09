from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from diffusers.models.attention import FeedForward
logger = logging.get_logger(__name__)  

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
    

class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.cross_attention_norm = cross_attention_norm

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        if cross_attention_norm:
            self.norm_cross = nn.LayerNorm(cross_attention_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        if processor is None:
            CrossAttnProcessor()
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if use_memory_efficient_attention_xformers:
    
            processor = XFormersCrossAttnProcessor(attention_op=attention_op)
        else:
            processor = CrossAttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `CrossAttention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                (
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask


class CrossAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class XFormersCrossAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states



AttnProcessor = Union[
    CrossAttnProcessor,
    XFormersCrossAttnProcessor,
]

    
class Conv3DDownsample(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super(Conv3DDownsample, self).__init__()
        
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU()  # Using GELU for better performance
        )
        self.pool = nn.AdaptiveAvgPool3d((12, 12, 12))

    def forward(self, x):
        x = self.conv(x)  
        x = self.pool(x)  
        return x


class Conv2DDownsample(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super(Conv2DDownsample, self).__init__()
        
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        x = self.conv(x)  
        x = self.pool(x)
        return x
    
class DualCrossAttentionMV(nn.Module):
    def __init__(self, query_dim_MV, query_dim_Structure,heads=8, dim_head=64, dropout=0.0,num_views=5 ,bias=False,processor=XFormersCrossAttnProcessor):
        super().__init__()
        
        self.voxeldownsample=Conv3DDownsample(query_dim_Structure, query_dim_Structure)
        
        self.normMV_no1 = LayerNorm32(query_dim_MV, elementwise_affine=False, eps=1e-6)
        self.normMV_clr1 = LayerNorm32(query_dim_MV, elementwise_affine=False, eps=1e-6)
        
        self.normMV_no2 = LayerNorm32(query_dim_MV, elementwise_affine=False, eps=1e-6)
        self.normMV_clr2 = LayerNorm32(query_dim_MV, elementwise_affine=False, eps=1e-6)

        self.input_layer_mv = nn.Linear(query_dim_MV,query_dim_MV)
        self.input_layer_stucture = nn.Linear(query_dim_Structure,query_dim_Structure)
          
        self.output_layer_no=nn.Linear(query_dim_MV, query_dim_MV)
        self.output_layer_clr= nn.Linear(query_dim_MV, query_dim_MV)

        
        self.ffn_no=FeedForward(query_dim_MV)
        self.ffn_clr= FeedForward(query_dim_MV)

        
        self.cross_attn_Structure_to_color = CrossAttention(query_dim_MV, query_dim_Structure, heads, dim_head, dropout, bias,processor=processor)
        self.cross_attn_Structure_to_normal = CrossAttention(query_dim_MV, query_dim_Structure, heads, dim_head, dropout, bias,processor=processor)

        self.num_views=num_views
    def forward(self, MV, structure):
        #1.input layer
        #MV(20,640,16,16)      #structure(2,4096,1024)bs  (16,16,16) dim
        MV_pre=MV
        MV=rearrange(MV,'B d h w->B h w d')
        MV=self.input_layer_mv(MV)
        
        MV_normal,MV_color=MV.chunk(2,dim=0)#10,640,16,16
        MV_normal=rearrange(MV_normal,'(b Nv)  h w d->b Nv d h w',Nv=self.num_views)
        MV_color=rearrange(MV_color,'(b Nv)  h w d->b Nv d h w',Nv=self.num_views)
        B,Nv,D,H, W=MV_normal.shape
        device=MV.device
        
        structure=self.input_layer_stucture(structure)
        structure=rearrange(structure, 'b (v1 v2 v3) d -> b d v1 v2 v3', v1=16, v2=16, v3=16)
        structure=self.voxeldownsample(structure)
        
        #2.get voxel_kv for image cross
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid_y = grid_y.unsqueeze(0).expand(B, H, W)        # (B, H, W)
        grid_x = grid_x.unsqueeze(0).expand(B, H, W)  
        idx_y = (11 - grid_y)  # (B, H, W)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
        slice0 = structure[batch_idx,        
                       :,                
                       grid_x,            
                       :,                
                       idx_y]             

        slice1 = structure[batch_idx,        
                       :,                
                       :,                
                       grid_x,            
                       idx_y]             

        voxel_kv = torch.stack([slice0, slice1, slice0, slice1], dim=1)
        voxel_kv=rearrange(voxel_kv, 'b Nv h w d v3 -> (b Nv h w) v3 d')
        
        MV_normal=rearrange(MV_normal[:,:4,:,:,:],'b Nv d h w->(b Nv h w) 1 d ')
        MV_color=rearrange(MV_color[:,:4,:,:,:],'b Nv d h w->(b Nv h w) 1 d ')

        
        #4.cross-attention
        MV_normal_norm=self.normMV_no1(MV_normal)
        MV_color_norm=self.normMV_clr1(MV_color)
        

        batch_size_B = MV_normal.size(0)

        sub_batch_B = batch_size_B // 4

        processed_mv_normal, processed_mv_color = [], []

        for i in range(4):
            start_B = i * sub_batch_B
            end_B = (i+1) * sub_batch_B
            mv_normal_sub = MV_normal_norm[start_B:end_B]
            mv_color_sub = MV_color_norm[start_B:end_B]
            voxel_kv_sub = voxel_kv[start_B:end_B]
    
            out_mv_normal = self.cross_attn_Structure_to_normal(mv_normal_sub, voxel_kv_sub)
            out_mv_color = self.cross_attn_Structure_to_color(mv_color_sub, voxel_kv_sub)
    
            processed_mv_normal.append(out_mv_normal)
            processed_mv_color.append(out_mv_color)
    


        MV_normal_att = torch.cat(processed_mv_normal, dim=0)
        MV_color_att = torch.cat(processed_mv_color, dim=0)
        MV_normal=MV_normal+MV_normal_att                                                                                                                           
        MV_color=MV_color+MV_color_att
         
        #ffn
        
        MV_normal_norm=self.normMV_no2(MV_normal)
        MV_color_norm=self.normMV_clr2(MV_color)
        
        MV_normal_ffn =self.ffn_no(MV_normal_norm)
        MV_color_ffn =self.ffn_clr(MV_color_norm)
        
        MV_normal=MV_normal+MV_normal_ffn
        MV_color=MV_color+MV_color_ffn
        
        
        #5.output_layer
        MV_normal =self.output_layer_no(MV_normal)
        MV_color =self.output_layer_clr(MV_color)
        
        
        MV_normal=rearrange(MV_normal,'(b Nv h w) 1 d ->b Nv d h w ',Nv=self.num_views-1,h=H,w=W) 
        MV_normal=torch.cat([MV_normal, torch.zeros((B,1,MV_normal.shape[2],H,W)).to(MV_normal)],dim=1)  
        MV_normal=rearrange(MV_normal,'b Nv d h w ->(b Nv) d h w ',Nv=self.num_views,h=H,w=W) 
        
        MV_color=rearrange(MV_color,'(b Nv h w) 1 d ->b Nv d h w ',Nv=self.num_views-1,h=H,w=W) 
        MV_color=torch.cat([MV_color, torch.zeros((B,1,MV_color.shape[2],H,W)).to(MV_color)],dim=1)  
        MV_color=rearrange(MV_color,'b Nv d h w ->(b Nv) d h w ',Nv=self.num_views,h=H,w=W) 
        
        MV=torch.cat([MV_normal,MV_color],dim=0)+MV_pre
        return  MV
      
class DualCrossAttentionSS(nn.Module):
    def __init__(self, query_dim_MV, query_dim_Structure, heads=8, dim_head=64, dropout=0.0,num_views=5 ,bias=False,processor=XFormersCrossAttnProcessor):
        super().__init__()


        self.normStructure1 = LayerNorm32(query_dim_Structure, elementwise_affine=True, eps=1e-6)

        self.normStructure2 = LayerNorm32(query_dim_Structure, elementwise_affine=True, eps=1e-6)
       
        self.imagedownsample=Conv2DDownsample(query_dim_MV, query_dim_MV)
        
        self.input_layer_mv = nn.Linear(query_dim_MV,query_dim_MV)
        self.input_layer_stucture = nn.Linear(query_dim_Structure,query_dim_Structure)
        
        self.output_layer_stucture = nn.Linear(query_dim_Structure,query_dim_Structure)
        

        self.ffn_stucture = FeedForward(query_dim_Structure)
        
     
        self.cross_attn_MV_to_Structure = CrossAttention(query_dim_Structure, query_dim_MV, heads, dim_head, dropout, bias,processor=processor)
        self.num_views=num_views
    def forward(self, MV, structure):
        #1.input layer
        #MV(20,640,16,16)      #structure(2,4096,1024)bs  (16,16,16) dim
        structure_pre=structure
        MV=rearrange(MV,'B d h w->B h w d')
        MV=self.input_layer_mv(MV)
        MV=self.imagedownsample(rearrange(MV,'B h w d->B d h w'))
        
        MV_normal,MV_color=MV.chunk(2,dim=0)#10,640,16,16
        MV_normal=rearrange(MV_normal,'(b Nv) d h w ->b Nv d h w',Nv=self.num_views)
        MV_color=rearrange(MV_color,'(b Nv)  d h w->b Nv d h w',Nv=self.num_views)
        B,Nv,D,H, W=MV_normal.shape
        device=MV.device
        
        structure=self.input_layer_stucture(structure)
        structure=rearrange(structure, 'b (v1 v2 v3) d -> b d v1 v2 v3', v1=16, v2=16, v3=16)
        B,D2,V,_,_=structure.shape
        
        
                
        #3.get image_kv for voxel cross
        grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(V, device=device), 
                                        torch.arange(V, device=device), 
                                        torch.arange(V, device=device), indexing='ij')  

        idx_z = 15 - grid_z  
        idx_z = idx_z.unsqueeze(0).expand(B, -1, -1, -1) 
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1, -1)  
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1, -1)  
        batch_idx = torch.arange(B, device=device).view(B, 1, 1,1).expand(B, V, V,V)
        kv_0 = MV_normal[batch_idx, 0, :, idx_z, grid_x]
        kv_2 = MV_normal[batch_idx, 2, :, idx_z, grid_x] 

        kv_1 = MV_normal[batch_idx, 1, :, idx_z, grid_y]  
        kv_3 = MV_normal[batch_idx, 3, :, idx_z, grid_y] 
        image_kv=torch.stack([kv_0, kv_2,kv_1,  kv_3], dim=-2)
        image_kv=rearrange( image_kv, 'b  v1 v2 v3 L d -> (b  v1 v2 v3) L d')
        
        structure=rearrange(structure,'b d v1 v2 v3->(b v1 v2 v3) 1 d ')
        
        
        #4.cross-attention
        structure_norm=self.normStructure1(structure)
        
        batch_size_S = structure.size(0)

        sub_batch_S = batch_size_S // 4

        processed_structure = []

        for i in range(4):

            start_S = i * sub_batch_S
            end_S = (i+1) * sub_batch_S
            structure_sub = structure_norm[start_S:end_S]
            image_kv_sub = image_kv[start_S:end_S]
    
      
            out_structure = self.cross_attn_MV_to_Structure(structure_sub, image_kv_sub)
  
            processed_structure.append(out_structure)


        structure_att = torch.cat(processed_structure, dim=0)

        structure=structure+structure_att
        #ffn
        
        structure_norm=self.normStructure2(structure)

        
        structure_ffn=self.ffn_stucture(structure_norm)
        
        structure=structure+structure_ffn
        
        
        #5.output_layer
        structure=self.output_layer_stucture(structure)
        
        structure=rearrange(structure,'(b v1 v2 v3) 1 d ->b (v1 v2 v3) d ',v1=V,v2=V,v3=V)  
        structure=structure+structure_pre
        return  structure
    
