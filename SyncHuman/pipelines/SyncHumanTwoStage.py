from typing import *
import torch
from torchvision import transforms
import torch.nn.functional as F
from ..utils import  postprocessing_utils
import os
import math
from PIL import Image
import json
from . import samplers
from ..models import SecondStage as models
from ..modules import sparse as sp
import numpy as np
class SyncHumanTwoStagePipeline:    
    def __init__(
        self,
        models,
    ):     
        
        
            
        self.models = models
        for model in self.models.values():
            model.eval()
        self._init_image_cond_model()
          
        self.slat_sampler = None
        self.slat_sampler_params = {}
        self.slat_normalization = None
        
        self.img_wh_mv=(768,768)
        self.face_wh=(512,512)
        
    @staticmethod
    def from_pretrained(path: str) -> "SyncHumanTwoStagePipeline":

        is_local = os.path.exists(f"{path}/pipeline.json")
        

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except:
                _models[k] = models.from_pretrained(v)

        pipeline = SyncHumanTwoStagePipeline(_models)
        pipeline._pretrained_args = args

        pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        pipeline.slat_sampler_params = args['slat_sampler']['params']

        pipeline.slat_normalization = args['slat_normalization']

        pipeline._init_image_cond_model()

        return pipeline
        
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
    
    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))
            
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg', pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform     
    
    
    
    def load_image_rgb(self, img_path, return_type='np'): 
        rgb = np.array(Image.open(img_path).resize(self.img_wh_mv))
        rgb = rgb.astype(np.float32) / 255. # [0, 1]
        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(rgb)
        else:
            raise NotImplementedError
        return img

 
        
    def _get_mv_generate(self, root):
       
        pred_vids=[0,1,2,3]
        img_tensors_out = []
        normal_tensors_out = []
        for i, vid in enumerate(pred_vids):
            # output image
            img_tensor= self.load_image_rgb(os.path.join(root,f"color_{vid}.png"), return_type='pt').permute(2, 0, 1)
            img_tensors_out.append(img_tensor)
            
            # output normal
            normal_tensor = self.load_image_rgb(os.path.join(root,f"normal_{vid}.png"), return_type="pt").permute(2, 0, 1)
            normal_tensors_out.append(normal_tensor)
            
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float() # (Nv, 3, H, W)
        #get face image
        normal_faces=np.ones((3,self.face_wh[1]*3, self.face_wh[0]*3), dtype=np.float32)
        img_faces = np.ones((3,self.face_wh[1]*3, self.face_wh[0]*3), dtype=np.float32)
        img_faces_load= self.load_image_rgb(os.path.join(root,"color_4.png"), return_type='pt').permute(2, 0, 1).float()
        normal_faces_load = self.load_image_rgb(os.path.join(root,"normal_4.png"), return_type="pt").permute(2, 0, 1).float()
        img_faces_load = F.interpolate(img_faces_load.unsqueeze(0), size=(self.face_wh[1], self.face_wh[0]), mode='bilinear', align_corners=False).squeeze(0)
        normal_faces_load = F.interpolate(normal_faces_load.unsqueeze(0), size=(self.face_wh[1], self.face_wh[0]), mode='bilinear', align_corners=False).squeeze(0)
        
        img_faces[:,:self.face_wh[0],self.face_wh[0]:self.face_wh[0]*2]=img_faces_load
        normal_faces[:,:self.face_wh[0],self.face_wh[0]:self.face_wh[0]*2]=normal_faces_load
        
        return {
            'mv_img': img_tensors_out,
            'mv_normal': normal_tensors_out,
            'faces_img':img_faces,
            'faces_normal': normal_faces
        }   
    def _get_coords_gen(self, root):
      
        data = np.load(os.path.join(root, 'latent.npz'))
        coords = torch.tensor(data['coords']).int()
        return coords 
    
    
    
    def preprocess_image(self,  root) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        input=Image.open(root)
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return   output
        
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:

        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
       
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
  
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
        
    @torch.no_grad()
    def encode_image_mv(self, image: Union[torch.Tensor, List[Image.Image]],up_size=896) -> torch.Tensor:
        """
        Encode the image.
        """
        BS, num_views, C, H, W = image.shape
        image = F.interpolate(image.reshape(BS*num_views, C, H, W), size=(up_size, up_size), 
                mode="bilinear",  align_corners=False)
        
        image = self.image_cond_model_transform(image).to(self.device)
        features=self.models['image_cond_model'].get_intermediate_layers(image,[4,11,17,23])
        features=torch.cat(list(features),dim=-1)
     
        out_shape =int( math.sqrt(features.shape[1]))
        patchtokens =features.reshape(BS,num_views,out_shape,out_shape,features.shape[-1]) 
        return patchtokens
    
    def get_cond_mv(self, cond):
        """
        Get the conditioning data.
        """
        cond = self.encode_image_mv(cond)
        return cond

  

    def run(
        self,
        image_path,
        outpath,
      
    ) -> Dict:

        with torch.no_grad(): 
                
            mv_generate=self._get_mv_generate(image_path)
            coords=self._get_coords_gen(image_path)
   
            #get the structured latent
            image = self.preprocess_image(os.path.join(image_path,'input.png'))
            cond = self.get_cond([image])
            slat = self.sample_slat(cond, coords.cuda())
                
                
            #get mulitiview normal and color
            mv_img = self.get_cond_mv(mv_generate['mv_img'].unsqueeze(0).cuda())
            mv_normal=self.get_cond_mv(mv_generate['mv_normal'].unsqueeze(0).cuda())
                
 

            gs = self.models['slat_decoder_gs'](slat, mv_img, mv_normal)
            mesh = self.models['slat_decoder_mesh'](slat, mv_img, mv_normal)
      
                
        os.makedirs(outpath, exist_ok=True)
        glb = postprocessing_utils.to_glb(
                gs[0],
                mesh[0],
                mv_generate['mv_img'],
                mv_generate['mv_normal'],
                mv_generate['faces_img'],
                mv_generate['faces_normal'],
                simplify=0.7,          
                texture_size=1024,     
            )
        glb.export(os.path.join(outpath,"ouput.glb"))
     

     



