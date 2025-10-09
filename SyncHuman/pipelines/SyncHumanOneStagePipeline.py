
from typing import  List, Optional, Union, Dict, Any
import PIL
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding

from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import  ImagePipelineOutput

import os
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from PIL import Image
import rembg
import torch.nn.functional as F
from torchvision import transforms
import json

import importlib
import json
import os
from pathlib import Path

import torch
from transformers import  CLIPVisionModelWithProjection, CLIPFeatureExtractor,CLIPTextModel

from ..utils.inference_utils import save_coords_to_npz,save_images,add_margin
from einops import rearrange
from ..utils.voxel_utils import writeocc
logger = logging.get_logger(__name__)


    

class SyncHumanOneStagePipeline:
    
    def __init__(
        self,
        sparse_structure_decoder,
        dinov2_model,
        feature_extractor: CLIPFeatureExtractor,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModel,
        SyncHuman_2D3DCrossSpaceDiffusion: UNet2DConditionModel,
        vae: AutoencoderKL,
        num_views: int = 5,
        device='cuda',
        dtype=torch.float16,
        mv_bg_color: str='white',
        mv_img_wh=(768,768),
        mv_crop_size=768

    ):
        super().__init__()
        
        
        self.register_modules(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            SyncHuman_2D3DCrossSpaceDiffusion=SyncHuman_2D3DCrossSpaceDiffusion,
            vae=vae,
            dinov2_model=dinov2_model,
            sparse_structure_decoder=sparse_structure_decoder,
        )
        
      

        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.num_views: int = num_views
        self.to(device,dtype)
        
        normal_prompt_embedding = torch.load('SyncHuman/data/fixed_prompt_embeds_7view/normal_embeds.pt')
        color_prompt_embedding = torch.load('SyncHuman/data/fixed_prompt_embeds_7view/clr_embeds.pt')
  
        self.normal_prompt_embedding = torch.stack([normal_prompt_embedding[0], normal_prompt_embedding[2], normal_prompt_embedding[3], normal_prompt_embedding[4], normal_prompt_embedding[6]] , 0)
        self.color_prompt_embedding = torch.stack([color_prompt_embedding[0], color_prompt_embedding[2], color_prompt_embedding[3], color_prompt_embedding[4], color_prompt_embedding[6]] , 0)
        
        prompt_embeddings = torch.cat([self.normal_prompt_embedding.unsqueeze(0), self.color_prompt_embedding.unsqueeze(0)], dim=0)
        self.prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
        self.mv_bg_color=mv_bg_color
        self.get_bg_color()
        self.mv_img_wh = mv_img_wh
        self.mv_crop_size=mv_crop_size

        
    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            setattr(self, name, module)  
    
    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | os.PathLike,
        device='cuda',
        dtype=torch.float16,
    ) -> "SyncHumanOneStagePipeline":
        
        load_dir = Path(load_directory)
        cfg_path = load_dir / "pipeline_config.json"

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = json.load(f)

        num_views = cfg.get("num_views", 5)
        device= cfg.get("device",'cuda')
        dtype = cfg.get("dtype", torch.float16)
        metadata: Dict[str, Dict[str, Any]] = cfg["metadata"]

        def _import_class(path: str):
            module_name, cls_name = path.rsplit(".", 1)
            return getattr(importlib.import_module(module_name), cls_name)

        components: Dict[str, Any] = {}

        for name, meta in metadata.items():
        
            subdir = load_dir / meta["subdir"]
            module_cls = _import_class(meta["class"])
            module = module_cls.from_pretrained(
                    subdir,
                )
            components[name] = module
        dinov2_model=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14_reg',pretrained=True)

        pipe = cls(
            sparse_structure_decoder=components["sparse_structure_decoder"],
            dinov2_model=dinov2_model,
            feature_extractor=components["feature_extractor"],
            image_encoder=components["image_encoder"],
            text_encoder=components["text_encoder"],
            SyncHuman_2D3DCrossSpaceDiffusion=components["SyncHuman_2D3DCrossSpaceDiffusion"],
            vae=components["vae"],
            num_views=num_views,
            device= device,
            dtype = dtype
        )

        return pipe
   
    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.dinov2_model.to(dtype=dtype)
            self.image_encoder.to(dtype=dtype)
            self.text_encoder.to(dtype=dtype)
            self.SyncHuman_2D3DCrossSpaceDiffusion.to(dtype=dtype)
            self.vae.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.sparse_structure_decoder.to(device)
            self.dinov2_model.to(device)
            self.image_encoder.to(device)
            self.text_encoder.to(device)
            self.SyncHuman_2D3DCrossSpaceDiffusion.to(device)
            self.vae.to(device)
      
    @property
    def _execution_device(self):
        if not hasattr(self.SyncHuman_2D3DCrossSpaceDiffusion, "_hf_hook"):
            return self.device
        for module in self.SyncHuman_2D3DCrossSpaceDiffusion.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        device,
        do_classifier_free_guidance,
        prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)


        if do_classifier_free_guidance:
            normal_prompt_embeds, color_prompt_embeds = torch.chunk(prompt_embeds, 2, dim=0)
            prompt_embeds = torch.cat([normal_prompt_embeds, normal_prompt_embeds, color_prompt_embeds, color_prompt_embeds], 0)

        return prompt_embeds

    def _encode_image(
        self,
        image_pil,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        noise_level: int=0,
        generator: Optional[torch.Generator] = None
    ):
        dtype = next(self.image_encoder.parameters()).dtype
        # ______________________________clip image embedding______________________________ 
        image = self.feature_extractor(images=image_pil, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        
        image_embeds = self.get_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
            )

        image_embeds = image_embeds.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            normal_image_embeds, color_image_embeds = torch.chunk(image_embeds, 2, dim=0)
            negative_prompt_embeds = torch.zeros_like(normal_image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, normal_image_embeds, negative_prompt_embeds, color_image_embeds], 0)
            
        # _____________________________vae input latents__________________________________________________
        def vae_encode(tensor):
            image_pt = torch.stack([TF.to_tensor(img) for img in tensor], dim=0).to(device)
            image_pt = image_pt * 2.0 - 1.0
            image_latents = self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor
            # Note: repeat differently from official pipelines     
            image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)
            return image_latents
        
        image_latents = vae_encode(image_pil)
              
        if do_classifier_free_guidance:
            normal_image_latents, color_image_latents = torch.chunk(image_latents, 2, dim=0)
            image_latents = torch.cat([torch.zeros_like(normal_image_latents), normal_image_latents, 
                                       torch.zeros_like(color_image_latents), color_image_latents], 0)

        return image_embeds, image_latents



    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator=None, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents

    def get_image_embeddings(
        self,
        image_embeds: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
                   
            ):
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )
        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )
        noise_level = noise_level.to(image_embeds.dtype)
        image_embeds = torch.cat((image_embeds, noise_level), 1)
        return image_embeds

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
        features = self.dinov2_model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond).to(dtype=self.dtype)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
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
        return output

    def get_bg_color(self):
        if self.mv_bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.mv_bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.mv_bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.mv_bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.mv_bg_color, float):
            bg_color = np.array([self.mv_bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        self.mv_bg_color=bg_color
      
    

    def load_img_face(self, img_path,  return_type='np', Imagefile=None):
       
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.mv_img_wh[0]

        if self.mv_crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.mv_crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        face_input = image_input.crop((256, 0, 512, 256)).resize((self.mv_img_wh[0], self.mv_img_wh[1]))
        
     
        img = np.array(image_input)
        face=np.array(face_input)

        img = img.astype(np.float32) / 255. # [0, 1]
        face = face.astype(np.float32) / 255. # [0, 1]

        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        img = img[...,:3] * alpha + self.mv_bg_color * (1 - alpha)

        alpha_face = face[...,3:4]
        face = face[...,:3] * alpha_face + self.mv_bg_color * (1 - alpha_face)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            face = torch.from_numpy(face)
        else:
            raise NotImplementedError
        
        return img, face 
    
    def get_mv_input(self,raw_img_path):
        image,face =self.load_img_face(raw_img_path,return_type='pt')
        img_tensors_in = [
            image.permute(2, 0, 1)
        ] * (self.num_views-1) + [
            face.permute(2, 0, 1)
        ]
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() 
        return img_tensors_in


    @torch.no_grad()
    def run_model(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image],
        image_raw:PIL.Image.Image,
        prompt: Union[str, List[str]],   
        prompt_embeds: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 10,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
        image_embeds: Optional[torch.FloatTensor] = None,
        rescale_t:Optional[int]= 3.0,
        verbose: bool = True,
        
    ):
       
    
        # 0. Default height and width 
        height = height or self.SyncHuman_2D3DCrossSpaceDiffusion.config.sample_size * self.vae_scale_factor
        width = width or self.SyncHuman_2D3DCrossSpaceDiffusion.config.sample_size * self.vae_scale_factor


        # 1. Define call parameters
        if isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
            assert batch_size >= self.num_views and batch_size % self.num_views == 0
        elif isinstance(image, PIL.Image.Image):
            image = [image]*self.num_views*2
            batch_size = self.num_views*2

        if isinstance(prompt, str):
            prompt = [prompt] * self.num_views * 2

     
        do_classifier_free_guidance = guidance_scale != 1.0#True

        # 2. Encode input prompt
        
        prompt_embeds = self._encode_prompt(
            device=self.device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )
        

        # 3. Encoder input image
        if isinstance(image, list):
            image_pil = image
        elif isinstance(image, torch.Tensor):
            image_pil = [TF.to_pil_image(image[i]) for i in range(image.shape[0])]
        noise_level = torch.tensor([noise_level], device=self.device)
        image_embeds, image_latents = self._encode_image(
            image_pil=image_pil,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
        )

        # 4. Prepare timesteps
        t_seq = np.linspace(1, 0, num_inference_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(num_inference_steps))
       
        # 5. Prepare latent variables
        num_channels_latents = self.SyncHuman_2D3DCrossSpaceDiffusion.config.out_channels
        
        latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=self.device,
                latents=latents,
            )
        
        image_trellis = self.preprocess_image(image_raw)
        cond_trellis = self.get_cond([image_trellis])
        reso = 16
        trellis_latent = torch.randn(1, 8, reso, reso, reso).to(self.device)
        

        # 6. Denoising loop
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            
            if do_classifier_free_guidance:
                normal_latents, color_latents = torch.chunk(latents, 2, dim=0)  
                latent_model_input = torch.cat([normal_latents, normal_latents, color_latents, color_latents], 0)
                trellis_latent_input=torch.concat([trellis_latent,trellis_latent],dim=0)
            else:
                latent_model_input = latents
            latent_model_input = torch.cat([
                    latent_model_input, image_latents
                ], dim=1)
  

            # predict the noise residual
            diffusion_out = self.SyncHuman_2D3DCrossSpaceDiffusion(
                latent_model_input,
                t*1000,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                trellis_x=trellis_latent_input,
                trellis_t=torch.tensor([1000 * t] * trellis_latent.shape[0], device=self.device,dtype=self.dtype),
                trellis_cond=torch.cat([cond_trellis["neg_cond"],cond_trellis["cond"]],dim=0),
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False)
            
            v_pred = diffusion_out[0]
            v_pre_trellis=diffusion_out[1]
          
                
            # perform guidance
            if do_classifier_free_guidance:
                normal_v_pred_uncond, normal_v_pred_text, color_v_pred_uncond, color_v_pred_text = torch.chunk(v_pred, 4, dim=0)
                v_pred_uncond, v_pred_text = torch.cat([normal_v_pred_uncond, color_v_pred_uncond], 0), torch.cat([normal_v_pred_text, color_v_pred_text], 0)
                v_pre_trellis_uncond,v_pre_trellis_cond= torch.chunk(v_pre_trellis, 2, dim=0)
                
                v_pred = v_pred_uncond + guidance_scale * (v_pred_text - v_pred_uncond)
                v_pre_trellis=  v_pre_trellis_uncond + guidance_scale * (v_pre_trellis_cond -  v_pre_trellis_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = latents - (t - t_prev) * v_pred
            trellis_latent =  trellis_latent - (t - t_prev) *  v_pre_trellis
        # 7. Post-processing
        if not output_type == "latent":
            if num_channels_latents == 8:
                latents = torch.cat([latents[:, :4], latents[:, 4:]], dim=0)
            with torch.no_grad():
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        
        sparse_structure=self.sparse_structure_decoder(trellis_latent)

        voxel=(sparse_structure>0).int()[0][0]
        coords = torch.argwhere(sparse_structure>0)[:, [0, 2, 3, 4]].int()
        
    
        
        if not return_dict:
            return (image,voxel)
        return ImagePipelineOutput(images=image), voxel,coords

    @torch.no_grad()
    def run(
        self,
        image_path: str,
        save_path: str,
        seed: int =43,
        guidance_scale:float = 3.0
       ):
        torch.manual_seed(seed)
    
        image_raw=Image.open(image_path)
        imgs_in = torch.cat([self.get_mv_input(image_path)]*2, dim=0)
        
            
        with torch.autocast("cuda"):
                
            out ,voxel,coords= self.run_model(
                imgs_in,image_raw, None, prompt_embeds=self.prompt_embeddings, 
                guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1,
            )
            out=out.images
            bsz = out.shape[0] // 2
            normals_pred = out[:bsz]
            images_pred = out[bsz:] 
                
            images_pred[0] = imgs_in[0]
            normals_face = F.interpolate(normals_pred[-1].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
            normals_pred[0][:, :256, 256:512] =  normals_face 
                    
            os.makedirs(save_path, exist_ok=True)
                    
             
            image_raw.save(os.path.join(save_path,'input.png'))
            save_images(images_pred, normals_pred, save_path)
            save_coords_to_npz(coords, os.path.join(save_path,'latent.npz'))
            voxels = voxel.unsqueeze(0)
            v = voxels.cpu().numpy()
            writeocc(v, save_path, "voxel.ply")


        torch.cuda.empty_cache()    
        

