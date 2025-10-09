
from torchvision.utils import  save_image
import numpy as np
import torch
from PIL import Image
def save_coords_to_npz(coords, output_path):
   

    coords_np = coords.cpu().numpy()
    coords_np = coords_np.astype(np.float32)
    
    np.savez(
        output_path,
        coords=coords_np,
    )
def save_images(images_pred, normals_pred, save_path):
   


    for idx in range(normals_pred.shape[0]):
        save_image(normals_pred[idx], f'{save_path}/normal_{idx}.png')
    for idx in range(images_pred.shape[0]):
        save_image(images_pred[idx], f'{save_path}/color_{idx}.png')


    concat_list = []
    for idx in range(images_pred.shape[0]):
        concat_list.append(images_pred[idx])
        concat_list.append(normals_pred[idx])


    concat_tensor = torch.cat(concat_list, dim=2) 

    save_image(concat_tensor, f'{save_path}/concat.png')

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

