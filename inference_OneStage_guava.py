import os
import os.path as osp
import json
from dataclasses import dataclass
import tyro
from tqdm import tqdm
import accelerate
import torchvision
from PIL import Image as PILImage
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline
from bbox_utils import crop_image_from_bbox


def _split_frame_key(frm_key):
    *segment_id, fn = frm_key.split('_')
    return '_'.join(segment_id), fn

def _split_whole_key(whole_key):
    video_id, frame_key = whole_key.split('/', 1)
    return video_id, frame_key

def uniform_sample_frames(all_frames, frames_per_video):
    """
    Uniformly sample frames_per_video frames from each video.
    
    Args:
        all_frames: List of frame keys in format "video_id/frame_key"
        frames_per_video: Number of frames to sample per video
        
    Returns:
        List of sampled frame keys
    """
    # Group frames by video_id
    video_frames = {}
    for whole_key in all_frames:
        video_id, frame_key = _split_whole_key(whole_key)
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append(whole_key)
    
    # Uniformly sample frames from each video
    sampled_frames = []
    for video_id, frames in video_frames.items():
        frames = sorted(frames)  # Sort to ensure consistent ordering
        n_frames = len(frames)
        if n_frames <= frames_per_video:
            sampled_frames.extend(frames)
        else:
            # Uniformly sample indices
            indices = np.linspace(0, n_frames - 1, frames_per_video, dtype=int)
            sampled_frames.extend([frames[i] for i in indices])
    
    print(f"Number of videos: {len(video_frames)}")
    print(f"Number of sampled frames: {len(sampled_frames)}")
    
    return sampled_frames


class FrameDataset(Dataset):
    """Dataset for loading frames and mattes."""
    
    def __init__(self, frame_keys, extra_info):
        self.frame_keys = frame_keys
        self.extra_info = extra_info
        self.frames_root = extra_info['frames_root']
        self.matte_root = extra_info['matte_root']
        
    def __len__(self):
        return len(self.frame_keys)
    
    def __getitem__(self, idx):
        whole_key = self.frame_keys[idx]
        video_id, frame_key = _split_whole_key(whole_key)
        seg_id, fn = _split_frame_key(frame_key)
        
        # Load image and matte
        img_path = os.path.join(self.frames_root, video_id, seg_id, f'{fn}.png')
        matte_path = os.path.join(self.matte_root, video_id, seg_id, f'{fn}.png')
        
        img = np.array(PILImage.open(img_path))
        matte = np.array(PILImage.open(matte_path))
        
        # Apply bounding box crop if available
        wbbox_info = self.extra_info.get('wbbox_info')
        if wbbox_info is not None:
            if video_id in wbbox_info:
                wbbox_info = wbbox_info[video_id]
            wbbox_xyxy = wbbox_info.get(frame_key)
            if wbbox_xyxy is not None:
                wbbox_xyxy = np.array(wbbox_xyxy, dtype=np.float32).squeeze(0)
                img, _ = crop_image_from_bbox(img, wbbox_xyxy, return_pad_mask=True)
                matte, _ = crop_image_from_bbox(matte, wbbox_xyxy, return_pad_mask=True)
        
        # Create RGBA image
        if len(matte.shape) == 2:
            matte = matte[..., None]
        
        # Ensure img is RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        # remove boundary
        matte[matte < 128] = 0
        matte[matte >= 128] = 255
        
        # Combine to RGBA
        rgba = np.concatenate([img, matte], axis=-1)
        
        return {
            'image_rgba': rgba,  # Return numpy array instead of PIL
            'whole_key': whole_key
        }


def main(args):
    accelerator = accelerate.Accelerator()
    data_root = args.data_root
    output_root = args.output_root

    # load data
    with open(os.path.join(data_root, 'dataset_frames.json'), 'r') as f:
        frames = json.load(f)
        
    train_frames = frames['train']
    val_frames = frames['valid']
    test_frames = frames['test']
    all_frames = train_frames + val_frames + test_frames
    proc_frames = uniform_sample_frames(all_frames, frames_per_video=args.frames_per_video)
    proc_frames.sort()
    print("Total proc frames: ", len(proc_frames))

    if args.reverse_order:
        proc_frames = proc_frames[::-1]
        print("Reversed frame order for processing.")

    with open(os.path.join(data_root, 'extra_info.json'), 'r') as f:
        extra_info = json.load(f)

    # Create dataset and dataloader
    dataset = FrameDataset(proc_frames, extra_info)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one frame at a time
        shuffle=False,
        num_workers=2,
    )

    # Load pipeline
    pipeline = SyncHumanOneStagePipeline.from_pretrained(
        './ckpts/OneStage',
    )
    pipeline.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()

    # Prepare with accelerator
    dataloader = accelerator.prepare(dataloader)
    pipeline = accelerator.prepare(pipeline)

    # Process each frame
    for batch in tqdm(dataloader, desc="Processing frames"):
        image_rgba_np = batch['image_rgba'][0].cpu().numpy()  # Get first item from batch and convert to numpy
        whole_key = batch['whole_key'][0]
        
        # Convert numpy array to PIL Image
        image_rgba = PILImage.fromarray(image_rgba_np, mode='RGBA')
        
        # Create save path
        save_path = os.path.join(output_root, whole_key)

        # check finished
        concat_fn = osp.join(save_path, 'concat.png')
        if osp.exists(concat_fn):
            print(f"Skipping {whole_key}, already processed.")
            continue
        
        # Run pipeline
        try:
            pipeline.run(
                image_path=image_rgba,
                save_path=save_path,
            )
        except Exception as e:
            print(f"Error processing {whole_key}: {e}")
            continue
    
    print(f"Processing complete! Results saved to {output_root}")


@dataclass
class Args:
    data_root: str
    output_root: str
    frames_per_video: int = 10
    reverse_order: bool = False

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)


