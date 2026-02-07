import torch
import numpy as np

def make_bbox_from_keypoints(kpts, full_xyxy, scale=torch.tensor([1.0]), offset=torch.tensor([0, 0]), boundary=20):
    """
    Make a bounding box from keypoints.
    Args:
        kpts: (N, K, 3) tensor, where N is the number of instances, K is the number of keypoints.
        full_xyxy: (N, 4) tensor, where each row is [x1, y1, x2, y2].
        boundary: padding around the bounding box.
    """
    x_min = torch.max(torch.min(kpts[..., 0], dim=1).values - boundary, full_xyxy[:, 0])
    y_min = torch.max(torch.min(kpts[..., 1], dim=1).values - boundary, full_xyxy[:, 1])
    x_max = torch.min(torch.max(kpts[..., 0], dim=1).values + boundary, full_xyxy[:, 2])
    y_max = torch.min(torch.max(kpts[..., 1], dim=1).values + boundary, full_xyxy[:, 3])
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    scale = scale.to(boxes.device)
    offset = offset.to(boxes.device)
    return ((boxes - full_xyxy[:, :2].repeat(1, 2)) * scale[:, None] + offset.repeat(1, 2)).int()

def enlarge_bbox(bbox, img_width, img_height, scale=1.1):
    """
    Enlarge the bounding box by a scale factor.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    L = max(w, h) * scale
    x1 = max(0, x1 - (L - w) / 2.0)
    y1 = max(0, y1 - (L - h) / 2.0)
    x2 = min(img_width, x2 + (L - w) / 2.0)
    y2 = min(img_height, y2 + (L - h) / 2.0)
    return torch.tensor([x1, y1, x2, y2]).to(bbox)

def enlarge_bbox_square(bbox_xyxy, scale=1.1):
    """
    Enlarge the bounding box to a square by a scale factor.
    Args:
        bbox_xyxy: (N, 4) tensor, [[x1, y1, x2, y2], ...]
    """
    x1 = bbox_xyxy[:, 0]
    y1 = bbox_xyxy[:, 1]
    x2 = bbox_xyxy[:, 2]
    y2 = bbox_xyxy[:, 3]
    w = x2 - x1
    h = y2 - y1
    L = (torch.max(w, h) * scale).int()
    cx = ((x1 + x2) / 2.0).int()
    cy = ((y1 + y2) / 2.0).int()
    x1 = cx - L / 2.0
    y1 = cy - L / 2.0
    x2 = cx + L / 2.0
    y2 = cy + L / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1).int().to(bbox_xyxy)

def crop_image_from_bbox(img, bbox_xyxy, pad_mode='constant', pad_value=0, return_pad_mask=False):
    """
    Crop the image from the bounding box, pad to match bbox size if needed.
    Args:
        img: (H, W, 3).
        bbox_xyxy: (N, 4), where each row is [x1, y1, x2, y2].
    Returns:
        img_crop: (N, h, w, 3).
    """
    if isinstance(bbox_xyxy, torch.Tensor):
        bbox_xyxy = bbox_xyxy.cpu().numpy()
    if isinstance(bbox_xyxy, list):
        bbox_xyxy = np.array(bbox_xyxy)

    if len(bbox_xyxy.shape) == 1:
        bbox_xyxy = bbox_xyxy[None, :]
        is_single = True
    else:
        is_single = False

    H, W = img.shape[:2]
    img_crops, pad_masks = [], []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1_pad = max(0, -x1)
        y1_pad = max(0, -y1)
        x2_pad = max(0, x2 - W)
        y2_pad = max(0, y2 - H)
        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(W, x2)
        y2_clamped = min(H, y2)
        img_crop = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        pad_mask = np.ones((y2_clamped - y1_clamped, x2_clamped - x1_clamped), dtype=np.uint8)
        if (x1_pad > 0) or (y1_pad > 0) or (x2_pad > 0) or (y2_pad > 0):
            if len(img_crop.shape) == 2:
                img_crop = np.pad(img_crop, ((y1_pad, y2_pad), (x1_pad, x2_pad)), mode=pad_mode, constant_values=pad_value)
            else:
                img_crop = np.pad(img_crop, ((y1_pad, y2_pad), (x1_pad, x2_pad), (0, 0)), mode=pad_mode, constant_values=pad_value)
            pad_mask = np.pad(pad_mask, ((y1_pad, y2_pad), (x1_pad, x2_pad)), mode='constant', constant_values=0)
        img_crops.append(img_crop)
        pad_masks.append(pad_mask)

    if is_single:
        img_crops = img_crops[0]
        pad_masks = pad_masks[0]
        
    if return_pad_mask:
        return img_crops, pad_masks
    return img_crops

