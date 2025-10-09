import torch



def project_and_normalize(ref_grid, src_proj, length):
    """

    @param ref_grid: b 3 n
    @param src_proj: b 4 4
    @param length:   int
    @return:  b, n, 2
    """
    src_grid = src_proj[:, :3, :3] @ ref_grid + src_proj[:, :3, 3:] # b 3 n
    div_val = src_grid[:, -1:]
    div_val[div_val<1e-4] = 1e-4
    src_grid = src_grid[:, :2] / div_val # divide by depth (b, 2, n)
    src_grid[:, 0] = src_grid[:, 0]/((length - 1) / 2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((length - 1) / 2) - 1 # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1) # (b, n, 2)
    return src_grid


def construct_project_matrix(x_ratio, y_ratio, Ks, poses):
    """
    @param x_ratio: float
    @param y_ratio: float
    @param Ks:      b,3,3
    @param poses:   b,3,4
    @return:
    """
    rfn = Ks.shape[0]
    scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=Ks.dtype, device=Ks.device)
    scale_m = torch.diag(scale_m)
    ref_prj = scale_m[None, :, :] @ Ks @ poses  # rfn,3,4
    pad_vals = torch.zeros([rfn, 1, 4], dtype=Ks.dtype, device=ref_prj.device)
    pad_vals[:, :, 3] = 1.0
    ref_prj = torch.cat([ref_prj, pad_vals], 1)  # rfn,4,4
    return ref_prj

def get_warp_coordinates(volume_xyz, warp_size, input_size, Ks, warp_pose):
    B, _, D, H, W = volume_xyz.shape
    ratio = warp_size / input_size
    warp_proj = construct_project_matrix(ratio, ratio, Ks, warp_pose) # B,4,4
    warp_coords = project_and_normalize(volume_xyz.view(B,3,D*H*W), warp_proj, warp_size).view(B, D, H, W, 2)
    return warp_coords


