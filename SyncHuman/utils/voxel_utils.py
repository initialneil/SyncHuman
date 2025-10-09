import numpy as np
import os
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
from .cam_utils import get_warp_coordinates


def write_ply(points, face_data, filename, text=True):

    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]

    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (4,))])
    face['vertex_indices'] = face_data

    ply_faces = PlyElement.describe(face, 'face')
    ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ply_vertexs, ply_faces], text=text).write(filename)

def occ2points(voxel):
    voxel = voxel.squeeze()
    points  = []
    total_num = 1
    for i in range(voxel.ndim):
        total_num *= voxel.shape[i]
    
    x, y, z = np.indices(voxel.shape)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    v = voxel.reshape(-1)

    for i in range(total_num):
        if v[i] == True:
            points.append(np.array([x[i],y[i],z[i]]))
 
    return np.array(points)

def occ2points_v2(voxel):
    voxel = voxel.squeeze()
    points  = []
    total_num = 1
    for i in range(voxel.ndim):
        total_num *= voxel.shape[i]
    
    x, y, z = np.indices(voxel.shape)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    v = voxel.reshape(-1)

    xyz = np.vstack([x, y, z]).T
    occ_mask = v == True
    points = xyz[occ_mask]
 
    return np.array(points)

def generate_faces(points):
    corners = np.zeros((8*len(points),3))
    faces = np.zeros((6*len(points),4))
    for index in range(len(points)):
        corners[index*8]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]-0.5])
        corners[index*8+1]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]-0.5])
        corners[index*8+2]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]-0.5])
        corners[index*8+3]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]-0.5])
        corners[index*8+4]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]+0.5])
        corners[index*8+5]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]+0.5])
        corners[index*8+6]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]+0.5])
        corners[index*8+7]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]+0.5])
        base=len(points)+8*index
        faces[index*6]= np.array([base+2, base+3,base+1,base+0])
        faces[index*6+1]= np.array([base+4, base+5, base+7,base+6])
        faces[index*6+2]= np.array([base+3, base+2, base+6,base+7])
        faces[index*6+3]= np.array([base+0, base+1, base+5,base+4])
        faces[index*6+4]= np.array([base+2, base+0,base+4,base+6])
        faces[index*6+5]= np.array([base+1, base+3,base+7,base+5])
    
    return corners, faces



def writeocc(voxel, save_path=".", filename="debug.ply"):
    points = occ2points(voxel)
    #print(points.shape)
    corners, faces = generate_faces(points)
    if points.shape[0] == 0:
        print('the predicted mesh has zero point!')
    else:
        points = np.concatenate((points,corners),axis=0)
        write_ply(points, faces, os.path.join(save_path,filename))



def project_condition(cond, w2cs, intrinsics, resolution=64, height=518, width=518, spatial_volume_length=0.6):
    dino_features = cond[:, 5:] # remoce CLS token and register token

    V = resolution
    B, _, C = dino_features.shape
    dino_h = height // 14
    dino_w = width // 14

    dino_features = dino_features.view(B, dino_h, dino_w, C).permute(0, 3, 1, 2)

    device = cond.device
    dtype = cond.dtype

    spatial_volume_verts = torch.linspace(-spatial_volume_length, spatial_volume_length, V, dtype=dtype, device=device) # range [-1, 1] by default
    spatial_volume_verts = torch.stack(torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
    spatial_volume_verts = spatial_volume_verts.reshape(1, V ** 3, 3)[:, :, (2, 1, 0)]  # change [B C D(Z) H(Y) W(X)] to [B C X Y Z]  
    spatial_volume_verts = spatial_volume_verts.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1) # B,3,V,V,V

    coords_source = get_warp_coordinates(spatial_volume_verts, dino_h, height, intrinsics, w2cs).view(B, V, V * V, 2)

    nan_num = torch.isnan(coords_source).sum()
    if nan_num > 0:
        print(f"nan_num: {nan_num}")

    unproj_feats_ = F.grid_sample(dino_features, coords_source, mode='bilinear', padding_mode='zeros', align_corners=True)
    unproj_feats_ = unproj_feats_.view(B, C, V, V, V)

    return unproj_feats_