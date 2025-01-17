from pointnet2_ops import pointnet2_utils
import torch
import torch.nn as nn
# from patchdiscriminator import PatchDiscriminator
# from pc_util import extract_knn_patch
import numpy as np
import os
from tqdm import tqdm
import open3d as o3d
import h5py
import fpsample


all_smpl_pcs = []

all_partial_pcs = []
# for person in persons:
nump = 1024

smpl_pcs = []

partial_pcs = []
objpath = "your_obj_path"
partial_pc_path = "your_ply_path"

obj = o3d.io.read_triangle_mesh(objpath)
R = obj.get_rotation_matrix_from_xyz((np.pi, 0, 0))
obj.rotate(R, center=(0, 0, 0))
vertices = torch.from_numpy(np.asarray(obj.vertices)).float().cuda()

pc = vertices.reshape(1,vertices.shape[0],vertices.shape[1])
point_indexes = pointnet2_utils.furthest_point_sample(pc.contiguous(), nump)

pc = pc.squeeze()[point_indexes[0].long(),:].cpu().numpy()
smpl_pcs.append(pc)

partial_pc = o3d.io.read_point_cloud(partial_pc_path)
R = partial_pc.get_rotation_matrix_from_xyz((np.pi, 0, 0))
partial_pc.rotate(R, center=(0, 0, 0))

points = np.asarray(partial_pc.points)
colors = np.array(partial_pc.colors)
fps_npdu_kdtree_samples_idx = fpsample.bucket_fps_kdline_sampling(points, 8192, h=9)
print(colors.shape)
pointsss = points[fps_npdu_kdtree_samples_idx]
colorsss = colors[fps_npdu_kdtree_samples_idx][...,:3]
pointcolor = np.concatenate((pointsss,colorsss),1)
partial_pcs.append(pointcolor)

smpl_pcs = np.asarray(smpl_pcs)
print(smpl_pcs.shape)
B=smpl_pcs.shape[0]


partial_pcs = np.asarray(partial_pcs)




smplcolor = np.zeros((1,nump,3))
smpl_pcs = np.concatenate((smpl_pcs,smplcolor),2)



all_smpl_pcs.append(smpl_pcs)
all_partial_pcs.append(partial_pcs)
depth_points = partial_pcs

all_smpl_pcs = np.array(all_smpl_pcs).reshape(1,1024,6)
all_partial_pcs = np.array(all_partial_pcs).reshape(1,8192,6)
print(all_smpl_pcs.shape)
print(all_partial_pcs.shape)
all_points = np.zeros((1,10000,3))
f = h5py.File("firststage.h5","w")
f.create_dataset("smpl_points",data=all_smpl_pcs)
f.create_dataset("all_points",data=all_points)
f.create_dataset("depth_points",data=all_partial_pcs)
f.close()

