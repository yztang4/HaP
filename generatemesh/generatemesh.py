import numpy as np
import open3d as o3d 
import torch
import pytorch3d
import pytorch3d.ops
from pytorch3d.io import IO
import pymeshlab

def read_generated_and_partial_pc(gp, pp):
    generated_path = gp
    partial_path = pp
    generated = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(generated_path).points))
    generated_normal = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(generated_path).normals)).unsqueeze(0)
    partial_pc = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(partial_path).points)).unsqueeze(0).float()[...,:3]
    generated = generated.unsqueeze(0).float()
    return generated, generated_normal, partial_pc

def two_stage_replace(generated, generated_normal, partial_pc, gp, pp, save_path):
    generated_path = gp
    partial_path = pp
    generated_all_idx = range(10000)
    partial_num = partial_pc.shape[1]

    dist,idx,_=pytorch3d.ops.ball_query(partial_pc.cuda(),generated.cuda(),K=50,radius=0.02,return_nn=True)         
    idx = idx.reshape(1,partial_num*50,1)
    idx = torch.unique(idx)[1:]
    unvisible_points_idx = list(set(generated_all_idx).difference(set(list(idx.cpu().numpy()))))

    tmpnum = idx.shape[0]
    idx = idx.reshape(1,tmpnum,1)
    query_knn_normal=pytorch3d.ops.knn_gather(generated.cuda(),idx=idx)
    tmpquery = query_knn_normal.reshape(1,tmpnum,3)


    pcl = query_knn_normal.reshape(tmpnum,3).cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    o3d.io.write_point_cloud("visible_test.ply", pcd)
    generated_all_idx = range(10000)
    unvisible_points_idx = np.asarray(unvisible_points_idx)

    allgeneratedpoints = (np.asarray(o3d.io.read_point_cloud(generated_path).points))
    allgeneratednormals = (np.asarray(o3d.io.read_point_cloud(generated_path).normals))

    unvisible_points = allgeneratedpoints[unvisible_points_idx,:]
    unvisible_normal = allgeneratednormals[unvisible_points_idx,:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unvisible_points)
    o3d.io.write_point_cloud("unvisible_test.ply", pcd)
    _,idx,_=pytorch3d.ops.ball_query(tmpquery.cuda(),partial_pc.cuda(),K=50,radius=0.022,return_nn=True)             #(B,M,K)    (B,M,K,3)
    idx = idx.squeeze()
    idx = idx.reshape(1,idx.shape[0]*50,1).squeeze()
    idx = torch.unique(idx)[1:]
    tmpnum = idx.shape[0]
    idx = idx.reshape(1,tmpnum,1)
    partial_replace=pytorch3d.ops.knn_gather(partial_pc.cuda(),idx=idx)
    partial_query = partial_replace.reshape(1, tmpnum, 3)

    _,idx,_=pytorch3d.ops.ball_query(partial_query.cuda(),partial_pc.cuda(),K=50,radius=0.022,return_nn=True)             #(B,M,K)    (B,M,K,3)
    idx = idx.squeeze()
    idx = idx.reshape(1,idx.shape[0]*50,1).squeeze()
    idx = torch.unique(idx)[1:]
    tmpnum = idx.shape[0]
    idx = idx.reshape(1,tmpnum,1)
    partial_replace=pytorch3d.ops.knn_gather(partial_pc.cuda(),idx=idx)
    partial_replace = partial_replace.reshape(1,tmpnum,3)
    _,idx,query_knn_pc=pytorch3d.ops.knn_points(partial_replace.cuda(),generated.cuda(),K=3,return_nn=True)             #(B,M,K)    (B,M,K,3)

    normals=pytorch3d.ops.knn_gather(generated_normal.cuda(),idx=idx)
    # query_knn_pc_local=partial_replace.unsqueeze(2)-query_knn_pc      #(B,M,K,3)
    distmat = torch.sqrt(torch.sum((partial_replace.unsqueeze(2)-query_knn_pc)**2,dim=3))

    distmatsum = torch.sum(distmat,2,keepdim=True)
    distmat = distmat/distmatsum
    distmat = distmat.unsqueeze(-1)

    estimated_normal=torch.sum(distmat*normals,dim=3,keepdim=True).squeeze()    #(B,M,K,1)

    pcl = partial_replace.reshape(tmpnum,3).cpu().numpy()
    final_replaced = torch.cat([torch.as_tensor(unvisible_points),torch.as_tensor(pcl)],0)
    final_normal = torch.cat([torch.as_tensor(unvisible_normal).cpu(),estimated_normal.cpu()],0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    o3d.io.write_point_cloud("partial_replace.ply", pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_replaced)
    pcd.normals = o3d.utility.Vector3dVector(final_normal)
    o3d.io.write_point_cloud(save_path, pcd)

def correct_normal(pc1w_path, save_path):
    import pymeshlab
    # here, compute the normal of replaced point clouds
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(save_path)
    ms.compute_normal_for_point_clouds()
    ms.save_current_mesh(save_path,save_vertex_normal = True)
    
    # here, compute the predicted normal of pdr network of 1w point clouds
    normal_pred = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(pc1w_path).normals)).unsqueeze(0).float()
    points_pred = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(pc1w_path).points)).unsqueeze(0).float()
    # here, compute the pymeshlab normal of 1w point clouds
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(pc1w_path)
    ms.compute_normal_for_point_clouds()
    ms.save_current_mesh(pc1w_path[:-4]+"pymesh.ply",save_vertex_normal = True)
    normal_pred_pymesh = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(pc1w_path[:-4]+"pymesh.ply").normals)).unsqueeze(0).float()


    normal_pymesh = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(save_path).normals)).unsqueeze(0).float()
    points_pymesh = torch.as_tensor(np.asarray(o3d.io.read_point_cloud(save_path).points)).unsqueeze(0).float()
    _,idx,_=pytorch3d.ops.knn_points(points_pred,points_pymesh,K=1,return_nn=True,return_sorted=False)  
    query_1nn_normal=pytorch3d.ops.knn_gather(normal_pymesh,idx=idx) 

    refine_daxiao = torch.mul(normal_pred.squeeze(),query_1nn_normal.squeeze()).sum(dim=1).sum()

    
    if refine_daxiao<0:
        normal_pymesh*=-1
    
    panduan = torch.mul(normal_pred,normal_pred_pymesh).sum(dim=1).sum()
    if panduan<0:
        normal_pred_pymesh*=-1

    tmpnormal = []

    for i in range(10000):
        if points_pred[0][i][1]<-0.95:
            tmpnormal.append(normal_pred[0][i].numpy())
            # finalpoints.append(points[i].numpy())
        else:
            tmpnormal.append(normal_pred_pymesh[0][i].numpy())
    tmpnormal = torch.as_tensor(np.asarray(tmpnormal)).unsqueeze(0)

    # stage_2
    _,idx,_=pytorch3d.ops.knn_points(points_pymesh,points_pred,K=1,return_nn=True,return_sorted=False)  
    query_1nn_normal=pytorch3d.ops.knn_gather(tmpnormal,idx=idx) 
    refine_daxiao2 = torch.mul(normal_pymesh.squeeze(),query_1nn_normal.squeeze()).sum(dim=1)
    _,idx,_=pytorch3d.ops.knn_points(points_pymesh,points_pred,K=1,return_nn=True,return_sorted=False)  
    query_1nn_normal=pytorch3d.ops.knn_gather(normal_pred,idx=idx) 
    refine_daxiao3 = torch.mul(normal_pymesh.squeeze(),query_1nn_normal.squeeze()).sum(dim=1)
    tmpidx = np.argwhere(refine_daxiao2.numpy()<0)
    tmpidx2 = np.argwhere(refine_daxiao3.numpy()<0)
    
    if tmpidx.squeeze().shape[0]>3000 or tmpidx2.squeeze().shape[0]>5000:
        for dd in tmpidx2:
            normal_pymesh[0][dd] *=-1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_pymesh.squeeze().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normal_pymesh.squeeze().numpy())

    o3d.io.write_point_cloud(save_path, pcd)
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(save_path)
    ms.generate_surface_reconstruction_screened_poisson()
    

    ms.save_current_mesh(save_path[:-4]+"possion.ply")

import os

generated_pc_path = "/public/sde/tangyingzhi/Project/generate_poisson/generatemesh/examples/wildgenerated.ply"

ms = pymeshlab.MeshSet()
ms.load_new_mesh(generated_pc_path)
ms.compute_normal_for_point_clouds()
ms.save_current_mesh(generated_pc_path,save_vertex_normal = True)



partial_pc_path = "/public/sde/tangyingzhi/Project/generate_poisson/generatemesh/examples/wildpartial.ply"
save_path = "/public/sde/tangyingzhi/Project/generate_poisson/generatemesh/examples/wildfinal.ply"
generated, generated_normal, partial_pc = read_generated_and_partial_pc(generated_pc_path,partial_pc_path)
two_stage_replace(generated, generated_normal, partial_pc, generated_pc_path, partial_pc_path,save_path)
correct_normal(generated_pc_path, save_path)
