# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import os
import torch
import matplotlib.pyplot as plt
import open3d as o3d 
import numpy as np 
from pytorch3d.transforms import RotateAxisAngle
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
import torch.nn as nn 
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer.mesh.textures import Textures
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import logging
from tqdm.auto import tqdm
from lib.common.render import query_color, image2vid
from lib.common.config import cfg
from lib.common.cloth_extraction import extract_cloth
from lib.dataset.mesh_util import (
    load_checkpoint,
    update_mesh_shape_prior_losses,
    get_optim_grid_image,
    blend_rgb_norm,
    unwrap,
    remesh,
    tensor2variable,
)
from pytorch3d import transforms
from pytorch3d.io import IO
from pytorch3d.io import load_ply
from lib.dataset.RefineSMPLDataset import TestDataset
from lib.net.local_affine import LocalAffine
from pytorch3d.structures import Meshes
from apps.ICON import ICON

# from pytorch3d.loss import point_mesh_face_distance
from apps.newp2f import point_mesh_face_distance
import os
from termcolor import colored
import argparse
import numpy as np
import numpy as np
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
import open3d as o3d 

import torch
torch.backends.cudnn.benchmark = True

logging.getLogger("trimesh").setLevel(logging.ERROR)


if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-colab", action="store_true")
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=2000)
    parser.add_argument("-patience", "--patience", type=int, default=10)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=10)
    parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=200)
    parser.add_argument("-hps_type", "--hps_type", type=str, default="pixie")
    parser.add_argument("-  ", action="store_true")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="smplestimation/example/")
    parser.add_argument("-out_dir", "--out_dir",
                        type=str, default="./results")
    parser.add_argument('-seg_dir', '--seg_dir', type=str, default=None)
    parser.add_argument(
        "-cfg", "--config", type=str, default="./configs/icon-filter.yaml"
    )

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymaf/configs/pymaf_config.yaml")

    cfg_show_list = [
        "test_gpus",
        [args.gpu_device],
        "mcube_res",
        256,
        "clean_mesh",
        True,
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    device = torch.device(f"cuda:{args.gpu_device}")

    # load model and dataloader
    # model = ICON(cfg)
    # model = load_checkpoint(model, cfg)

    dataset_param = {
        'image_dir': args.in_dir,
        'seg_dir': args.seg_dir,
        'colab': args.colab,
        'hps_type': args.hps_type   # pymaf/pare/pixie
    }

    def VF2Mesh(verts, faces):

        if not torch.is_tensor(verts):
            verts = torch.tensor(verts)
        if not torch.is_tensor(faces):
            faces = torch.tensor(faces)

        if verts.ndimension() == 2:
            verts = verts.unsqueeze(0).float()
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0).long()

        verts = verts.to(device)
        faces = faces.to(device)

        mesh = Meshes(verts, faces).to(device)

        mesh.textures = TexturesVertex(
            verts_features=(mesh.verts_normals_padded() + 1.0) * 0.5
        )

        return mesh

    import torch

    def torch_normalize_pc(data,B):

        all_points_mean = (
                        (torch.amax(data, dim=1)).reshape(B, 1, 3) +
                        (torch.amin(data, dim=1)).reshape(B, 1, 3)
                    ) / 2
        all_points_std = torch.amax((
                        (torch.amax(data, dim=1)).reshape(B, 1, 3) -
                        (torch.amin(data, dim=1)).reshape(B, 1, 3)
                    ), axis=-1).reshape(B, 1, 1) / 2
        data = (data-all_points_mean)/all_points_std

        return data

    if args.hps_type == "pixie" and "pamir" in args.config:
        print(colored("PIXIE isn't compatible with PaMIR, thus switch to PyMAF", "red"))
        dataset_param["hps_type"] = "pymaf"

    dataset = TestDataset(dataset_param, device)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))
    
    pbar = tqdm(dataset)
    for data in pbar:
        # print(data)
        pbar.set_description(f"{data['name']}")
        
        in_tensor = {"name": data['name'], "smpl_faces": data["smpl_faces"], "image": data["image"], "mask": data["mask"]}
        print(in_tensor['name'])
        
        partial_pc = o3d.io.read_point_cloud("smplestimation/example/"+in_tensor['name']+".ply")
        partial_pc = torch.from_numpy(np.asarray(partial_pc.points)).to(device).squeeze().unsqueeze(0).float()
        # partial_pc = load_ply("/home/tang_21/Project/ICON/thuman_example/"+in_tensor['name']+".ply")

        partial_pc2 = partial_pc
        partial_pc = Pointclouds(partial_pc)

        smpl_verts, _, original_joints = dataset.smpl_model(
                    shape_params=tensor2variable(data["betas"], device),
                    expression_params=tensor2variable(data["exp"], device),
                    body_pose=tensor2variable(data["body_pose"], device),
                    global_pose=tensor2variable(data["global_orient"], device),
                    jaw_pose=tensor2variable(data["jaw_pose"], device),
                    left_hand_pose=tensor2variable(
                        data["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(
                        data["right_hand_pose"], device),
                )
        

        original_joints = original_joints.detach()
        mesh = VF2Mesh(smpl_verts, in_tensor["smpl_faces"])
        IO().save_mesh(mesh,"smplestimation/example/initial_"+in_tensor['name']+".obj")

        verts, faces_idx, _ = load_obj("smplestimation/example/initial_"+in_tensor['name']+".obj")
        faces = faces_idx.verts_idx

      
        verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb.to(device))

      
        mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
        )
        R, T = look_at_view_transform(2.2, 0, 180) 
        
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0,
            faces_per_pixel=1, 
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )

        # Get the output from rasterization

        fragments = rasterizer(mesh)

        pix_to_face = fragments.pix_to_face  
        # (F, 3) where F is the total number of faces across all the meshes in the batch
        packed_faces = mesh.faces_packed() 
        # (V, 3) where V is the total number of verts across all the meshes in the batch
        packed_verts = mesh.verts_packed() 

        vertex_visibility_map = torch.zeros(packed_verts.shape[0])   # (V,)
        allface = range(20908)
        allpoints = range(10475)
        # Indices of unique visible faces
        visible_faces = pix_to_face.unique()   # (num_visible_faces )
        # Get Indices of unique visible verts using the vertex indices in the faces
        visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
        unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )
        klvisible_faces = list(visible_faces.clone().cpu().numpy())
        klvisible_points = list(unique_visible_verts_idx.clone().cpu().numpy())
        unvisible_face = list(set(allface).difference(set(klvisible_faces)))
        unvisible_points = list(set(allpoints).difference(set(klvisible_points)))
        # print(len(unvisible_face))
        unvisible_face = torch.from_numpy(np.asarray(unvisible_face)).to(device)
        unvisible_points = torch.from_numpy(np.asarray(unvisible_points)).to(device)

        # Update visibility indicator to 1 for all visible vertices 
        vertex_visibility_map[unique_visible_verts_idx] = 1.0
        # packed_verts = packed_verts.cpu()
        

        visible_points_init = packed_verts[unique_visible_verts_idx]
        num_vis = visible_points_init.shape[0]
        visible_points_init = visible_points_init.reshape(1,num_vis,3)
        invisible_points_init = packed_verts[unvisible_points]
        num_invis = invisible_points_init.shape[0]
        invisible_points_init = invisible_points_init.reshape(1,num_invis,3)
        init_cd,_ = chamfer_distance(visible_points_init,invisible_points_init)
        
        # The optimizer and variables
        optimed_pose = torch.tensor(
            data["body_pose"], device=device, requires_grad=True
        )  # [1,23,3,3]
        optimed_trans = torch.tensor(
            data["trans"], device=device, requires_grad=True
        )  # [3]
        optimed_betas = torch.tensor(
            data["betas"], device=device, requires_grad=True
        )  # [1,10]
        optimed_orient = torch.tensor(
            data["global_orient"], device=device, requires_grad=True
        )  # [1,1,3,3]
        optimed_scale = torch.tensor(
            data["scale"], device=device, requires_grad=True
        ) 

        paa = transforms.matrix_to_axis_angle(optimed_pose)
        oaa = transforms.matrix_to_axis_angle(optimed_orient)
       
        paa = torch.tensor(
            paa, device=device, requires_grad=True
        )  # [1,1,3,3]
        oaa = torch.tensor(
            oaa, device=device, requires_grad=True
        )
        
        optimizer_smpl = torch.optim.SGD(
            [paa, oaa, optimed_trans, optimed_betas, optimed_scale],
            lr=2e-2,
            momentum=0.9,
        )



        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        losses = {
            "cloth": {"weight": 1e1, "value": 0.0}, 
            "p2f": {"weight": 1, "value": 0.0},            # Cloth: Normal_recon - Normal_pred
            "stiffness": {"weight": 1e5, "value": 0.0},         # Cloth: [RT]_v1 - [RT]_v2 (v1-edge-v2)
            "rigid": {"weight": 1e5, "value": 0.0},             # Cloth: det(R) = 1
            "edge": {"weight": 0, "value": 0.0},                # Cloth: edge length
            "nc": {"weight": 0, "value": 0.0},                  # Cloth: normal consistency
            "laplacian": {"weight": 1e2, "value": 0.0},         # Cloth: laplacian smoonth
            "normal": {"weight": 1e0, "value": 0.0},            # Body: Normal_pred - Normal_smpl
            "silhouette": {"weight": 1e1, "value": 0.0},        # Body: Silhouette_pred - Silhouette_smpl
        }

        # # smpl optimization

        loop_smpl = tqdm(
            range(args.loop_smpl if cfg.net.prior_type != "pifu" else 1))

        per_data_lst = []

        for i in loop_smpl:

            per_loop_lst = []

            optimizer_smpl.zero_grad()

            pose =  transforms.axis_angle_to_matrix(paa)
            orient =  transforms.axis_angle_to_matrix(oaa)


            if dataset_param["hps_type"] != "pixie":
                smpl_out = dataset.smpl_model(
                    betas=optimed_betas,
                    body_pose=pose,
                    global_orient=orient,
                    pose2rot=False,
                )

                smpl_verts = ((smpl_out.vertices) +
                              optimed_trans) * data["scale"]
            else:
                smpl_verts, _, joints = dataset.smpl_model(
                    shape_params=optimed_betas,
                    expression_params=tensor2variable(data["exp"], device),
                    body_pose=pose,
                    global_pose=orient,
                    jaw_pose=tensor2variable(data["jaw_pose"], device),
                    left_hand_pose=tensor2variable(
                        data["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(
                        data["right_hand_pose"], device),
                )

                smplxpa = {
                    'optimed_betas':optimed_betas.detach().cpu(),
                    'expression_params':tensor2variable(data["exp"], device).detach().cpu(),
                    'body_pose':pose.detach().cpu(),
                    'global_pose':orient.detach().cpu(),
                    'jaw_pose':tensor2variable(data["jaw_pose"], device).detach().cpu(),
                    'left_hand_pose':tensor2variable(
                        data["left_hand_pose"], device).detach().cpu(),
                    'right_hand_pose': tensor2variable(
                        data["right_hand_pose"], device).detach().cpu()
                }
                
            
            
            
            smpl_verts = (smpl_verts + optimed_trans) * optimed_scale
            # smpl_verts = torch_normalize_pc(smpl_verts,1)

            T_mask_F, T_mask_B = dataset.render_silo(
                smpl_verts *
                torch.tensor([1.0, -1.0, -1.0]
                             ).to(device), in_tensor["smpl_faces"]
            )
            siloloss = torch.abs(T_mask_F - in_tensor["mask"]).mean()
            mesh = VF2Mesh(smpl_verts, in_tensor["smpl_faces"]).cuda()
            packed_verts_train = mesh.verts_packed()
            visible_points_train = packed_verts_train[unique_visible_verts_idx.cuda()].reshape(1,num_vis,3)
            invisible_points_train = packed_verts_train[unvisible_points.cuda()].reshape(1,num_invis,3)
           
            vp,_ = chamfer_distance(visible_points_train,partial_pc)
            mesh = mesh.cuda()
            visible_faces = visible_faces.cuda()
            unvisible_face = unvisible_face.cuda()
            partial_pc = partial_pc.cuda()
            
           
            p2floss = 10*point_mesh_face_distance(mesh, partial_pc,visible_faces) + 3*vp - 0.2*point_mesh_face_distance(mesh, partial_pc,unvisible_face) + 0.1*torch.abs(joints[...,:2]-original_joints[...,:2]).mean() + 0.1*torch.abs(optimed_betas).mean() 
           
            pbar_desc = "SMPL REFINE --- "
            losses["normal"]["value"] = (p2floss).mean()
            pbar_desc += f"Total: {p2floss.item():.4f}"
            loop_smpl.set_description(pbar_desc)
            p2floss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(p2floss)
            
        IO().save_mesh(mesh,"smplestimation/example/"+in_tensor['name']+".obj")
        
    print("dadfasdfasdfasdf")