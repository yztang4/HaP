This module is build based on [PDR](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement). Please follow [PDR](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement) to prepare the environment. 


To train the diffusion model, you should prepare a ".h5" file contains "smpl_points", "all_points" (ground truth point cloud) and "depth_points" keys. To train the refinement model, you should prepare a ".h5" file contains "coarse_pc", "gt_pc" (ground truth point cloud) and "depth_points" keys. 

train diffusion model command line:

`python distributedsmpl.py --config exp_configs/mvp_configs/smpl_thuman_condition.json`

train refinement model command line:

`python distributedsmpl.py --config exp_configs/mvp_configs/refine_pdr_generation.json`

generate coarse point cloud with the diffusion model:

`python generate_samples_distributed_smpl.py --execute --gather_results --remove_original_files --config exp_configs/mvp_configs/smpl_thuman_condition.json --ckpt_name pointnet_ckpt_731249.pkl --batch_size 4 --phase test --device_ids '0'`

generate refined point cloud with the refinement model:

`CUDA_VISIBLE_DEVICES=2 python generate_samples_smpl_withnormal.py --config exp_configs/mvp_configs/refine_pdr_generation.json --ckpt_name pointnet_ckpt_299999.pkl --batch_size 8 --phase test --device_ids '0'`

The generated files will be saved in "pointnet2/mvp_dataloader/data"