# Human as Points: Explicit Point-based 3D Human Reconstruction from Single-view RGB Images
Tang Yingzhi, Zhang Qijian, Hou Junhui, Liu Yebin

The official pytorch implementation of HaP.

## CODE
HaP is a single-view human reconstruction framework, it has four modules:
1. Depth Estimation
2. SMPL Estimation and Rectification
3. Point Cloud Generation
4. Depth Replacement

**To train and test HaP, you need to prepare the training data first. Please refer to [2K2K](https://github.com/SangHunHan92/2K2K) or [IntegratedPIFu](https://github.com/kcyt/IntegratedPIFu)**
Both [2K2K](https://github.com/SangHunHan92/2K2K) and [IntegratedPIFu](https://github.com/kcyt/IntegratedPIFu) provide detailed rendering script to prepare the RGB images and depth maps. Additional, [IntegratedPIFu](https://github.com/kcyt/IntegratedPIFu) also provides the blender project to prepare the normal map. 

For each module, I provide a folder to run the code. 

In the **generatemesh** folder, we provide an "in-the-wild" image example, you can run the script to see example generated meshes. 

We sincerely thank the authors of [ICON](https://github.com/YuliangXiu/ICON), [2K2K](https://github.com/SangHunHan92/2K2K),  [IntegratedPIFu](https://github.com/kcyt/IntegratedPIFu), [PDR](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement), [EcoDepth](https://github.com/aradhye2002/ecodepth) and [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation) for their excellent work and the released code. Please consider citing their papers.

## CityUHuman Dataset 
**Agreement**

The CityuHuman dataset (the "Dataset") is available for non-commercial research purposes only. Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, as training data for a commercial product, for commercial ergonomic analysis (e.g. product design, architectural design, etc.), or production of other artifacts for commercial purposes including, for example, web services, movies, television programs, mobile applications, or video games. The Dataset may not be reproduced, modified and/or made available in any form to any third party without CityU’s prior written permission.

You agree not to reproduce, modified, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of derived data in any form to any third party without CityU's prior written permission.

You agree not to further copy, publish or distribute any portion of the Dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

City University of Hong Kong reserves the right to terminate your access to the Dataset at any time.

**Download Link**

Send an e-mail to TANG (yztang4-c@my.cityu.edu.hk) and CC Prof. Hou (jh.hou@cityu.edu.hk) for the download link.

NOTE: For privacy protection, please blur the faces if the images or the models appear in any materials that will be published (such as paper, video, poster, etc.) 

Please consider citing our paper.
