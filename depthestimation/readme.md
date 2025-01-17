**In this module, you will generate a partial point cloud from a RGB image.**

To train this module, please make sure you have already prepare the data, i.e., RGB images and corrsponding depth maps.   

In the paper, we selected [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation) to train the depth estimation module. However, more effective monocular depth estimation methods have been proposed, such as [EcoDepth](https://github.com/aradhye2002/ecodepth), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything). Actually, you are free to choose any depth estimation method. In this work, we didn't use normal map during the training, but you can also use the normal map to further improve the performance of depth estimation.

Since our contributions are not focusing on this module, please clone the repositories of these great works to train the depth estimation model.

If you choose to use [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation), please run:

`git clone https://github.com/SwinTransformer/MIM-Depth-Estimation.git`

If you choose to use [EcoDepth](https://github.com/aradhye2002/ecodepth), please run:

`git clone https://github.com/Aradhye2002/EcoDepth.git`

Then, you should follow their instructions to prepare the environment and train the model. 



**At this stage, you should finally obtain image and partial point cloud pairs, such as 1.png and 1.ply. An example folder containing a pair is provided in the SMPL rectification directory.**
