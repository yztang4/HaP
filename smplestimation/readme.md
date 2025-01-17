**In this module, you will obtain an SMPL model well-algined with the partial point cloud.**

This module is build based on [ICON](https://github.com/YuliangXiu/ICON). Please run:

`git clone https://github.com/YuliangXiu/ICON`

And then follow the instructions of ICON to prepare the environment and the necessary data. 
Then run:

`mv refinesmplhap.py ICON/apps` 

`mv newp2f.py ICON/apps` 

`mv RefineSMPLDataset.py ICON/lib/dataset` 

We provide an example folder which contains an image and a partial point cloud, you can run:

`python -m apps.refinesmplhap`

The refined SMPL model will be saved as "1.obj" in the example folder.

We also provide a script "preparediffusiondata.py" to prepare the test data for the next module. Remember, you need to change the "all_points" to the ground truth points to prepare the training data.
