import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random
import copy
import sys
import warnings
import pickle


sys.path.insert(0, os.path.dirname(__file__))
from mvp_data_utils import augment_cloud

class SMPLTHUMAN(data.Dataset):
    def __init__(self, data_dir, train=True, npoints=2048, novel_input=True, novel_input_only=False,
                        scale=1, rank=0, world_size=1, random_subsample=False, num_samples=1000,
                        augmentation=False, return_augmentation_params=False,
                        include_generated_samples=False, generated_sample_path=None,
                        randomly_select_generated_samples=False, # randomly select a trial from multi trial generations
                        use_mirrored_partial_input=False, number_partial_points=2048,
                        load_pre_computed_XT=False, T_step=100, XT_folder=None,
                        append_samples_to_last_rank=True):
        self.return_augmentation_params = return_augmentation_params
        
            
        if train:
            # self.input_path = '%s/thuman_8ktrain.h5' % data_dir
            self.input_path = '%sxxxxx.h5' % data_dir
        else:
            
            self.input_path = '%sxxxx.h5' % data_dir
            
        self.npoints = npoints
        self.train = train # controls the trainset and testset
        # self.benchmark = benchmark
        self.augmentation = augmentation # augmentation could be a dict or False

        input_file = h5py.File(self.input_path, 'r')
        if train:
            
            self.input_data = np.array((input_file['depth_points'][()]))
            self.gt_data  = np.array((input_file['all_points'][()]))
            self.smpl = np.array((input_file['smpl_points'][()]))
            
        else:
            self.input_data = np.array((input_file['depth_points'][()]), dtype=np.float32)
            self.gt_data  = np.array((input_file['all_points'][()]), dtype=np.float32)
            self.smpl = np.array((input_file['smpl_points'][()]), dtype=np.float32)

        B, N = self.gt_data.shape[:2]
        self.labels = np.zeros((B,1))
        

        input_file.close()

        # load XT generated from a trained DDPM
        self.load_pre_computed_XT = load_pre_computed_XT
        if load_pre_computed_XT:
            if train:
                XT_folder = os.path.join(XT_folder, 'train')
            else:
                XT_folder = os.path.join(XT_folder, 'test')
            self.T_step = T_step
            XT_file = os.path.join(XT_folder, 'mvp_generated_data_2048pts_T%d.h5' % T_step)
            self.XT_file = XT_file
            generated_XT_file = h5py.File(XT_file, 'r')
            self.generated_XT = np.array(generated_XT_file['data'])
            generated_XT_file.close()

        # load X0 generated from a trained DDPM
        self.include_generated_samples = include_generated_samples
        self.generated_sample_path = generated_sample_path
        self.randomly_select_generated_samples = randomly_select_generated_samples
        if include_generated_samples:
            # generated_samples/T1000_betaT0.02_shape_completion_no_class_condition_scale_1_no_random_replace_partail_with_complete/ckpt_1403999/
            generated_samples_file = os.path.join(data_dir, generated_sample_path)
            if randomly_select_generated_samples:
                files = os.listdir(generated_samples_file)
                files = [f for f in files if f.startswith('trial')]
                files = [os.path.join(generated_samples_file, f) for f in files]
                files = [generated_samples_file] + files
                generated_samples_file = random.choice(files)
                print('Randomly select file %s for generated samples from %d files' % (generated_samples_file, len(files)))

            # if benchmark:
            #     generated_samples_file = os.path.join(generated_samples_file, 'benchmark')
            # else:
            if train:
                generated_samples_file = os.path.join(generated_samples_file, 'train')
            else:
                generated_samples_file = os.path.join(generated_samples_file, 'test')
            generated_samples_file = os.path.join(generated_samples_file, 'mvp_generated_data_2048pts.h5')

            generated_file = h5py.File(generated_samples_file, 'r')
            self.generated_sample = np.array(generated_file['data'])
            generated_file.close()

        
        self.scale = scale
        # shapes in mvp dataset range from -0.5 to 0.5
        # we rescale the, to make the, range from -scale to scale 
        
        self.input_data = self.input_data * scale
        # if not benchmark:
        self.gt_data = self.gt_data  * scale
        if self.include_generated_samples:
            self.generated_sample = self.generated_sample  * scale
        if self.load_pre_computed_XT:
            self.generated_XT = self.generated_XT * scale

        print('partial point clouds:', self.input_data.shape)
        # if not benchmark:
        print('gt complete point clouds:', self.gt_data.shape)
        if self.include_generated_samples:
            print('DDPM generated complete point clouds:', self.generated_sample.shape)
        if self.load_pre_computed_XT:
            print('DDPM generated intermediate complete point clouds:', self.generated_XT.shape)
        self.labels = self.labels.astype(int)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # deepcopy is necessary here, because we may alter partial and complete for data augmentation
        # it will change the original data if we do not deep copy
        result = {}

        result['partial'] = copy.deepcopy(self.input_data[index])
        # if not self.benchmark:
        result['complete'] = copy.deepcopy(self.gt_data[index])
        result['smpl'] = copy.deepcopy(self.smpl[index])

        if self.include_generated_samples:
            result['generated'] = copy.deepcopy(self.generated_sample[index])
        if self.load_pre_computed_XT:
            result['XT'] = copy.deepcopy(self.generated_XT[index])

        # augment the point clouds
        if isinstance(self.augmentation, dict):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(result_list, self.augmentation,
                                                                return_augmentation_params=True)
            else:
                result_list = augment_cloud(result_list, self.augmentation, return_augmentation_params=False)
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]
            if self.include_generated_samples:
                # add noise to every point in the point cloud generated by a trained DDPM
                # this is used to train the refinement network
                sigma = self.augmentation.get('noise_magnitude_for_generated_samples', 0)
                if sigma > 0:
                    noise = np.random.normal(scale=sigma, size=result['generated'].shape)
                    noise = noise.astype(result['generated'].dtype)
                    result['generated'] = result['generated'] + noise

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        result['label'] = self.labels[index]

        return result

if __name__ == '__main__':

    import pdb
    aug_args = {'pc_augm_scale':1.5, 'pc_augm_rot':True, 'pc_rot_scale':30.0, 'pc_augm_mirror_prob':0.5, 'pc_augm_jitter':False, 'translation_magnitude': 0.1}
    aug_args =  False
    include_generated_samples=False
    # benchmark = False
    generated_sample_path='generated_samples/T1000_betaT0.02_shape_completion_mirror_rot_60_scale_1.2_translation_0.05/ckpt_623999'
    dataset = SMPLTHUMAN('/public/sdi/tangyingzhi/Project', train=False, npoints=2048, novel_input=True, novel_input_only=False,
                            augmentation=aug_args, scale=1,
                            random_subsample=True, num_samples=1000,
                            include_generated_samples=include_generated_samples, 
                            generated_sample_path=generated_sample_path,
                            use_mirrored_partial_input=True, number_partial_points=3072,
                            rank=0, world_size=1,
                            load_pre_computed_XT=False, T_step=10, 
                            XT_folder='data/mvp_dataset/generated_samples/T1000_betaT0.02_shape_completion_avg_max_pooling_mirror_rot_90_scale_1.2_translation_0.2/ckpt_545999/',
                            append_samples_to_last_rank=False,
                            return_augmentation_params=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    for i, data in enumerate(dataloader):
        # label, partial, complete = data
        label, partial, complete = data['label'], data['partial'], data['complete']

        print('index %d partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f]' % (
            i, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),))
        