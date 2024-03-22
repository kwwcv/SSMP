# modified based on 'nuscenes_dataloader.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from data.data_utils import classify_speed_level
from utils.utils import voxel2bev
    
class DatasetNuscenes(Dataset):
    def __init__(self, dataset_root=None, split='train', future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-3., 2.]]), num_category=5, scenes_ids=None):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        Input:
        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data

        Return:
        padded_voxel_points (seq_len, w, h, d): input BEV sequence
        all_disp_field_gt (future_len, w, h, 2): motion ground truth
        all_valid_pixel_maps (future_len, w, h)
        non_empty_map (w, h): 1 for occupied, 0 for empty
        pixel_cat_map (w, h, num_category): category ground truth
        motion_state_gt (w, h, 2): motion_state ground truth, [1,0] for static, [0,1] for moving 
        """
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        if (split == 'train') and scenes_ids is not None:
            scenes = np.load('data/split.npy', allow_pickle=True).item().get('train')
            scenes = scenes[scenes_ids]
            scenes_ids = [sce.split('_')[-1] for sce in scenes]
            seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                if (os.path.isdir(os.path.join(self.dataset_root, d))) and (d.split('_')[0] in scenes_ids)]
        else:
            seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                        if os.path.isdir(os.path.join(self.dataset_root, d))]
            
        seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
            if (os.path.isfile(os.path.join(seq_dir, f))) and ('0' in f)]
        self.seq_files = seq_files

        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.future_frame_skip = future_frame_skip

    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):
        seq_file = self.seq_files[idx]
        gt_data_handle = np.load(seq_file, allow_pickle=True)
        gt_dict = gt_data_handle.item()


        dims = gt_dict['3d_dimension']
        num_future_pcs = gt_dict['num_future_pcs']
        num_past_pcs = gt_dict['num_past_pcs']
        pixel_indices = gt_dict['pixel_indices']

        sparse_disp_field_gt = gt_dict['disp_field']
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

        sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
        all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
        all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

        sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
        pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
        pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

        padded_voxel_points = list()  
        for i in range(num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            indices = indices[(indices[:,0]<dims[0]) & (indices[:,1]<dims[1]) & (indices[:,2]<dims[2])]
            curr_voxels = np.zeros(dims, dtype=np.bool_)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        # Classify speed-level (ie, state estimation: static or moving)
        motion_state_gt = classify_speed_level(all_disp_field_gt, total_future_sweeps=num_future_pcs,
                                              future_frame_skip=self.future_frame_skip)

        return padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
            non_empty_map, pixel_cat_map, num_past_pcs, num_future_pcs, motion_state_gt
    
class DatasetNuscenesGS(Dataset):
    def __init__(self, dataset_root=None, GSdata_root=None, split='train', future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-3., 2.]]), num_category=5, scenes_ids=None):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        num_category: The number of object categories (including the background)

        Return:
        padded_voxel_points (seq_len, w, h, d): input BEV sequence
        all_disp_field_gt (future_len, w, h, 2): motion ground truth
        all_valid_pixel_maps (future_len, w, h)
        non_empty_map (w, h): 1 for occupied, 0 for empty
        pixel_cat_map (w, h, num_category): category ground truth
        motion_state_gt (w, h, 2): motion_state ground truth, [1,0] for static, [0,1] for moving 
        source_bevs: List of non-empty bev cells' 2D corrdinates (N_t, 2)
        target_bev: non-empty bev cells' 2D corrdinates (N_t, 2)
        """
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        if (split == 'train') and scenes_ids is not None:
            scenes = np.load('data/split.npy', allow_pickle=True).item().get('train')
            scenes = scenes[scenes_ids]
            scenes_ids = [sce.split('_')[-1] for sce in scenes]
            seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                if (os.path.isdir(os.path.join(self.dataset_root, d))) and (d.split('_')[0] in scenes_ids)]
        else:
            seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                        if os.path.isdir(os.path.join(self.dataset_root, d))]
        seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
                    if (os.path.isfile(os.path.join(seq_dir, f))) and ('0' in f)]
        self.seq_files = seq_files

        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.future_frame_skip = future_frame_skip
        self.voxel_path = GSdata_root

    def __len__(self):
        return self.num_sample_seqs
    
    @staticmethod
    def collate_fn(batch):
        padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map, \
        num_past_pcs, num_future_pcs, motion_state_gt, source_bevs, target_bev = zip(*batch)
        # 
        max_len = 3000
        pad_coord = np.zeros([1,2])

        source_bev_list = []
        target_bev_list = []
        for source_bev in source_bevs:
            t_bev_list = []
            for bev in source_bev:
                if len(bev) < max_len:
                    pad_len = max_len - len(bev)
                    t_bev_list.append(np.vstack([bev, np.repeat(pad_coord, pad_len, axis=0)]))
                elif len(bev) == max_len:
                    t_bev_list.append(bev)
                else:
                    sampled_index = sorted(random.sample(range(len(bev)), max_len))
                    t_bev_list.append(bev[sampled_index])
            source_bev_list.append(np.stack(t_bev_list, axis=0))

        for bev in target_bev:
            if len(bev) < max_len:
                pad_len = max_len - len(bev)
                target_bev_list.append(np.vstack([bev, np.repeat(pad_coord, pad_len, axis=0)]))
            elif len(bev)== max_len:
                target_bev_list.append(bev)
            else:
                sampled_index = sorted(random.sample(range(len(bev)), max_len))
                target_bev_list.append(bev[sampled_index])

        padded_source_bev = np.stack(source_bev_list, axis=0)
        padded_target_bev = np.stack(target_bev_list, axis=0)

        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points, axis=0))
        all_disp_field_gt = torch.from_numpy(np.stack(all_disp_field_gt, axis=0))
        all_valid_pixel_maps = torch.from_numpy(np.stack(all_valid_pixel_maps, axis=0))
        non_empty_map = torch.from_numpy(np.stack(non_empty_map, axis=0))
        pixel_cat_map = torch.from_numpy(np.stack(pixel_cat_map, axis=0))
        motion_state_gt = torch.from_numpy(np.stack(motion_state_gt, axis=0))

        return padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map, \
        num_past_pcs, num_future_pcs, motion_state_gt, padded_source_bev, padded_target_bev

    def __getitem__(self, idx):
        seq_file = self.seq_files[idx]
        gt_data_handle = np.load(seq_file, allow_pickle=True)
        gt_dict = gt_data_handle.item()

        dims = gt_dict['3d_dimension']
        num_future_pcs = gt_dict['num_future_pcs']
        num_past_pcs = gt_dict['num_past_pcs']
        pixel_indices = gt_dict['pixel_indices']

        sparse_disp_field_gt = gt_dict['disp_field']
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

        sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
        all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
        all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

        sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
        pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
        pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

        padded_voxel_points = list()
        for i in range(num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            indices = indices[(indices[:,0]<dims[0]) & (indices[:,1]<dims[1]) & (indices[:,2]<dims[2])]
            curr_voxels = np.zeros(dims, dtype=np.bool_)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        # Classify speed-level (ie, state estimation: static or moving)
        motion_state_gt = classify_speed_level(all_disp_field_gt, total_future_sweeps=num_future_pcs,
                                              future_frame_skip=self.future_frame_skip)
        
        source_bevs = []
        # load coordinate voxels 
        voxels = np.load(os.path.join(self.voxel_path, seq_file.split('/')[-2]+'.npy'), allow_pickle=True).item()
        for i in range(num_past_pcs):
            source_bevs.append(voxel2bev(voxels[f'ng_voxel_indices_{i}']))
        target_bev = voxel2bev(voxels['ng_voxel_indices_9'])
        return padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
            non_empty_map, pixel_cat_map, num_past_pcs, num_future_pcs, motion_state_gt, source_bevs, target_bev




