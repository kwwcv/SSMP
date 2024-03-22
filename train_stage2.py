# modified based on 'train_single_seq.py in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
import random
from shutil import copytree, copy
from model import MotionNet

from data.nuscenes_dataloader import DatasetNuscenes, DatasetNuscenesGS
from test import eval_motion_displacement
from utils.misc import random_seed, AverageMeter, check_folder, update_ema_variables
from utils.loss import compute_sloss, compute_uloss
from utils.data_aug import decode_augment, data_augment, mask_w_gs, unlabel_mix
from utils.utils import label_refine


out_seq_len = 20  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)
future_frame_skip = 0

pred_adj_frame_distance = True  # Whether to predict the relative offset between frames
use_weighted_loss = True  # Whether to set different weights for different grid cell categories for loss computation

parser = argparse.ArgumentParser()
# data
parser.add_argument('-d', '--data', default='/bev_nuScenes/train/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--GSdata_root', default="/bevGS_nuScenes/train", type=str, help='The path to the preprocessed ground removal BEV training data')
parser.add_argument('--val_data', default='/bev_nuScenes/val/', type=str, help='The path to the preprocessed sparse BEV validation data')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
# data split
parser.add_argument('--preset_semi', default='/preset/1%_split.npy', type=str, help='Whether to use preset data spliting, if not random split')
# model
parser.add_argument('--resume', default=None, type=str, help='path of pretrained teacher model')
# training
parser.add_argument('--niter', default=20000, type=int, help='Number of iterations')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')
parser.add_argument('--log_freq', default=1000, type=int, help='Iteration interval to log')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='./logs/stage2/', help='The path to the output log file and checkpoint')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--ratio', default=100, type=int, help='ratio for labeled data')
parser.add_argument('--alpha', default=0.999, type=float, help='Alpha factor for EMA')
# module
parser.add_argument('--if_lf', action='store_true', help='label refinement')
parser.add_argument('--if_bevmix', action='store_true')
# label refinement parameter
parser.add_argument('--lf_topk', default=5, type=int, help='topk')
parser.add_argument('--lf_dist_thres', default=10, type=int, help='dist_thres')
parser.add_argument('--lf_disp_thres', default=1, type=float, help='deta disp thres')
parser.add_argument('--lf_weighted_diff_thres', default=0.6, type=float, help='lf_weighted_diff_thres')

args = parser.parse_args()

print(args)
random_seed(args.seed)

def main():
    start_iter = 1
    # Whether to log the training information
    if args.log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        # if args.resume == '':
        model_save_path = check_folder(logger_root)
        model_save_path = check_folder(os.path.join(model_save_path, 'train_single_seq'))
        model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging Evaluation details
        eval_file_name = os.path.join(model_save_path, 'eval.txt')
        eval_saver = open(eval_file_name, 'w')
        
        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        # Copy the code files as logs
        copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
        copytree('data', os.path.join(model_save_path, 'data'))
        copytree('utils', os.path.join(model_save_path, 'utils'))
        python_files = [f for f in os.listdir('.') if f.endswith('.py')]
        for f in python_files:
            copy(f, model_save_path)

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])

    if args.preset_semi is not None:
        preset_dict = np.load(args.preset_semi, allow_pickle=True).item()
        labeled_ids = preset_dict['labeled']
        unlabeled_ids = preset_dict['unlabeled']
    else:
        scenes = np.load('data/split.npy', allow_pickle=True).item().get('train')
        ids = [i for i in range(0, len(scenes))]
        labeled_ids = random.sample(ids, int(len(scenes)//args.ratio))
        unlabeled_ids = list(set(ids).difference(set(labeled_ids)))

    labeled_trainset = DatasetNuscenes(dataset_root=args.data, split='train', future_frame_skip=future_frame_skip,
                                voxel_size=voxel_size, area_extents=area_extents, num_category=cell_category_num, scenes_ids=labeled_ids)
    
    l_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworker, drop_last=True)
    print("Labeled dataset size:", len(labeled_trainset))

    # unlabeled dataset
    unlabeled_trainset = DatasetNuscenesGS(dataset_root=args.data, GSdata_root=args.GSdata_root, split='train', future_frame_skip=future_frame_skip,
                                voxel_size=voxel_size, area_extents=area_extents, num_category=cell_category_num, scenes_ids=unlabeled_ids)
    

    u_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworker, drop_last=True,
                                                collate_fn=unlabeled_trainset.collate_fn)
    print("unLabeled dataset size:", len(unlabeled_trainset))

    # test dataset 
    testset = DatasetNuscenes(dataset_root=args.val_data, split='test', future_frame_skip=future_frame_skip,
                            voxel_size=voxel_size, area_extents=area_extents, num_category=5)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.nworker)

    # Teacher model
    Tmodel = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
    Tmodel = nn.DataParallel(Tmodel)
    Tmodel = Tmodel.to(device)
    # Student model
    Smodel = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
    Smodel = nn.DataParallel(Smodel)
    Smodel = Smodel.to(device)
    # Load pretrained weights
    checkpoint = torch.load(args.resume)
    Tmodel.load_state_dict(checkpoint)
    Smodel.load_state_dict(checkpoint)
    print("Load model from {}".format(args.resume))

    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='sum')
    optimizer = optim.Adam(Smodel.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000, 16000, 24000], gamma=0.5)

    # Iteration
    l_batch_iteration = iter(l_trainloader)
    u_batch_iteration = iter(u_trainloader)

    running_s_loss_disp = AverageMeter('S_Disp', ':.4f')  # for motion prediction error
    running_s_loss_class = AverageMeter('S_Obj_Cls', ':.4f')  # for cell classification error
    running_s_loss_motion = AverageMeter('S_Motion_Cls', ':.4f')  # for state estimation error
    running_mix_loss_disp = AverageMeter('Mix_Disp', ':.4f')  # for motion prediction error

    # Train student model, inference teacher model
    Smodel.train()
    Tmodel.eval()
    for iteration in range(start_iter, args.niter+1):
        if iteration % args.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print("Iter {}, learning rate {}".format(iteration, lr))
            if args.log:
                saver.write("Iter: {}, lr: {}\t".format(iteration, lr))
                saver.flush()

        optimizer.zero_grad()
        ########### labeled supervision ###########
        try:
            l_padded_voxel_points, l_disp_gt, l_valid_pixel_maps, l_non_empty_map, l_class_gt, \
                past_steps, future_steps, l_motion_gt  = next(l_batch_iteration)
        except StopIteration:
            l_batch_iteration = iter(l_trainloader)
            l_padded_voxel_points, l_disp_gt, l_valid_pixel_maps, l_non_empty_map, l_class_gt, \
                past_steps, future_steps, l_motion_gt = next(l_batch_iteration)
        # Move to GPU/CPU
        l_padded_voxel_points = l_padded_voxel_points.to(device)

        ########### unlabeled supervision #############
        try:
            u_padded_voxel_points, _, _, u_non_empty_map, _, \
                past_steps, future_steps, _, u_source_bevs, u_target_bev = next(u_batch_iteration)
        except StopIteration:
            u_batch_iteration = iter(u_trainloader)
            u_padded_voxel_points, _, _, u_non_empty_map, _, \
                past_steps, future_steps, _, u_source_bevs, u_target_bev = next(u_batch_iteration)
        # Move to GPU/CPU
        u_padded_voxel_points = u_padded_voxel_points.to(device)
        u_source_bev = u_source_bevs[:,-1,...].copy()
        # generate pseudo label
        # data augmentation
        with torch.no_grad():
            weak_u_padded_voxel_points, aug_list = data_augment(u_padded_voxel_points.clone()) # weak augmentation
            u_disp_gt, u_class_gt, u_motion_gt = Tmodel(weak_u_padded_voxel_points) # generate pseudo label
            u_disp_gt = u_disp_gt.view(args.batch_size, out_seq_len, 2, 256, 256)
            u_disp_gt, u_class_gt, u_motion_gt = decode_augment(u_disp_gt, u_class_gt, u_motion_gt, aug_list)

        if pred_adj_frame_distance:
            for c in range(1, u_disp_gt.size(1)):
                u_disp_gt[:, c, ...] = u_disp_gt[:, c, ...] + u_disp_gt[:, c - 1, ...]

        u_class_gt = u_class_gt.permute(0, 2, 3, 1)
        u_motion_gt = u_motion_gt.permute(0, 2, 3, 1)
        u_disp_gt, u_class_gt, u_motion_gt = mask_w_gs(u_disp_gt, u_class_gt, u_motion_gt, u_source_bev) # mask with ground segmentation
        if args.if_lf:
            u_disp_gt, u_non_empty_map = label_refine(device, u_source_bev, u_target_bev, u_non_empty_map, u_disp_gt, 
                                                      topk=args.lf_topk, dist_thres=args.lf_dist_thres, args=args) # refine pseudo lablels

        u_disp_gt = u_disp_gt.permute(0,1,3,4,2) # (B,out_seq,w,h,2)
        u_padded_voxel_points, u_disp_gt, u_class_gt, u_motion_gt, u_non_empty_map, u_source_bevs, _ \
            = data_augment(u_padded_voxel_points, u_disp_gt, u_class_gt, u_motion_gt, u_non_empty_map, source_bevs=u_source_bevs) # data augmentation

        l_padded_voxel_points, l_disp_gt, l_class_gt, l_motion_gt, l_non_empty_map, l_valid_pixel_maps, _ \
            = data_augment(l_padded_voxel_points, l_disp_gt, l_class_gt, l_motion_gt, l_non_empty_map, l_valid_pixel_maps) # data augmentation 

        if args.if_bevmix:
            mix_input, mix_disp_gt, mix_class_gt, mix_motion_gt, mix_non_empty_map = \
                unlabel_mix(u_padded_voxel_points, u_disp_gt, u_class_gt, u_motion_gt, u_source_bevs, u_non_empty_map) # Inplace change U data
        else:
            mix_input, mix_disp_gt, mix_class_gt, mix_motion_gt, mix_non_empty_map = u_padded_voxel_points, u_disp_gt, u_class_gt, u_motion_gt, u_non_empty_map


        # pack labeled and unlabeled data
        padded_voxel_points = torch.cat([l_padded_voxel_points, mix_input], dim=0) # input
        # Make prediction
        disp_pred, class_pred, motion_pred = Smodel(padded_voxel_points)
        # Unpack
        l_mask = torch.zeros(args.batch_size*2, device=device, dtype=torch.bool)
        l_mask[:args.batch_size] = True
        u_mask = torch.zeros(args.batch_size*2, device=device, dtype=torch.bool)
        u_mask[args.batch_size:2*args.batch_size] = True
        # Compute and back-propagate the losses
        s_loss, s_loss_disp, s_loss_class, s_loss_motion = \
            compute_sloss(device, out_seq_len, l_disp_gt, l_valid_pixel_maps,
                                l_class_gt, disp_pred, criterion, l_non_empty_map, class_pred, 
                                l_motion_gt, motion_pred, pred_adj_frame_distance, use_weighted_loss, l_mask) # supervised loss
        
        u_loss, u_loss_disp = \
            compute_uloss(device, out_seq_len, mix_disp_gt,
                                mix_class_gt, disp_pred, mix_non_empty_map, class_pred, 
                                mix_motion_gt, motion_pred, pred_adj_frame_distance, use_weighted_loss, u_mask) # consistency loss
        
        loss = s_loss + u_loss
        loss.backward() # backward proporgations
        optimizer.step()
        scheduler.step()
        update_ema_variables(Smodel, Tmodel, alpha=args.alpha) # EMA

        running_s_loss_disp.update(s_loss_disp)
        running_s_loss_class.update(s_loss_class)
        running_s_loss_motion.update(s_loss_motion)
        running_mix_loss_disp.update(u_loss_disp)

        print("iter {}\n {}, \t{}, \t{}, \t{}, \t".
            format(iteration, running_s_loss_disp, running_s_loss_class, running_s_loss_motion, running_mix_loss_disp))

        if iteration % args.log_freq == 0:

            if args.log:
                saver.write("{}\t{}\t{}\t{}\n".format(running_s_loss_disp, running_s_loss_class, running_s_loss_motion, running_mix_loss_disp))
                saver.flush()
                torch.save(Tmodel.state_dict(), os.path.join(model_save_path, 'iter_' + str(iteration) + '.pth'))
                
                eval_saver.write(f'iter:{iteration}\n')
                eval_saver.flush()
                eval_motion_displacement(Tmodel, eval_saver, use_adj_frame_pred=True, dataloader=testloader, future_frame_skip=future_frame_skip,
                                sample_nums=len(testset), use_motion_state_pred_masking=True)
    if args.log:
        saver.close()


if __name__ == "__main__":
    main()
