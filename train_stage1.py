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
from data.nuscenes_dataloader import DatasetNuscenes
from test import eval_motion_displacement
from utils.misc import random_seed, AverageMeter, check_folder
from utils.loss import compute_sloss

out_seq_len = 20  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)
future_frame_skip = 0 # How many future frames to skip

pred_adj_frame_distance = True  # Whether to predict the relative offset between frames
use_weighted_loss = True  # Whether to set different weights for different grid cell categories for loss computation

parser = argparse.ArgumentParser()
# data
parser.add_argument('-d', '--data', default='/bev_nuScenes/train/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--val_data', default='/bev_nuScenes/val/', type=str, help='The path to the preprocessed sparse BEV validation data')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
# data split
parser.add_argument('--preset_semi', default='/preset/1%_split.npy', type=str, help='Whether to use preset data spliting, if not random split')
# training
parser.add_argument('--niter', default=20000, type=int, help='Number of iterations')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')
parser.add_argument('--lr', default=0.002, type=float, help='Learning Rate')
parser.add_argument('--ratio', default=100, type=int, help='ratio for labeled data')
parser.add_argument('--log_freq', default=1000, type=int, help='Iteration interval to log')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='./logs/stage1/', help='The path to the output log file and checkpoint')
parser.add_argument('--seed', default=1, type=int, help='random seed')

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
    else:
        scenes = np.load('data/split.npy', allow_pickle=True).item().get('train')
        ids = [i for i in range(0, len(scenes))]
        labeled_ids = random.sample(ids, int(len(scenes)//args.ratio))
    # labeled dataset
    labeled_trainset = DatasetNuscenes(dataset_root=args.data, split='train', future_frame_skip=future_frame_skip,
                                voxel_size=voxel_size, area_extents=area_extents, num_category=cell_category_num, scenes_ids=labeled_ids)
    
    l_trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworker, drop_last=True)
    print("Labeled dataset size:", len(labeled_trainset))

    # test dataset 
    testset = DatasetNuscenes(dataset_root=args.val_data, split='test', future_frame_skip=future_frame_skip,
                            voxel_size=voxel_size, area_extents=area_extents, num_category=5)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.nworker)

    model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000, 16000], gamma=0.5)

    l_batch_iteration = iter(l_trainloader)
    running_loss_disp = AverageMeter('Disp', ':.6f')  # for motion prediction error
    running_loss_class = AverageMeter('Obj_Cls', ':.6f')  # for cell classification error
    running_loss_motion = AverageMeter('Motion_Cls', ':.6f')  # for state estimation error

    model.train()
    for iteration in range(start_iter, args.niter+1):
        if iteration % args.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print("Iter {}, learning rate {}".format(iteration, lr))
            if args.log:
                saver.write("Iter: {}, lr: {}\t".format(iteration, lr))
                saver.flush()

        optimizer.zero_grad()
        # scheduler.step()
        ########### labeled supervision ###########
        try:
            l_padded_voxel_points, l_disp_gt, l_valid_pixel_maps, l_non_empty_map, l_class_gt, \
                past_steps, future_steps, l_motion_gt = next(l_batch_iteration)
        except StopIteration:
            l_batch_iteration = iter(l_trainloader)
            l_padded_voxel_points, l_disp_gt, l_valid_pixel_maps, l_non_empty_map, l_class_gt, \
                past_steps, future_steps, l_motion_gt = next(l_batch_iteration)
        # Move to GPU/CPU
        l_padded_voxel_points = l_padded_voxel_points.to(device)
        # ---------------------------------------------------------------------
        # -- Compute the masked displacement loss
        # Make prediction
        l_disp_pred, l_class_pred, l_motion_pred = model(l_padded_voxel_points)

        # Compute and back-propagate the losses
        s_loss, s_loss_disp, s_loss_class, s_loss_motion = \
            compute_sloss(device, out_seq_len, l_disp_gt, l_valid_pixel_maps,
                                l_class_gt, l_disp_pred, criterion, l_non_empty_map, l_class_pred, 
                                l_motion_gt, l_motion_pred, pred_adj_frame_distance, use_weighted_loss)
        
        s_loss.backward() # backward proporgations
        optimizer.step()
        scheduler.step()

        running_loss_disp.update(s_loss_disp)
        running_loss_class.update(s_loss_class)
        running_loss_motion.update(s_loss_motion)
        
        if iteration % args.log_freq == 0:

            if args.log:
                saver.write("{}\t{}\t{}\n".format(running_loss_disp, running_loss_class, running_loss_motion))
                saver.flush()

                torch.save(model.state_dict(), os.path.join(model_save_path, 'iter_' + str(iteration) + '.pth'))
                
                eval_saver.write(f'iter:{iteration}\n')
                eval_saver.flush()
                eval_motion_displacement(model, eval_saver, use_adj_frame_pred=pred_adj_frame_distance, dataloader=testloader, 
                                         future_frame_skip=future_frame_skip, sample_nums=len(testset), use_motion_state_pred_masking=True)
                model.train()
                
        print("{}, \t{}, \t{}, \tat iter {}".
            format(running_loss_disp, running_loss_class, running_loss_motion, iteration))

    if args.log:
        saver.close()



if __name__ == "__main__":
    main()
