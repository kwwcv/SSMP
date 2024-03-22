# modified based on 'test.py' in MotionNet(https://www.merl.com/research/?research=license-request&sw=MotionNet)
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from model import MotionNet
from sklearn.metrics import confusion_matrix
from data.nuscenes_dataloader import DatasetNuscenes


color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}


def eval_motion_displacement(model, saver, use_adj_frame_pred=False,
                             dataloader=None, sample_nums=None, future_frame_skip=0,
                             num_future_sweeps=20, use_motion_state_pred_masking=False):
    """
    Evaluate the motion prediction results.

    model_path: The path to the trained model
    saver: the log file
    use_adj_frame_pred: Whether to predict the relative offset between frames
    future_frame_skip: How many future frames need to be skipped within a contiguous sequence (ie, [1, 2, ... 20])
    split: [val/test]
    num_future_sweeps: The number of future frames
    use_motion_state_pred_masking: Whether to threshold the displacement predictions with predicted state estimation
    """

    device = 'cuda'
    model.eval()

    # The speed intervals for grouping the cells
    # speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])  # unit: m/s
    # We do not consider > 20m/s, since objects in nuScenes appear inside city and rarely exhibit very high speed
    speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, 3 + 1)  # We evaluate predictions at [0.2, 0.4, ..., 1]s
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)  # "20" is because the LIDAR scanner is 20Hz

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)

    pixel_acc = 0  # for computing mean pixel classification accuracy
    overall_cls_pred = list()  # to compute classification accuracy for each object category
    overall_cls_gt = list()  # to compute classification accuracy for each object category

    for i, data in enumerate(dataloader, 0):
        print(f'Testing: {i}/{len(dataloader)}',end='\r')
        padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
            non_empty_map, pixel_cat_map_gt, past_steps, future_steps, motion_gt = data

        padded_voxel_points = padded_voxel_points.to(device)

        with torch.no_grad():
            disp_pred, class_pred, motion_pred = model(padded_voxel_points)


            pred_shape = disp_pred.size()
            disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
            disp_pred = disp_pred.contiguous()
            disp_pred = disp_pred.cpu().numpy()

            if use_adj_frame_pred:
                for c in range(1, disp_pred.shape[1]):
                    disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]

            if use_motion_state_pred_masking:
                motion_pred_numpy = motion_pred.cpu().numpy()
                motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)
                mask = motion_pred_numpy == 0

                class_pred_numpy = class_pred.cpu().numpy()
                class_pred_cat = np.argmax(class_pred_numpy, axis=1)
                class_mask = class_pred_cat == 0  # background mask

                # For those with very small movements, we consider them as static
                last_pred = disp_pred[:, -1, :, :, :]
                last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)  # out: (batch, h, w)
                thd_mask = last_pred_norm <= 0.2

                cat_weight_map = np.ones_like(class_pred_cat, dtype=np.float32)
                cat_weight_map[mask] = 0.0
                cat_weight_map[class_mask] = 0.0
                cat_weight_map[thd_mask] = 0.0
                cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)

                disp_pred = disp_pred * cat_weight_map

        # Pre-processing
        all_disp_field_gt = all_disp_field_gt.numpy()  # (bs, seq, h, w, channel)
        future_steps = future_steps.numpy()[0]

        valid_pixel_maps = all_valid_pixel_maps[:, -future_steps:, ...].contiguous()
        valid_pixel_maps = valid_pixel_maps.numpy()

        all_disp_field_gt = all_disp_field_gt[:, -future_steps:, ]
        all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
        all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

        # -----------------------------------------------------------------------------------
        # Compute the evaluation metrics
        # First, compute the displacement prediction error;
        # Compute the static and moving cell masks, and
        # Iterate through the distance intervals and group the cells based on their speeds;
        upper_thresh = 0.2
        upper_bound = (future_frame_skip + 1) / 20 * upper_thresh

        static_cell_mask = all_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)  # along the temporal axis
        moving_cell_mask = np.logical_not(static_cell_mask)

        for j, d in enumerate(distance_intervals):
            for slot, s in enumerate((selected_future_sweeps - 1)):  # selected_future_sweeps: [4, 8, ...]
                curr_valid_pixel_map = valid_pixel_maps[:, s]

                if j == 0:  # corresponds to static cells
                    curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
                else:
                    # We use the displacement between keyframe and the last sample frame as metrics
                    last_gt_norm = all_disp_field_gt_norm[:, -1]
                    mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                    curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                    curr_mask = np.logical_and(curr_mask, moving_cell_mask)

                # Since in nuScenes (with 32-line LiDAR) the points (cells) in the distance are very sparse,
                # we evaluate the performance for cells within the range [-30m, 30m] along both x, y dimensions.
                border = 8
                roi_mask = np.zeros_like(curr_mask, dtype=np.bool_)
                roi_mask[:, border:-border, border:-border] = True
                curr_mask = np.logical_and(curr_mask, roi_mask)

                cell_idx = np.where(curr_mask == True)

                gt = all_disp_field_gt[:, s]
                pred = disp_pred[:, s]
                norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

                cell_groups[j][slot].append(norm_error[cell_idx])

        # -----------------------------------------------------------------------------------
        # Second, compute the classification accuracy
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        non_empty_map_numpy = non_empty_map.numpy()
        class_pred_numpy = class_pred.cpu().numpy()

        # Convert the category map
        max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
        filter_mask = max_prob == 1.0  # Note: some of the cell probabilities are soft probabilities
        pixel_cat_map_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1  # category starts from 1 (background), etc
        pixel_cat_map_numpy = (pixel_cat_map_numpy * non_empty_map_numpy * filter_mask).astype(np.int64)

        class_pred_numpy = np.transpose(class_pred_numpy, (0, 2, 3, 1))
        class_pred_numpy = np.argmax(class_pred_numpy, axis=-1) + 1
        class_pred_numpy = (class_pred_numpy * non_empty_map_numpy * filter_mask).astype(np.int64)

        border = 8
        roi_mask = np.zeros_like(non_empty_map_numpy)
        roi_mask[:, border:-border, border:-border] = 1.0

        tmp = pixel_cat_map_numpy == class_pred_numpy
        denominator = np.sum(non_empty_map_numpy * filter_mask * roi_mask)
        pixel_acc += np.sum(tmp * non_empty_map_numpy * filter_mask * roi_mask) / denominator

        # For computing confusion matrix, in order to compute classification accuracy for each category
        count_mask = non_empty_map_numpy * filter_mask * roi_mask
        idx_fg = np.where(count_mask > 0)

        overall_cls_gt.append(pixel_cat_map_numpy[idx_fg])
        overall_cls_pred.append(class_pred_numpy[idx_fg])

        # print("Finish sample [{}/{}]".format(i + 1, int(np.ceil(sample_nums / float(batch_size)))))

    # Compute the statistics
    dump_res = []

    # Compute the statistics of displacement prediction error
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]

        saver.write("--------------------------------------------------------------\n")
        saver.write("For cells within speed range [{}, {}]:\n\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}".\
                format(selected_future_sweeps[s], mean_error, error_quantile_50)
            saver.write(msg + "\n")
            saver.flush()

        saver.write("--------------------------------------------------------------\n\n")

        dump_res.append(dump_error + dump_error_quantile_50)

    # Compute the statistics of mean pixel classification accuracy
    pixel_acc = pixel_acc / sample_nums
    saver.write("Mean pixel classification accuracy: {}\n".format(pixel_acc))

    # Compute the mean classification accuracy for each object category
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)
    cm = confusion_matrix(overall_cls_gt, overall_cls_pred)
    cm_sum = np.sum(cm, axis=1)
    mean_cat = cm[np.arange(5), np.arange(5)] / cm_sum
    cat_map = {0: 'Bg', 1: 'Vehicle', 2: 'Ped', 3: 'Bike', 4: 'Others'}

    for i in range(len(mean_cat)):
        saver.write("mean cat accuracy of {}: {}\n".format(cat_map[i], mean_cat[i]))
    saver.write("mean instance acc: {}\n".format(np.mean(mean_cat)))
    saver.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='/bev_nuScenes/test/', type=str, help='The path to the [val/test] dataset')
    parser.add_argument('-m', '--model', default=None, type=str, help='The path to the trained model')
    parser.add_argument('-l', '--log', default=None, type=str, help='The path to the txt file for saving eval results')
    parser.add_argument('-s', '--split', default='test', type=str, help='Which split [val/test]')
    parser.add_argument('-b', '--bs', default=8, type=int, help='Batch size')
    parser.add_argument('-w', '--worker', default=8, type=int, help='The number of workers')
    parser.add_argument('-n', '--net', default='MotionNet', type=str, help='Which network [MotionNet/MotionNetMGDA]')
    parser.add_argument('-j', '--jitter', action='store_false', help='Whether to apply jitter suppression')

    args = parser.parse_args()
    print(args)

    # Datasets
    testset = DatasetNuscenes(dataset_root=args.data, split='test', future_frame_skip=0, num_category=5)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.worker)
    # model initialize
    device = 'cuda'
    model = MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=13)
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    # Logging Evaluation details
    eval_file_name = os.path.join(args.log, 'eval.txt')
    eval_saver = open(eval_file_name, 'w')
    eval_motion_displacement(model, eval_saver, use_adj_frame_pred=True, dataloader=testloader, 
                                sample_nums=len(testset), use_motion_state_pred_masking=args.jitter)

