import torch
import torch.nn.functional as F
import numpy as np

# Compute and back-propagate the loss
def compute_sloss(device, future_frames_num, disp_gt, valid_pixel_maps, pixel_cat_map_gt,
                        disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred, pred_adj_frame_distance, use_weighted_loss, mask=None, bg_factor=0.02):

    # mask
    if mask is not None:
        disp_pred = disp_pred.view(len(mask), future_frames_num, disp_pred.shape[-3], disp_pred.shape[-2], disp_pred.shape[-1])
        disp_pred = disp_pred[mask]
        disp_pred = disp_pred.view(-1, disp_pred.shape[-3], disp_pred.shape[-2], disp_pred.shape[-1])
        class_pred = class_pred[mask]
        motion_pred = motion_pred[mask]
    # Compute the displacement loss
    disp_gt = disp_gt.view(-1, disp_gt.size(2), disp_gt.size(3), disp_gt.size(4))
    disp_gt = disp_gt.permute(0, 3, 1, 2).to(device)

    # valid_pixel_maps = all_valid_pixel_maps[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return [None] * 3
    # ---------------------------------------------------------------------
    # -- Generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))
        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

    # ---------------------------------------------------------------------
    # -- Compute the masked displacement loss
    if use_weighted_loss:  # Note: have also tried focal loss, but did not observe noticeable improvement
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1
        cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
        weight_vector = [bg_factor, 1.0, 1.0, 1.0, 1.0]  # [bg, car & bus, ped, bike, other]
        for k in range(5):
            mask = pixel_cat_map_gt_numpy == (k + 1)
            cat_weight_map[mask] = weight_vector[k]

        cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
        cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
        map_shape = cat_weight_map.size()

        loss_disp = criterion(disp_gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
        loss_disp = loss_disp.reshape(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
        loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num
    else:
        loss_disp = criterion(disp_gt * valid_pixel_maps, disp_pred * valid_pixel_maps) / valid_pixel_num

    # ---------------------------------------------------------------------
    # -- Compute the grid cell classification loss
    non_empty_map = non_empty_map.view(-1, 256, 256)
    non_empty_map = non_empty_map.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)

    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    if use_weighted_loss:
        map_shape = cat_weight_map.size()
        cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
        loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    else:
        loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1)
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Compute the speed level classification loss
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    if use_weighted_loss:
        motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
        motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
        weight_vector = [bg_factor, 1.0]
        for k in range(2):
            mask = motion_gt_numpy == (k + 1)
            motion_weight_map[mask] = weight_vector[k]

        motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
        loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
    else:
        loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1)
    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    loss = loss_disp + loss_class + loss_motion

    return loss, loss_disp.item(), loss_class.item(), loss_motion.item()

def compute_uloss(device, future_frames_num, disp_gt, pixel_cat_map_gt, disp_pred, non_empty_map, 
                        class_pred, motion_gt, motion_pred, pred_adj_frame_distance, use_weighted_loss, mask):
    '''
    Input:
    disp_gt: (B,out_seq,w,h,2)
    '''
    # CPU/GPU
    disp_gt = disp_gt.permute(0,1,4,2,3)
    disp_gt = disp_gt.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.to(device)
    motion_gt = motion_gt.to(device)
    non_empty_map = non_empty_map.to(device)

    # mask
    disp_pred = disp_pred.view(len(mask), future_frames_num, disp_pred.shape[-3], disp_pred.shape[-2], disp_pred.shape[-1])
    disp_pred = disp_pred[mask]
    disp_pred = disp_pred.view(-1, disp_pred.shape[-3], disp_pred.shape[-2], disp_pred.shape[-1])
    class_pred = class_pred[mask]
    motion_pred = motion_pred[mask]
    # ---------------------------------------------------------------------
    # -- Generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))
        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
    
    # ---------------------------------------------------------------------
    # get non-ground and ground mask
    # ng_mask = torch.zeros_like(non_empty_map, dtype=torch.bool)
    # ng_mask[batch_indices, source_bev_coord_int[:,:,0].view(-1), source_bev_coord_int[:,:,1].view(-1)] = True
    # # mask invalid area
    weighted_matrix = torch.ones_like(non_empty_map, dtype=torch.float)
    weighted_matrix[disp_gt[:,-1,:,:,:].norm(dim=1) <= 0.2] = 0.1 #0.1

    indices = torch.nonzero(non_empty_map)
    loss_disp= F.smooth_l1_loss((disp_pred*weighted_matrix[:,None,None,:,:])[indices[:,0], :,:, indices[:,1], indices[:,2]], 
                                    (disp_gt*weighted_matrix[:,None,None,:,:])[indices[:,0], :,:, indices[:,1], indices[:,2]])
    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    loss = loss_disp

    return loss, loss_disp.item()