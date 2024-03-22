import numpy as np
import torch

from utils.ot import Cost_Gaussian_function, OT

def voxel2bev(voxel):
    voxel = voxel[:,:2]
    contiguous_array = np.ascontiguousarray(voxel).view(
        np.dtype((np.void, voxel.dtype.itemsize * voxel.shape[1])))
    _, unique_indices = np.unique(contiguous_array, return_index=True)
    voxel = voxel[unique_indices]
    return voxel


def Init_with_prediction(bev_coordinate, disp_preds):
    '''
    Input:
    bev_coordinate: (B, N, 2)
    disp_preds: (B, future_len, 2, w, h)

    Output:
    bev_coordinate (After warping): (B, N, 2)
    disp_pred_reference: (B, N, future_len, 2)
    '''
    batch, sample_num, _ = bev_coordinate.shape
    # Compute the minimum and maximum voxel coordinates
    bev_coordinate = bev_coordinate.to(torch.int64)
    batch_indices = torch.arange(batch)[:, None].repeat(1, sample_num).view(-1)
    # get correspondent prediction
    disp_pred_reference = disp_preds[batch_indices, :, :, bev_coordinate[:, :, 0].view(-1), bev_coordinate[:, :, 1].view(-1)]
    disp_pred_reference = disp_pred_reference.view(batch, sample_num, disp_pred_reference.shape[-2], disp_pred_reference.shape[-1])
    # add 
    bev_coordinate = bev_coordinate + disp_pred_reference[:,:,-1] * 4

    return bev_coordinate, disp_pred_reference

def label_refine(device, source_bev_coord, target_bev_coord, non_empty_map, disp_gt, topk=5, dist_thres=10, args=None):
    '''
    Input:
    source_bev_coord (B, N, 2)
    target_bev_coord (B, N, 2)
    non_empty_map (B, w, h)
    disp_gt (B, future_len, 2, w, h): raw pseudo label

    Output:
    disp_gt (B, future_len, 2, w, h): regenerated pseudo label
    non_empty_map (B, w, h): mask for certain cells
    '''
    source_bev_coord = torch.from_numpy(source_bev_coord).to(device)
    target_bev_coord = torch.from_numpy(target_bev_coord).to(device)
    source_bev_coord_int = source_bev_coord.to(torch.int64)
    batch, sample_num = source_bev_coord.shape[0], source_bev_coord.shape[1]

    warped_source_bev_coord, correspond_disp_gt = Init_with_prediction(source_bev_coord, disp_gt.clone())
    source_bev_coord = source_bev_coord.transpose(2, 1) * 0.25
    target_bev_coord = target_bev_coord.transpose(2, 1) * 0.25
    warped_source_bev_coord = warped_source_bev_coord.transpose(2, 1) * 0.25
    #### find correspondence by optimal transport ####
    # compute cost matrix
    Cost_dist, Support = Cost_Gaussian_function(warped_source_bev_coord.to(torch.float), target_bev_coord.to(torch.float), threshold_2=20)
    Cost = Cost_dist
    # Solve Optimzal transport problem
    T = OT(Cost, epsilon=0.03, OT_iter=4)
    T_indices = T.max(2).indices
    matrix2 = torch.nn.functional.one_hot(T_indices, num_classes=T.shape[2])
    valid_map = matrix2 * Support
    valid_vector = valid_map.sum(dim=2).to(torch.bool)
    # pseudo label
    nn_res_2_point = torch.gather(target_bev_coord, dim=2, index=T_indices[:,None,:].repeat(1,2,1))
    warped_pseudo_label = nn_res_2_point - source_bev_coord
    warped_pseudo_label = warped_pseudo_label * valid_vector[:,None,:]
    # compute the difference between before-after warping
    delta_disp = torch.norm(correspond_disp_gt[:,:,-1,:].transpose(2, 1) - warped_pseudo_label, dim=1)

    # propogate information from certain part to the uncertain part
    certain_masks = (delta_disp<args.lf_disp_thres) # certainty mask

    
    for b in range(batch):
        certain_mask = certain_masks[b]
        certain_point = source_bev_coord_int[b].to(torch.float64)[certain_mask]
        uncertain_point = source_bev_coord_int[b].to(torch.float64)[~certain_mask]
        dist_matrix = uncertain_point[:,None,:] - certain_point[None,:,:] # computer the distances between uncertain and certain cells
        dist_matrix = dist_matrix.norm(dim=-1)

        topk_values, topk_indices = torch.topk(dist_matrix, k=topk, dim=-1, largest=False)
        distance_mask = (topk_values < dist_thres).any(dim=-1) # filter out isolated uncertain points

        gauss_weight = torch.exp(-topk_values / 5) 
        gauss_weight_norm = gauss_weight/ gauss_weight.sum(dim=-1)[:,None] # weights based on distance

        topk_indices = topk_indices[:,:,None,None].repeat(1, 1, correspond_disp_gt.shape[-2], correspond_disp_gt.shape[-1])
        certain_label = correspond_disp_gt[b, certain_mask]
        certain_label = certain_label[None,:,:].repeat(gauss_weight_norm.shape[0], 1, 1, 1)
        certain_label = torch.gather(certain_label, dim=1, index=topk_indices)

        weighted_mean = (certain_label[:,:,-1] * gauss_weight[:,:,None]).sum(dim=1) /  gauss_weight.sum(dim=-1)[:,None]
        weighted_diff = torch.abs((certain_label[:,:,-1] - weighted_mean[:,None,:]) / (weighted_mean[:,None,:] + 1e-6))
        weighted_diff = torch.exp(-(weighted_diff.sum(dim=-1) * gauss_weight).sum(dim=1) / gauss_weight.sum(dim=1))
        homo_mask = (weighted_diff > args.lf_weighted_diff_thres) # filter based on neighbor consistency

        mask = distance_mask & homo_mask

        uncertain_label = (certain_label * gauss_weight_norm[:,:,None,None]).sum(dim=1)
        uncertain_label = uncertain_label[mask]
        count_uncertain_point = uncertain_point[mask].to(torch.int64)
        discard_uncertain_point = uncertain_point[~mask].to(torch.int64)
        disp_gt[b,:,:,count_uncertain_point[:,0],count_uncertain_point[:,1]] = uncertain_label.permute(1,2,0).to(torch.float32)
        non_empty_map[b, discard_uncertain_point[:,0], discard_uncertain_point[:,1]] = 0

    return disp_gt, non_empty_map