import torch
import random

def decode_augment(disp_gt, class_gt, motion_gt, aug_list):
    # horizontal flip: id 1
    # vertical flip: id 2
    '''
    Input 
    disp_gt: (B,out_seq,2,w,h,)
    '''
    Batch = disp_gt.shape[0]
    for b in range(Batch):
        for aug in aug_list[b]:
            if aug == 1:
                disp_gt[b] = disp_gt[b].flip(dims=[2]) # location flip
                disp_gt[b,:,0] = -disp_gt[b,:,0] # direction flip
                class_gt[b] = class_gt[b].flip(dims=[0])
                motion_gt[b] = motion_gt[b].flip(dims=[0])
            elif aug == 2:
                disp_gt[b] = disp_gt[b].flip(dims=[3])
                disp_gt[b,:,1] = -disp_gt[b,:,1]  # direction flip
                class_gt[b] = class_gt[b].flip(dims=[1])
                motion_gt[b] = motion_gt[b].flip(dims=[1])
    return disp_gt, class_gt, motion_gt

def data_augment(inputs, disp_gt=None, class_gt=None, motion_gt=None, non_empty_map=None, valid_pixel_maps=None, source_bevs=None, factor=0.5):
    '''
    Data augmentation (Inplace)
    Input: [B,seq_len,w,h,c]; disp_gt: [B,future_len,w,h,2]
    class_gt: [B,h,w,cls_num]; motion_gt: [B,h,w,2]
    non_empty_map: [B,h,w]; valid_pixel_maps: [B, future_len, w, h]
    '''
    aug_list = []
    Batch = inputs.shape[0]
    for b in range(Batch):
        b_aug = []
        # horizontal flip: id 1
        if random.random() < factor:
            inputs[b] = inputs[b].flip(dims=[1])
            if disp_gt is not None: 
                disp_gt[b] = disp_gt[b].flip(dims=[1]) # location flip
                disp_gt[b,...,0] = -disp_gt[b,...,0] # direction flip
            if class_gt is not None: class_gt[b] = class_gt[b].flip(dims=[0])
            if motion_gt is not None: motion_gt[b] = motion_gt[b].flip(dims=[0])
            if non_empty_map is not None: non_empty_map[b] = non_empty_map[b].flip(dims=[0])
            if valid_pixel_maps is not None: valid_pixel_maps[b] = valid_pixel_maps[b].flip(dims=[1])
            if source_bevs is not None: source_bevs[b,:,:,0] = 255 - source_bevs[b,:,:,0]
            b_aug.append(1)
        # vertical flip: id 2
        if random.random() < factor:
            inputs[b] = inputs[b].flip(dims=[2])
            if disp_gt is not None: 
                disp_gt[b] = disp_gt[b].flip(dims=[2])
                disp_gt[b,...,1] = -disp_gt[b,...,1]  # direction flip
            if class_gt is not None: class_gt[b] = class_gt[b].flip(dims=[1])
            if motion_gt is not None: motion_gt[b] = motion_gt[b].flip(dims=[1])
            if non_empty_map is not None: non_empty_map[b] = non_empty_map[b].flip(dims=[1])
            if valid_pixel_maps is not None: valid_pixel_maps[b] = valid_pixel_maps[b].flip(dims=[2])
            if source_bevs is not None: source_bevs[b,:,:,1] = 255 - source_bevs[b,:,:,1]
            b_aug.append(2)
        aug_list.append(b_aug)
    #return
    auged_samples = [inputs]
    if disp_gt is not None: auged_samples.append(disp_gt)
    if class_gt is not None: auged_samples.append(class_gt)
    if motion_gt is not None: auged_samples.append(motion_gt)
    if non_empty_map is not None: auged_samples.append(non_empty_map)
    if valid_pixel_maps is not None: auged_samples.append(valid_pixel_maps)
    if source_bevs is not None: auged_samples.append(source_bevs)
    auged_samples.append(aug_list)

    return auged_samples
    
def mask_w_gs(disp_gt, motion_gt, cat_gt, ng_bev):
    device = disp_gt.device
    batch, sample_num = ng_bev.shape[0], ng_bev.shape[1]
    batch_indices = torch.arange(batch)[:, None].repeat(1, sample_num).view(-1)
    if not torch.is_tensor(ng_bev):
        ng_bev = torch.from_numpy(ng_bev).to(disp_gt.device).to(torch.int64)

    ng_disp_gt = disp_gt[batch_indices,:,:,ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]
    disp_gt = disp_gt * 0
    disp_gt[batch_indices,:,:,ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)] = ng_disp_gt

    ng_motion_gt = motion_gt[batch_indices,ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]
    motion_gt[:,:,:] = torch.tensor([1,0,0,0,0], device=device)
    motion_gt[batch_indices,ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)] = ng_motion_gt

    ng_cat_gt = cat_gt[batch_indices,ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]
    cat_gt[:,:,:] = torch.tensor([1,0], device=device)
    cat_gt[batch_indices,ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)] = ng_cat_gt

    return disp_gt, motion_gt, cat_gt

def unlabel_mix(l_input, l_disp_gt, l_class_gt, l_motion_gt, ng_bevs, non_empty_map):
    '''
    Input:
    ng_bev: non-ground bev coordinates for labeled data
    '''
    device = l_input.device
    l_disp_gt = l_disp_gt.to(device)
    l_class_gt = l_class_gt.to(device)
    l_motion_gt = l_motion_gt.to(device)
    if not torch.is_tensor(ng_bevs):
        ng_bevs = torch.from_numpy(ng_bevs).to(l_input.device).to(torch.int64).to(device)
    ng_bev = ng_bevs[:,-1,...].clone()
    batch, sample_num = ng_bev.shape[0], ng_bev.shape[1]
    batch_indices = torch.arange(batch)[:, None].repeat(1, sample_num).to(device)
    temp_indices = torch.arange(5)[:, None].repeat(1, sample_num).to(device)
    
    u_input, u_disp_gt, u_class_gt, u_motion_gt, non_empty_map =\
          l_input.flip(dims=[0]), l_disp_gt.flip(dims=[0]), l_class_gt.flip(dims=[0]), l_motion_gt.flip(dims=[0]), non_empty_map.flip(dims=[0])

    u_input[batch_indices[:,None].repeat(1,5,1).view(-1), temp_indices[None,...].repeat(batch,1,1).view(-1), ng_bevs[...,0].view(-1), ng_bevs[...,1].view(-1)] \
        = l_input[batch_indices[:,None].repeat(1,5,1).view(-1), temp_indices[None,...].repeat(batch,1,1).view(-1), ng_bevs[...,0].view(-1), ng_bevs[...,1].view(-1)] 
    u_disp_gt[batch_indices.view(-1), :, ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1), :]  = l_disp_gt[batch_indices.view(-1), :, ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1), :]
    u_class_gt[batch_indices.view(-1), ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]  = l_class_gt[batch_indices.view(-1), ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]
    u_motion_gt[batch_indices.view(-1), ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]  = l_motion_gt[batch_indices.view(-1), ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)]
    non_empty_map[batch_indices.view(-1), ng_bev[:,:,0].view(-1), ng_bev[:,:,1].view(-1)] = 1

    return u_input, u_disp_gt, u_class_gt, u_motion_gt, non_empty_map
