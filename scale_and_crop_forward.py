import os
import torch
import numpy as np
tensor_to_numpy = lambda x: x.detach().cpu().numpy()
def scale_and_crop_forward(self,normalized_augmentations):
    
    normalized_augmentations0 = normalized_augmentations
    device = normalized_augmentations.device
    n_stack = 40
    normalized_augmentations_stack = []
    for i in range(n_stack):
        normalized_augmentations_i = torch.nn.functional.interpolate(normalized_augmentations,scale_factor = 1 + i*(1)/(n_stack-1),mode='bilinear',align_corners=False)
        normalized_augmentations_stack.append(normalized_augmentations_i)
    normalized_augmentations_stack = list(reversed(normalized_augmentations_stack))
    """
    device = normalized_augmentations.device
    last_CAM_CLASS = self.CAM_CLASSES[-1]
    # import ipdb;ipdb.set_trace()
    return last_CAM_CLASS(normalized_augmentations)
    """
    # roll by a small amount
    # normalized_augmentations = torch.roll(normalized_augmentations,(-15,15),dims=(-1,-2))
    # dutils.img_save(normalized_augmentations[0],'ref_rolled.png')
    #==================================================================================
    import mycam_
    device = normalized_augmentations.device
    cam_stack = []
    weights = []
    feat_stack = [0 for _ in (self.CAM_CLASSES)]
    probs = None
    for si,normalized_augmentations in enumerate(normalized_augmentations_stack):
        with torch.inference_mode():
            ignr_feats,scores = self.CAM_CLASSES[0].cnn_forward_pass_(normalized_augmentations)
            if scores.ndim == 4:
                scores = scores.mean(dim=(-1,-2))
            probs = torch.softmax(scores,dim=1)
            weight = probs[:,self.target_id]/probs.amax(dim=1)
            weights.append(weight.item())
            
        if si == 0:
            feat_map_sizes = []
            for CAM_CLASS in self.CAM_CLASSES:
                assert len(CAM_CLASS.feat_layers) == 1
                feat_layer = CAM_CLASS.feat_layers[0]
                feats = feat_layer.feats
                assert feats.ndim == 4
                feat_map_sizes.append(feats.shape[-2:])            
        
        for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
            list_of_feats_bchw = CAM_CLASS.get_last_feats_()
            feats_bchw = torch.cat(list_of_feats_bchw,dim=1)
            feat_stack[ci] = feat_stack[ci] + weight* torch.nn.functional.interpolate(feats_bchw,size=feat_map_sizes[ci],mode='bilinear')
    weights = np.array(weights)
    weight_sum = weights.sum()
    # weight_sum = n_stack
    import ipdb;ipdb.set_trace()
    all_cams = []
    for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
        feat_stack[ci] = feat_stack[ci]/weight_sum
        cam =  CAM_CLASS.get_raw_cam_from_feats(feat_stack[ci],clip=True)
        all_cams.append(cam)
    cam_as_gt = mycam_.get_fused_cams(all_cams,feat_map_sizes[-1],self.NORMALIZE,device=device)
    cam_224 = torch.nn.functional.interpolate(cam_as_gt,(224,224),mode = 'bilinear')
    cam_224 = tensor_to_numpy(cam_224)
    return cam_224,scores,probs
                
