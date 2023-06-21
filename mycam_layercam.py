import mycam_
import torch
tensor_to_numpy = lambda x: x.detach().cpu().numpy()
class MyCAM_multiscale():
    def __init__(self,cnn,dataset):
        # self.CAM_CLASSES = []
        self.NORMALIZE = True
        self.dataset = dataset
        mycam_.create_subcams_for_layercam(self,cnn,dataset)
        # self.CAM_CLASSES.append(self.myCAM_CLASS2)
        self.RESIZE = None
    def solve(self,normalized_augmentations,target_id):
        device = normalized_augmentations.device
        normalized_augmentations =normalized_augmentations.requires_grad_(True)
        ignr_feats,scores = self.CAM_CLASSES[0].cnn_forward_pass_(normalized_augmentations,inference_mode=False)
        
        scores[target_id].sum().backward()
        feat_map_sizes = []
        for CAM_CLASS in self.CAM_CLASSES:
            assert len(CAM_CLASS.feat_layers) == 1
            feat_layer = CAM_CLASS.feat_layers[0]
            feats = feat_layer.feats
            assert feats.ndim == 4
            feat_map_sizes.append(feats.shape[-2:])  
        all_cams = []
        for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
            list_of_feats_bchw = CAM_CLASS.get_last_feats_()
            feats_bchw = torch.cat(list_of_feats_bchw,dim=1)
                        
            list_of_grads_bchw = CAM_CLASS.get_last_grads_()
            grads_bchw = torch.cat(list_of_grads_bchw,dim=1)
                        
            CAM_CLASS.w = grads_bchw.mean(dim=(0,-1,-2)).detach()
            CAM_CLASS.b = 0*grads_bchw.mean().detach()
            CAM_CLASS.feats_mean = 0*feats_bchw.mean(dim=(0,-1,-2)).detach()
            CAM_CLASS.feats_std = torch.ones_like(CAM_CLASS.feats_mean).detach()
            
            cam =  CAM_CLASS.get_raw_cam_from_feats(feats_bchw,clip=True)
            cam = mycam_.set_group_max_as_1(cam)
            all_cams.append(cam)
        # import ipdb;ipdb.set_trace()
        cams_fused = mycam_.get_fused_cams(all_cams,feat_map_sizes[-1],self.NORMALIZE,device=device)
        cam_224 = torch.nn.functional.interpolate(cams_fused,(224,224),mode = 'bilinear')
        probs = torch.softmax(scores,dim=1)
        scores=scores[:,target_id]
        probs = probs[:,target_id]
        if scores.ndim == 4:
            scores=scores.mean(dim=(-1,-2))
            probs = probs.mean(dim=(-1,-2))
            
        import ipdb;ipdb.set_trace()
        return cam_224,scores,probs

        
    # """
    def __call__(self,normalized_augmentations):
        if os.environ.get('SCALE_AND_CROP',False) == '1':
            from scale_and_crop_forward import scale_and_crop_forward
            return scale_and_crop_forward(self,normalized_augmentations)
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
        if self.__dict__.get('RESIZE',None) != None:
            normalized_augmentations = torch.nn.functional.interpolate(normalized_augmentations,(self.RESIZE,self.RESIZE),mode='bilinear',align_corners=False)
        import mycam_
        device = normalized_augmentations.device
        with torch.inference_mode():
            ignr_feats,scores = self.CAM_CLASSES[0].cnn_forward_pass_(normalized_augmentations)
        feat_map_sizes = []
        for CAM_CLASS in self.CAM_CLASSES:
            assert len(CAM_CLASS.feat_layers) == 1
            feat_layer = CAM_CLASS.feat_layers[0]
            feats = feat_layer.feats
            assert feats.ndim == 4
            feat_map_sizes.append(feats.shape[-2:])            
        probs = torch.softmax(scores,dim=1)
        all_cams = []
        for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
            list_of_feats_bchw = CAM_CLASS.get_last_feats_()
            feats_bchw = torch.cat(list_of_feats_bchw,dim=1)
            cam =  CAM_CLASS.get_raw_cam_from_feats(feats_bchw,clip=True)
            if self.NORMALIZE:
                # cam = set_max_as_1(cam)
                cam = mycam_.set_group_max_as_1(cam)  
            all_cams.append(cam)
        cam_as_gt = mycam_.get_fused_cams(all_cams,feat_map_sizes[-1],self.NORMALIZE,device=device)
        cam_224 = torch.nn.functional.interpolate(cam_as_gt,(224,224),mode = 'bilinear')
        cam_224 = tensor_to_numpy(cam_224)
        #==================================================================================
        # cam_224 = np.roll(cam_224,(15,-15),axis=(-1,-2))
        import ipdb;ipdb.set_trace()
        return cam_224,scores,probs
                    
