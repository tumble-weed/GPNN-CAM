import numpy as np
import torch
import os
def set_max_as_1(cam):                    
    M = cam.max(axis=-1,keepdims=True).max(axis=-2,keepdims=True)
    denom = (M + (M == 0).astype(np.float32))
    cam = cam/denom
    return cam

def set_group_max_as_1(cam):                    
    M = cam.max()
    denom = (M + (M == 0).astype(np.float32))
    cam = cam/denom
    return cam
def get_fused_cams(all_cams,feat_map_size,NORMALIZE,device,
                #    interpolation_mode='nearest'
                interpolation_mode='bilinear',
                softmax_factor=1,
                   ):
    # device = all_cams[0].device
    softmax_factor=1
    MODE = 'prod'
    cam_as_gt = 1
    for ix,cami in enumerate(all_cams):
        if not isinstance(cami,torch.Tensor):
            cami = torch.tensor(cami,device=device)
        if NORMALIZE:
            
            if ix == 0:
                factor= softmax_factor
            if ix == 1:
                factor= softmax_factor
            if ix == 2:
                factor= softmax_factor
            # if len(all_cams) -1 == ix:
            #     factor = None
            # print(factor)
            if factor is not None:
                assert cami.min() >= 0 and cami.max() <= 1
                cami = torch.sigmoid(factor*(cami - 0.5))
                """
                cami = (cami - torch.sigmoid(factor*(torch.zeros_like(cami) - 0.5)) )/ (
                    torch.sigmoid(factor*(torch.ones_like(cami) - 0.5)) - torch.sigmoid(factor*(torch.zeros_like(cami) - 0.5))
                )
                """
        # cami = torch.sigmoid(20*(torch.tensor(cami,device=device)))
        
        camiup = torch.nn.functional.interpolate(cami[:,None],feat_map_size,
                                                mode = interpolation_mode
            )
        # camiup = torch.sigmoid(20*(camiup - 0.5))
        if MODE == 'prod':
            cam_as_gt = cam_as_gt * camiup
        elif MODE == 'sum':
            cam_as_gt = cam_as_gt + camiup
        elif MODE == 'max':
            if ix == 0:
                cam_as_gt =  camiup
            else:
                cam_as_gt = torch.maximum(cam_as_gt, camiup)
    if MODE == 'sum':
        cam_as_gt = cam_as_gt/len(all_cams)
        pass
    elif MODE == 'prod':
        # cam_as_gt = cam_as_gt**(1/len(all_cams))
        pass
    return cam_as_gt
def get_feat_layer_grads(cnn,feats,target_id,dataset):
    feats = feats.clone().requires_grad_(True)
    feats.retain_grad()
    print(cnn.__class__.__name__.lower())
    if 'vgg' in cnn.__class__.__name__.lower():
        if dataset == 'pascal' or dataset == 'voc':
            scores = cnn.classifier(feats)
        elif dataset == 'imagenet':
            x = cnn.avgpool(feats)
            x = x.reshape(x.size(0), -1)
            scores = cnn.classifier(x)            

        if True:
            scores[:,target_id].sum().backward()
        elif False:
            (scores[:,target_id].amax(dim=-1).amax(dim=-1)).sum().backward()
        elif False:
            (scores - scores.amax(dim=1,keepdim=True).detach())[:,target_id].amax(dim=-1).amax(dim=-1).sum().backward()
             
             
        grad = feats.grad
    elif 'resnet' in cnn.__class__.__name__.lower():
        z = cnn.avgpool(feats)
        if ('voc' not in dataset) and ('pascal' not in dataset):
            z = z.view(z.size(0), -1)
        scores = cnn.fc(z)
        scores[:,target_id].sum().backward()
        grad = feats.grad
    else:
        assert False
    # import ipdb;ipdb.set_trace()
    return grad                
def solve_stable(self,normalized_augmentations,target_id,dataset):
    
    # NORMALIZE = True

    device = normalized_augmentations.device
    normalized_augmentations =normalized_augmentations.requires_grad_(True)
    with torch.inference_mode(mode=True):
        ignr_feats,scores = self.CAM_CLASSES[0].cnn_forward_pass_(normalized_augmentations)
    probs_max = torch.softmax(scores,dim=1).amax(dim=1).squeeze()
    probs = torch.softmax(scores,dim=1)[:,target_id].squeeze()
    # scores[:,target_id].sum().backward()
    feat_map_sizes = []
    for CAM_CLASS in self.CAM_CLASSES:
        assert len(CAM_CLASS.feat_layers) == 1
        feat_layer = CAM_CLASS.feat_layers[0]
        feats = feat_layer.feats
        assert feats.ndim == 4
        feat_map_sizes.append(feats.shape[-2:])
    # import ipdb;ipdb.set_trace()
    all_cams = []
    for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
        if ci == 0:
            
            list_of_feats_bchw = CAM_CLASS.get_last_feats_()
            feats_bchw = torch.cat(list_of_feats_bchw,dim=1)
            if os.environ.get('DBG_MYCAM_GRAD',False) == '1':

                grads_bchw = get_feat_layer_grads(CAM_CLASS.hooked_cnn.cnn,feats_bchw,target_id,dataset)
                if True and 1:
                    gradcam_w = grads_bchw.mean(dim=(-1,-2),keepdim=True)
                    cam_as_gt = (gradcam_w * feats_bchw).sum(dim=1,keepdim=True).clip(0)
                elif False and 2:
                    cam_as_gt = (grads_bchw * feats_bchw).sum(dim=1,keepdim=True).clip(0)
                
                # cam_as_gt = cam_as_gt/cam_as_gt.max()
                if os.environ.get('NO_INITIAL_REGRESSION',False) == '1':
                    
                    avg_grads_with_mask(CAM_CLASS,feats_bchw,torch.ones(feats_bchw[:,:1].shape,device=device).float(),target_id,dataset)
                    """
                    
                    CAM_CLASS.w = gradcam_w.mean(dim=(0,-1,-2)).detach()
                    CAM_CLASS.b = 0*gradcam_w.mean().detach()
                    CAM_CLASS.feats_mean = 0*feats_bchw.mean(dim=(0,-1,-2)).detach()
                    CAM_CLASS.feats_std = torch.ones_like(CAM_CLASS.feats_mean).detach()
                    
                    """
                    
                    

                else:
                    
                    if os.environ.get('USE_SAMPLE_WEIGHTS',False) == '1':
                        sample_weight= (probs/probs_max).detach().cpu().numpy()
                        sample_weight=sample_weight[...,None].repeat(np.prod(feats_bchw.shape[-2:]),axis=-1).flatten()
                    else:
                        sample_weight=None
                    CAM_CLASS.solve2_using_previous_forward(feats_bchw.detach(),cam_as_gt.detach(),
                                                            sample_weight=sample_weight)
            else:
                CAM_CLASS.solve_using_previous_forward(feats_bchw,scores,target_id,
                                                       sample_weight=None)
        else:
            cam_as_gt = get_fused_cams(all_cams,feat_map_sizes[ci],self.NORMALIZE,device=device)
            
            list_of_feats_bchw = CAM_CLASS.get_last_feats_()
            feats_bchw = torch.cat(list_of_feats_bchw,dim=1)                
            if os.environ.get('GRADAVG_LOWER',False) == '1':
                # import ipdb;ipdb.set_trace()
                avg_grads_with_mask(CAM_CLASS,feats_bchw,cam_as_gt,target_id,dataset)
                """
                grads_bchw = get_feat_layer_grads(CAM_CLASS.hooked_cnn.cnn,feats_bchw,target_id,dataset)
                CAM_CLASS.w = (cam_as_gt*grads_bchw).sum(dim=(0,-1,-2)).detach()/cam_as_gt.sum()
                CAM_CLASS.b = 0*gradcam_w.mean().detach()
                CAM_CLASS.feats_mean = 0*feats_bchw.mean(dim=(0,-1,-2)).detach()
                CAM_CLASS.feats_std = torch.ones_like(CAM_CLASS.feats_mean).detach()
                # import ipdb;ipdb.set_trace()
                """
            else:
                if False and 'make later maps smaller':
                    cam_as_gt = get_fused_cams(all_cams,feat_map_sizes[0],self.NORMALIZE,device=device)
                    feats_bchw_small= torch.nn.functional.interpolate(feats_bchw,feat_map_sizes[0],mode='bilinear',antialias=True)
                    CAM_CLASS.solve2_using_previous_forward(feats_bchw_small,cam_as_gt,sample_weight=None)
                else:
                    CAM_CLASS.solve2_using_previous_forward(feats_bchw,cam_as_gt,sample_weight=None)

        cam =  CAM_CLASS.get_raw_cam_from_feats(feats_bchw,clip=True)
        if True:
            if self.NORMALIZE:
                # cam = set_max_as_1(cam)
                cam = set_group_max_as_1(cam)
                
            # import ipdb; ipdb.set_trace()
            if scores.ndim == 4:
                scores= scores.mean(dim=(-1,-2))
            assert cam.ndim == 3
            # cam = cam * tensor_to_numpy(scores)[:,target_id,None,None]
        all_cams.append(cam)
        cams_remaining = len(self.CAM_CLASSES) - 1 - ci
        if not isinstance(cam,torch.Tensor):
            cam = torch.tensor(cam,device=device)
        if cams_remaining > 0:
            cam_up = torch.nn.functional.interpolate(cam[:,None],feat_map_sizes[ci+1],mode = 'bilinear')
    # assert isinstance(cam,torch.Tensor)
    # assert cam.ndim == 3
    cam_as_gt = get_fused_cams(all_cams,feat_map_sizes[-1],self.NORMALIZE,device=device)
    assert cam_as_gt.ndim == 4
    cam_224 = torch.nn.functional.interpolate(cam_as_gt,(224,224),mode = 'bilinear')
    # import ipdb;ipdb.set_trace()
    probs = torch.softmax(scores,dim=1)
    return cam_224,scores,probs
"""
def solve_stable_old(self,normalized_augmentations,target_id):
    # import ipdb;ipdb.set_trace()
    NORMALIZE = True
    SINGLE_FORWARD = True
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
    # import ipdb;ipdb.set_trace()
    all_cams = []
    for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
        if ci == 0:
            if not SINGLE_FORWARD:
                CAM_CLASS.solve(normalized_augmentations,target_id)
            else:
                list_of_feats_bchw = CAM_CLASS.get_last_feats_()
                feats_bchw = torch.cat(list_of_feats_bchw,dim=1)
                CAM_CLASS.solve_using_previous_forward(feats_bchw,scores,target_id)
        else:
            cam_as_gt = get_fused_cams(all_cams,feat_map_sizes[ci],NORMALIZE,device=device)
            if not SINGLE_FORWARD:
                CAM_CLASS.solve2(normalized_augmentations,target_id,cam_as_gt,forward_pass_done=True)
            else:
                list_of_feats_bchw = CAM_CLASS.get_last_feats_()
                feats_bchw = torch.cat(list_of_feats_bchw,dim=1)                
                CAM_CLASS.solve2_using_previous_forward(feats_bchw,cam_as_gt)
        if not SINGLE_FORWARD:
            cam,scores,probs = CAM_CLASS.get_raw_cam(normalized_augmentations,clip=True)
        else:
            cam =  CAM_CLASS.get_raw_cam_from_feats(feats_bchw,clip=True)
        if True:
            if NORMALIZE:
                # cam = set_max_as_1(cam)
                cam = set_group_max_as_1(cam)
                
            # import ipdb; ipdb.set_trace()
            if scores.ndim == 4:
                scores= scores.mean(dim=(-1,-2))
            assert cam.ndim == 3
            # cam = cam * tensor_to_numpy(scores)[:,target_id,None,None]
        all_cams.append(cam)
        cams_remaining = len(self.CAM_CLASSES) - 1 - ci
        if not isinstance(cam,torch.Tensor):
            cam = torch.tensor(cam,device=device)
        if cams_remaining > 0:
            cam_up = torch.nn.functional.interpolate(cam[:,None],feat_map_sizes[ci+1],mode = 'bilinear')
    # assert isinstance(cam,torch.Tensor)
    # assert cam.ndim == 3
    cam_as_gt = get_fused_cams(all_cams,feat_map_sizes[-1],NORMALIZE,device=device)
    assert cam_as_gt.ndim == 4
    cam_224 = torch.nn.functional.interpolate(cam_as_gt,(224,224),mode = 'bilinear')
    # import ipdb;ipdb.set_trace()
    probs = torch.softmax(scores,dim=1)
    return cam_224,scores,probs
""" 
def solve_logistic(self,normalized_augmentations,target_id):
    """
    # import ipdb;ipdb.set_trace()
    
    device = normalized_augmentations.device
    with torch.inference_mode():
        ignr_feats,ignr_scores = self.CAM_CLASSES[0].cnn_forward_pass_(normalized_augmentations)
    
    feat_map_sizes = []
    for CAM_CLASS in self.CAM_CLASSES:
        assert len(CAM_CLASS.feat_layers) == 1
        feat_layer = CAM_CLASS.feat_layers[0]
        feats = feat_layer.feats
        assert feats.ndim == 4
        feat_map_sizes.append(feats.shape[-2:])
    # import ipdb;ipdb.set_trace()
    all_cams = []
    for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
        if ci == 0:
            CAM_CLASS.solve(normalized_augmentations,target_id)
        else:
            if False:
                CAM_CLASS.solve2(normalized_augmentations,target_id,cam_up)
            else:
                cam_as_gt = 1
                for cami in all_cams:
                    cami = torch.sigmoid(20*(torch.tensor(cami,device=device) - 0.5))
                    camiup = torch.nn.functional.interpolate(torch.tensor(cami,device=device)[:,None],feat_map_sizes[ci],mode = 'bilinear')
                    # camiup = torch.sigmoid(20*(camiup - 0.5))
                    cam_as_gt = cam_as_gt * camiup
                    cam_as_gt = (cam_as_gt > 0.).float()
                    def inverse_sigmoid(x):
                        return torch.log(x/(1-x))
                    eps = 1e-4
                    cam_as_gt = inverse_sigmoid(cam_as_gt.clip(eps,1-eps))
                CAM_CLASS.solve2(normalized_augmentations,target_id,cam_as_gt)
        cam,scores,probs = CAM_CLASS.get_raw_cam(normalized_augmentations,clip=True)
        if ci > 0:
            # for later cams just do sigmoid
           cam = torch.sigmoid(torch.tensor(cam,device=device)).cpu().numpy()
        else:
            # for first cam just make max 1
            M = cam.max(axis=-1,keepdims=True).max(axis=-2,keepdims=True)
            denom = (M + (M == 0).astype(np.float32))
            cam = cam/denom
            # import ipdb; ipdb.set_trace()
            if scores.ndim == 4:
                scores= scores.mean(dim=(-1,-2))
            assert cam.ndim == 3
            # cam = cam * tensor_to_numpy(scores)[:,target_id,None,None]
        all_cams.append(cam)
        cams_remaining = len(self.CAM_CLASSES) - 1 - ci
        if cams_remaining > 0:
            cam_up = torch.nn.functional.interpolate(torch.tensor(cam,device=device)[:,None],feat_map_sizes[ci+1],mode = 'bilinear')
    cam_224 = torch.nn.functional.interpolate(torch.tensor(cam,device=device)[:,None],(224,224),mode = 'bilinear')
    import ipdb;ipdb.set_trace()
    """
    
def create_subcams(self,cnn,dataset):
    from mycam import MyCAM
    self.use_pca = False
    self.CAM_CLASSES = []
    for _ in range(1):
        if dataset == 'imagenet':

            
                
            # self.myCAM_CLASS2 = MyCAM(cnn,dataset,feat_layers=[cnn.features[33]])
            if 'vgg' in str(cnn.__class__):
                OFFSET = -2
                assert [
                    'conv2d' in cnn.features[43 + OFFSET].__class__.__name__.lower(),
                    'batchnorm2d' in cnn.features[43 + OFFSET].__class__.__name__.lower(),
                ]
                
                self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.features[43 + OFFSET]])
                
                self.CAM_CLASSES.append(self.myCAM_CLASS)
                if os.environ.get('N_CAM_LAYERS',False) in ['2','1']:
                    break                    
                
                self.CAM_CLASSES.append(
                    MyCAM(cnn,dataset,feat_layers=[cnn.features[33 + OFFSET]],use_pca = self.use_pca)
                )                
                if os.environ.get('N_CAM_LAYERS',False) in ['1']:
                    break                    
                
                if True:

                    self.CAM_CLASSES.append(
                        MyCAM(cnn,dataset,feat_layers=[cnn.features[23 + OFFSET]],use_pca = self.use_pca)
                    )
                    # self.CAM_CLASSES.append(
                    #     MyCAM(cnn,dataset,feat_layers=[cnn.features[13 + OFFSET]],use_pca = self.use_pca)
                    # )
            elif 'resnet' in str(cnn.__class__):
                # import ipdb;ipdb.set_trace()
                self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.layer4])
                self.CAM_CLASSES.append(self.myCAM_CLASS)                
                if os.environ.get('N_CAM_LAYERS',False) in ['2','1']:
                    break                    
                
                if True:
                    self.CAM_CLASSES.append(
                        MyCAM(cnn,dataset,feat_layers=[cnn.layer3],use_pca = self.use_pca))
                    if os.environ.get('N_CAM_LAYERS',False) in ['1']:
                        break                    
                    
                    self.CAM_CLASSES.append(
                            MyCAM(cnn,dataset,feat_layers=[cnn.layer2],use_pca = self.use_pca))
                    if False:
                        self.CAM_CLASSES.append(
                            MyCAM(cnn,dataset,feat_layers=[cnn.layer1],use_pca = self.use_pca))

        elif dataset == 'pascal':
            if 'vgg' in str(cnn.__class__):
                # OFFSET = -2
                OFFSET = 0
                OFFSET2 = 0
                # assert 'linear' in cnn.features[30 + OFFSET].__class__.__name__.lower()
                # assert any([
                #     'conv2d' in cnn.features[30 + OFFSET].__class__.__name__.lower(),
                #     'batchnorm2d' in cnn.features[30 + OFFSET].__class__.__name__.lower(),
                # ])
                
                self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.features[30 + OFFSET]],keep_grad = os.environ.get('DBG_MYCAM_GRAD',False) == '1')
                self.CAM_CLASSES.append(self.myCAM_CLASS)                
                if os.environ.get('N_CAM_LAYERS',False) in ['2','1']:
                    break
                # import ipdb;ipdb.set_trace()
                # MaxPool: 4,9,16,23,30 
                # self.myCAM_CLASS2 = MyCAM(cnn,dataset,feat_layers=[cnn.features[23]])
                for i2,fix in enumerate([
                    
                23+OFFSET+OFFSET2,
                16+OFFSET+OFFSET2,
                
                
                # 9,
                # 4,
                # 16-2,
                # 9-2,
                # 4-2
                ]):
                    self.CAM_CLASSES.append(
                    MyCAM(cnn,dataset,feat_layers=[cnn.features[fix]],use_pca = self.use_pca)
                )
                    
                    if os.environ.get('N_CAM_LAYERS',False) in ['2'] and i2 == 0:
                        break                    
        
            elif 'resnet' in str(cnn.__class__):
                pass
                # import ipdb;ipdb.set_trace()
                self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.layer4])
                self.CAM_CLASSES.append(self.myCAM_CLASS)             
                if os.environ.get('N_CAM_LAYERS',False) in ['2','1']:
                    break                    
                                   
                self.CAM_CLASSES.append(
                    MyCAM(cnn,dataset,feat_layers=[cnn.layer3],use_pca = self.use_pca))
                if os.environ.get('N_CAM_LAYERS',False) in ['1']:
                    break                    
                
                self.CAM_CLASSES.append(
                    MyCAM(cnn,dataset,feat_layers=[cnn.layer2],use_pca = self.use_pca))
                if False:

                    self.CAM_CLASSES.append(
                        MyCAM(cnn,dataset,feat_layers=[cnn.layer1],use_pca = self.use_pca))                
                    
        
        
def create_subcams_for_layercam(self,cnn,dataset):
    from mycam import MyCAM
    self.use_pca = False
    self.CAM_CLASSES = []
    if dataset == 'imagenet':

        
            
        # self.myCAM_CLASS2 = MyCAM(cnn,dataset,feat_layers=[cnn.features[33]])
        if 'vgg' in str(cnn.__class__):
            OFFSET = -2
            assert [
                'conv2d' in cnn.features[43 + OFFSET].__class__.__name__.lower(),
                'batchnorm2d' in cnn.features[43 + OFFSET].__class__.__name__.lower(),
            ]
            self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.features[43 + OFFSET]],
                                     keep_grad = True)
            self.CAM_CLASSES.append(self.myCAM_CLASS)
            self.CAM_CLASSES.append(
                MyCAM(cnn,dataset,feat_layers=[cnn.features[33 + OFFSET]],use_pca = self.use_pca,
                      keep_grad = True)
            )                
            if True:

                self.CAM_CLASSES.append(
                    MyCAM(cnn,dataset,feat_layers=[cnn.features[23 + OFFSET]],use_pca = self.use_pca,keep_grad = True)
                )
                # self.CAM_CLASSES.append(
                #     MyCAM(cnn,dataset,feat_layers=[cnn.features[13 + OFFSET]],use_pca = self.use_pca)
                # )
        elif 'resnet' in str(cnn.__class__):
            # import ipdb;ipdb.set_trace()
            self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.layer4],keep_grad = True)
            self.CAM_CLASSES.append(self.myCAM_CLASS)                
            if True:
                self.CAM_CLASSES.append(
                        MyCAM(cnn,dataset,feat_layers=[cnn.layer3],use_pca = self.use_pca,keep_grad = True)
                    )
                self.CAM_CLASSES.append(
                        MyCAM(cnn,dataset,feat_layers=[cnn.layer2],use_pca = self.use_pca,keep_grad = True)
                        
                        )
                if False:
                    self.CAM_CLASSES.append(
                        MyCAM(cnn,dataset,feat_layers=[cnn.layer1],use_pca = self.use_pca))

    elif dataset == 'pascal':
        if 'vgg' in str(cnn.__class__):
            # OFFSET = -2
            OFFSET = 0
            OFFSET2 = 0
            # assert 'linear' in cnn.features[30 + OFFSET].__class__.__name__.lower()
            # assert any([
            #     'conv2d' in cnn.features[30 + OFFSET].__class__.__name__.lower(),
            #     'batchnorm2d' in cnn.features[30 + OFFSET].__class__.__name__.lower(),
            # ])
            
            self.myCAM_CLASS = MyCAM(
                                    cnn,dataset,feat_layers=[cnn.features[30 + OFFSET]],
                                     keep_grad = True
                                     )
            self.CAM_CLASSES.append(self.myCAM_CLASS)                
            # import ipdb;ipdb.set_trace()
            # MaxPool: 4,9,16,23,30 
            # self.myCAM_CLASS2 = MyCAM(cnn,dataset,feat_layers=[cnn.features[23]])
            for fix in [
                
            23+OFFSET+OFFSET2,
            16+OFFSET+OFFSET2,
            
            
            # 9,
            # 4,
            # 16-2,
            # 9-2,
            # 4-2
            ]:
                self.CAM_CLASSES.append(
                MyCAM(cnn,dataset,feat_layers=[cnn.features[fix]],use_pca = self.use_pca,
                      keep_grad = True)
            )
        elif 'resnet' in str(cnn.__class__):
            pass
            # import ipdb;ipdb.set_trace()
            self.myCAM_CLASS = MyCAM(cnn,dataset,feat_layers=[cnn.layer4],keep_grad = True)
            self.CAM_CLASSES.append(self.myCAM_CLASS)                                
            self.CAM_CLASSES.append(
                MyCAM(cnn,dataset,feat_layers=[cnn.layer3],use_pca = self.use_pca,keep_grad = True)
                )
            self.CAM_CLASSES.append(
                MyCAM(cnn,dataset,feat_layers=[cnn.layer2],use_pca = self.use_pca,keep_grad = True)
                )
            if False:

                self.CAM_CLASSES.append(
                    MyCAM(cnn,dataset,feat_layers=[cnn.layer1],use_pca = self.use_pca))                
                
    
def avg_grads_with_mask(CAM_CLASS,feats_bchw,cam_as_gt,target_id,dataset):
    grads_bchw = get_feat_layer_grads(CAM_CLASS.hooked_cnn.cnn,feats_bchw,target_id,dataset)
    CAM_CLASS.w = (cam_as_gt*grads_bchw).sum(dim=(0,-1,-2)).detach()/cam_as_gt.sum()
    CAM_CLASS.b = 0*CAM_CLASS.w.mean().detach()
    CAM_CLASS.feats_mean = 0*feats_bchw.mean(dim=(0,-1,-2)).detach()
    CAM_CLASS.feats_std = torch.ones_like(CAM_CLASS.feats_mean).detach()
    pass
    