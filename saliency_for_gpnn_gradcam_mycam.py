from captum.attr import Saliency
from torchvision.models import resnet50
from model.utils import weighted_median_filter
from model.my_gpnn  import extract_patches
# from model.my_gpnn  import combine_patches
from aggregation import combine_patches
import torch
import gradcam
import numpy as np
import debug
import os
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
from gradcam import normalize_tensor
import copy
import dutils
from apex import amp
import importlib
import register_ipdb
importlib.reload(register_ipdb)
printd = print
'''
def normalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    return out
'''

def normalize_pascal(t):
    bgr_mean = [103.939, 116.779, 123.68]
    mean = [m / 255. for m in reversed(bgr_mean)]
    std = [1 / 255.] * 3
    out = normalize_tensor(t,vgg_mean=mean,vgg_std=std)
    return out
    # vgg_transform = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Resize(size),
    #         torchvision.transforms.Normalize(mean=mean,std=std),
    #         ]
    #     )
    # ref = vgg_transform(im_pil).unsqueeze(0)        
def get_cams(augmentations,target_id,method=None,cnn=None,      
            dataset='imagenet',
             ref=None,
             ):
    
    device = augmentations.device    
    #===================================================    
    GRADCAM_IMPL = 'libre_cam'
    print('should go to libre_cam')
    # import ipdb;ipdb.set_trace()
    if method in ['gpnn-mycam','gpnn-loadgen-mycam']:
        methods = ['gpnn-mycam']
    elif method in ['gpnn-mycam-gradavg-lower',
                    'gpnn-loadgen-mycam-gradavg-lower']:
        methods = ['gpnn-mycam']
        os.environ['GRADAVG_LOWER'] = '1'
    elif method in ['gpnn-loadgen-mycam-dff']:
        methods = ['gpnn-mycam']
        os.environ['USE_DFF'] = '1'
    elif method in ['gpnn-loadgen-mycam-1-layer',
                    'gpnn-loadgen-mycam-2-layer',
                    'gpnn-loadgen-mycam-3-layer']:
        methods = ['gpnn-mycam']
        if method == 'gpnn-loadgen-mycam-1-layer':
            os.environ['N_CAM_LAYERS'] = '1'
        elif method == 'gpnn-loadgen-mycam-2-layer':
            os.environ['N_CAM_LAYERS'] = '2'

    else:
        assert False,'method not recognized'

    out = {}


    assert augmentations.min() >= 0 - 1e-5
    assert augmentations.max() <= 1 + 1e-5    
    if dataset == 'imagenet':
        normalized_augmentations = normalize_tensor(augmentations)
    elif dataset == 'pascal':
        #TODO: instead of DBG_CORRECT_PASCAL_NORMALIZATION, keep the normalization ON by default
        normalized_augmentations = normalize_pascal(augmentations)
    is_top_class = None
    if False:
        with torch.no_grad():
            scores = cnn(normalized_augmentations)
        if scores.ndim == 4:
            scores = scores.mean(dim=(-1,-2))
        probs = torch.softmax(scores,dim=1)
        is_top_class = probs[:,target_id] == probs.max(dim=-1)[0]
        probs = probs[:,target_id]
        scores = scores[:,target_id]
        # dutils.cipdb('DBG_AVG_SALIENCY')
    #############################################################
    #
    # Concurrent CAMS
    #
    #############################################################
    for methodi in methods:
        if methodi  == 'gpnn-mycam':
            if os.environ.get('DBG_LAYERCAM',False) == '1':
                from mycam_layercam import MyCAM_multiscale
                myCAM_CLASS = MyCAM_multiscale(cnn,dataset)
            elif os.environ.get('USE_DFF',False) == '1':
                import ipdb;ipdb.set_trace()
            else:
                from mycam import MyCAM,MyCAM_multiscale
                # myCAM_CLASS = MyCAM(cnn,dataset)
                myCAM_CLASS = MyCAM_multiscale(cnn,dataset)      
                          
    if False:            
        normalized_augmentations = normalized_augmentations.contiguous().requires_grad_(True)
        if os.environ.get('USE_AMP',False) == '1':
            normalized_augmentations = normalized_augmentations.half()
        if True:
            # dutils.cipdb('DBG_REGRESSION')
            dutils.cipdb('DBG_MODIFIED_GRADCAM')
            output = cnn(normalized_augmentations)
            
            score = output[:, target_id].sum()
            probs = torch.softmax(output,dim=1)[:,target_id]
            if scores.ndim == 3:
                scores = scores.mean(dim=(-1,-2))
            if probs.ndim == 3:
                probs = probs.mean(dim=(-1,-2))                
            assert scores.ndim == 1,'might not work for pascal VOC'
        cnn.zero_grad()
        # score.backward()
        if os.environ.get('USE_AMP',False) == '1':
            with amp.scale_loss(score,[]) as scaled_score:
                scaled_score.backward()
        else:
            score.backward()
    
    if 'gpnn-mycam' in methods:
        # import ipdb;ipdb.set_trace()
        cams,scores,probs = myCAM_CLASS.solve(normalized_augmentations,target_id)
        # cams,scores,probs = myCAM_CLASS(normalized_augmentations)
        if isinstance(cams,torch.Tensor):
            cams = tensor_to_numpy(cams)
        if cams.ndim == 4:
            cams = cams[:,0,...]
        if cams.ndim == 2:
            cams = cams[None,...]
        out['gpnn-mycam'] = (cams,scores,probs,is_top_class,myCAM_CLASS)
    # import ipdb;ipdb.set_trace()
    if normalized_augmentations.shape[0] > 1:
        dutils.cipdb('DBG_MYCAM')
        dutils.cipdb('DBG_REGRESSION')

    return out

def get_saliency(augmentations,aggregation_results,ignore_patch_size,target_id,saliency_method='gradcam',cnn=None,cnns=[],dataset='imagenet',running_saliency=None,running_saliencys=[],ref=None):
    if len(cnns) == 0:
        cnns= [cnn]
    if len(running_saliencys) == 0:
        running_saliencys = [running_saliency]
    device = augmentations.device

    
    if os.environ.get('SCALE_AND_CROP',False) == '1' and not os.environ.get('NO_SCALE_AND_CROP',False) == '1':
        rng = np.random.RandomState(123)
        n_aug = 1
        # if dataset == 'imagenet':
        #     n_aug = 4
        augmentations_for_cam = []
        scales= []
        crops_tl = []
        b_crops_tl = []
        # masks= []
        
        for a in augmentations:
            a = a[None,...]
            for j in range(n_aug):
                # sample random scale in 0.75,1.25
                # for smaller scale pad to original size
                # for larger take random crop, by sampling random top left
                if True:
                    scale = rng.rand(1).item()*1 + 1
                else:
                    print(colorful.red('^^^^^^^^only high scales^^^^^^^^^'))
                    scale = np.random.rand(1).item()*0. + 1
                a1 = torch.nn.functional.interpolate(a,scale_factor=scale,mode='bilinear',align_corners=False)
                scales.append(scale)
                # sample top left for random crop
                crop_tl_y = 0
                if max(0,a1.shape[-2] - a.shape[-2]) > 0: 
                    crop_tl_y = rng.randint(0,max(0,a1.shape[-2] - a.shape[-2]))
                crop_tl_x = 0
                if max(0,a1.shape[-1] - a.shape[-1]) > 0:
                    crop_tl_x = rng.randint(0,max(0,a1.shape[-1] - a.shape[-1]))
                a1 = a1[...,crop_tl_y:crop_tl_y+a.shape[-2],crop_tl_x:crop_tl_x+a.shape[-1]]
                crops_tl.append((crop_tl_y,crop_tl_y+a.shape[-2],crop_tl_x,crop_tl_x+a.shape[-1]))

                # add padding uniformly in case of smaller scale
                if scale < 1:
                    ydeficit = a.shape[-2] - a1.shape[-2]
                    xdeficit = a.shape[-1] - a1.shape[-1]
                    padleft = xdeficit//2
                    padright = xdeficit - padleft
                    padtop = ydeficit//2
                    padbottom = ydeficit - padtop
                    a1 = torch.nn.functional.pad(a1,(padleft,padright,padtop,padbottom),
                                                #  mode='constant',value=0
                                                # mode = 'replicate'
                                                mode = 'reflect'
                                                 )
                    
                    # mask = torch.nn.functional.pad(mask,((a.shape[-1] - a1.shape[-1])//2,-(a.shape[-1] - a1.shape[-1])//2,(a.shape[-1] - a1.shape[-1])//2, -(a.shape[-1] - a1.shape[-1])//2),mode='constant',value=0)
                    # b_crops_tl.append((a.shape[-1] - mask.shape[-1],a.shape[-1] - mask.shape[-1]))
                    
                    b_crops_tl.append((
                        padleft,
                        a.shape[-2]-padright + 1,
                        padtop, 
                        a.shape[-1]-padbottom + 1))
                else:
                    b_crops_tl.append((0,a1.shape[-2],0,a1.shape[-1]))
                # import ipdb; ipdb.set_trace()
                augmentations_for_cam.append(a1)
                # masks.append(mask)
        augmentations_for_cam = torch.cat(augmentations_for_cam,dim=0)
    else:
        augmentations_for_cam = augmentations
    outs = []
    
    # import ipdb;ipdb.set_trace()
    for cnn,running_saliency in zip(cnns,running_saliencys):
        cam_out = get_cams(augmentations_for_cam,target_id,method=saliency_method,cnn=cnn,dataset=dataset,
        ref= ref)
        # cam_outs.append(cam_out)
        out = {}
        if dataset == 'imagenet':
            normalized_ref = normalize_tensor(ref)
        elif dataset == 'pascal':
            #TODO: instead of DBG_CORRECT_PASCAL_NORMALIZATION, keep the normalization ON by default
            normalized_ref = normalize_pascal(ref)
        # import ipdb; ipdb.set_trace()
        for k in cam_out.keys():
            cams,scores,probs,is_correct_class,myCAM_CLASS = cam_out[k]
            """
            out[k] = dict(saliency=saliency,raw_saliency=unpermuted_saliency,avg_saliency=avg_saliency,
        std_saliency=std_saliency,cams=cams,mass_sum=mass_sum)
            """
            avg_saliency,ref_scores,ref_probs = myCAM_CLASS(normalized_ref)
            # import ipdb;ipdb.set_trace()
            if False:
                dutils.img_save(avg_saliency,'avg_saliency.png',cmap='jet')
                dutils.img_save(ref,'ref.png')
            out[k] = dict(avg_saliency=avg_saliency,mass_sum=1)
            # out[k] = {kk:tensor_to_numpy(v) for kk,v in out[k].items()}

        # out = \
        # dict(saliency=saliency,raw_saliency=unpermuted_saliency,avg_saliency=avg_saliency,
        # std_saliency=std_saliency,cams=cams,mass_sum=mass_sum)
        # import ipdb;ipdb.set_trace()
        outs.append(out)
    dutils.cipdb('DBG_LOAD_AUGMENTATIONS')
    return outs
