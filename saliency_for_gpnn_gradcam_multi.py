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
def get_cams(augmentations,target_id,method=None,gradcam_scale_cams = False,cnn=None,dataset='imagenet',
             aggregation_results=None,
             patch_size = None,
             indicators_shape = None,
             ref=None,
             ):
    
    device = augmentations.device
    # model = resnet50(pretrained=True)
    # import IPython;IPython.embed()
    # resnet50()
    #GRADCAM_IMPL = 'pytorch_grad_cam'
    #===================================================
    dutils.aggregation_results = aggregation_results
    dutils.patch_size = patch_size
    dutils.indicators_shape = indicators_shape
    #===================================================    
    GRADCAM_IMPL = 'libre_cam'
    print('should go to libre_cam')
    # import ipdb;ipdb.set_trace()
    if method == 'allcam':
        methods = ['relevancecam','gradcam','gradcampp']
    elif method == 'allcam-onlynew':
        methods = [
                    # 'cameras',
                   'layercam']
    elif method in ['allcam2-scale','gpnn-loadgen']:
        methods = [
                    'relevancecam',
                    'gradcam','gradcampp',
                   'layercam'
                   ]        
        
        # import ipdb; ipdb.set_trace()
    elif method in ['gpnn-mycam','gpnn-loadgen-mycam']:
        methods = ['gpnn-mycam']
    else:
        methods = [method]
    out = {}


    assert augmentations.min() >= 0 - 1e-5
    assert augmentations.max() <= 1 + 1e-5    
    if dataset == 'imagenet':
        normalized_augmentations = normalize_tensor(augmentations)

    elif dataset == 'pascal':
        #TODO: instead of DBG_CORRECT_PASCAL_NORMALIZATION, keep the normalization ON by default
        normalized_augmentations = normalize_pascal(augmentations)

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
        if methodi in ['gradcam','gradcampp']:
            from gradcam2.libre_gradcam2 import my_GradCAM_multi
            gCAM_CLASS = my_GradCAM_multi(cnn)
            if 'gCAM_CLASS' not in dutils.__dict__:
                dutils.gCAM_CLASS = gCAM_CLASS
        elif methodi == 'relevancecam':
            from gradcam2.libre_gradcam2 import my_RelevanceCAM
            rCAM_CLASS = my_RelevanceCAM(cnn)
            if 'gCAM_CLASS' not in dutils.__dict__:
                dutils.rCAM_CLASS = rCAM_CLASS
        elif methodi  == 'gpnn-mycam':
            from mycam import MyCAM
            myCAM_CLASS = MyCAM(cnn,dataset)
        elif methodi == 'layercam':
            from benchmark.benchmark_utils import ChangeDir,AddPath
            if 'vgg' in str(cnn.__class__).lower():
                #todo hardcoded for vgg16
                modelname = 'vgg16'
            elif 'resnet' in str(cnn.__class__).lower():
                #todo hardcoded for resnet50
                modelname = 'resnet50'            
            with AddPath(os.path.join('LayerCAM-jittor')):
                
                if modelname  ==  "vgg16":
                    layer_name = "features_30"
                    model_dict = dict(type='vgg16', arch=cnn, layer_name=layer_name, input_size=(224, 224))
                    # only pooling layers:
                    if len(cnn.features) == 44:
                        model_dicts = [
                            dict(type='vgg16', arch=cnn, layer_name='features_23', input_size=(224, 224)),
                            dict(type='vgg16', arch=cnn, layer_name='features_33', input_size=(224, 224)),
                            dict(type='vgg16', arch=cnn, layer_name='features_43', input_size=(224, 224)),
                            
                        ]
                    elif len(cnn.features) == 31:
                        # import ipdb; ipdb.set_trace()
                        model_dicts = [
                            dict(type='vgg16', arch=cnn, layer_name='features_16', input_size=(224, 224)),
                            dict(type='vgg16', arch=cnn, layer_name='features_23', input_size=(224, 224)),
                            dict(type='vgg16', arch=cnn, layer_name='features_30', input_size=(224, 224)),
                            
                        ]
                    else:
                        assert False
                elif modelname == "resnet50":
                    layer_name = "layer4"
                    # model_dict = dict(type='resnet50', arch=cnn, layer_name=layer_name, input_size=(224, 224))
                    # import ipdb; ipdb.set_trace()
                    model_dicts = [
                        dict(type='resnet50', arch=cnn, layer_name='layer1', input_size=(224, 224)),
                        dict(type='resnet50', arch=cnn, layer_name='layer2', input_size=(224, 224)),
                        dict(type='resnet50', arch=cnn, layer_name='layer3', input_size=(224, 224)),
                        dict(type='resnet50', arch=cnn, layer_name='layer4', input_size=(224, 224))
                        
                    ]
                else:
                    raise NotImplementedError            
                from cam.layercam import LayerCAM
                layercams = [LayerCAM(model_dict) for model_dict in model_dicts]

            
    normalized_augmentations = normalized_augmentations.contiguous().requires_grad_(True)
    if os.environ.get('USE_AMP',False) == '1':
        normalized_augmentations = normalized_augmentations.half()
    #!
    # with torch.autocast('cuda'):
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
        myCAM_CLASS.solve(normalized_augmentations,target_id)
        cams,scores,probs = myCAM_CLASS(normalized_augmentations)
        assert not isinstance(cams,torch.Tensor)
        if cams.ndim == 4:
            cams = cams[:,0,...]
        if cams.ndim == 2:
            cams = cams[None,...]
        out['gpnn-mycam'] = (cams,scores,probs,is_top_class)
    # import ipdb;ipdb.set_trace()
    if normalized_augmentations.shape[0] > 1:
        dutils.cipdb('DBG_MYCAM')
        dutils.cipdb('DBG_REGRESSION')
    if 'relevancecam' in methods:
        # with torch.autocast('cuda'):
        if True:
            cams = rCAM_CLASS.after_forward(normalized_augmentations,output.float(),class_idx=target_id,retain_graph=False)
            # import ipdb; ipdb.set_trace()
        # cams = CAM_CLASS(normalized_augmentations, class_idx=target_id)
        """
        if cams.ndim ==3:
            probs = torch.ones(cams.shape[0],device=device) 
            scores = torch.ones(cams.shape[0],device=device)
        else:
            assert False, 'relevance cam should not be returning 2d cams, but 3d cams'
            # probs = 1
            # scores = 1
        """
        # import ipdb;ipdb.set_trace()
        print('TODO: how about using relevancecam instead of gradcam?')
        # assert cams.__class__ 
        assert not isinstance(cams,torch.Tensor)
        # import ipdb;ipdb.set_trace()
        if cams.ndim == 4:
            cams = cams[:,0,...]
        if cams.ndim == 2:
            cams = cams[None,...]
        out['relevancecam'] = (cams,scores,probs,is_top_class)
        printd('relevancecam done')
            
    # if methodi in ['gradcam','gradcampp']:
    if 'gradcam' in methods or 'gradcampp' in methods:
        """
        gCAM_CLASS(normalized_augmentations, 
                                class_idx=target_id,)
        """
        _,cams,cams_pp,scores_ignore,probs_ignore = gCAM_CLASS.after_backward(normalized_augmentations, 
                                                                                scores,probs,class_idx=target_id,
                                                                                retain_graph=False)
        # import ipdb; ipdb.set_trace()
        # if methodi == 'gradcampp':
        #     # import ipdb;ipdb.set_trace()
        #     cams = cams_pp
        # if cams.ndim == 2:
        #     cams = cams[None,...]
        if 'gradcam' in methods:
            cams,scores_ignore,probs_ignore = copy.deepcopy(cams),copy.deepcopy(scores_ignore.detach()),copy.deepcopy(probs_ignore.detach())
            out['gradcam'] = (cams,scores,probs,is_top_class)
        if 'gradcampp' in methods:
            cams,scores_ignore,probs_ignore = copy.deepcopy(cams_pp),copy.deepcopy(scores_ignore.detach()),copy.deepcopy(probs_ignore.detach())
            out['gradcampp'] = (cams,scores,probs,is_top_class)            
        dutils.cipdb('DBG_AVG_SALIENCY')
        printd('gradcam done')

    if 'layercam' in methods:
        
        layercam_maps = []
        for layercam in layercams:
            layercam_mapi = layercam.after_backward(normalized_augmentations,output,class_idx=target_id,retain_graph=False)
            layercam_maps.append(layercam_mapi)
        if isinstance(layercam_maps[0],torch.Tensor):
            layercam_map =  torch.stack(layercam_maps).max(0)[0]                    
        else:
            layercam_map =  np.stack(layercam_maps).max(0)                    
        layercam_map = layercam_map.squeeze(1)
        out['layercam'] = (layercam_map,scores,probs,is_top_class)            
    # import ipdb; ipdb.set_trace()
    #############################################################
    #
    # END<Concurrent CAMS>
    #
    #############################################################    
    # dutils.cipdb('TRACE_STEP_BY_STEP')
    if False:
        for methodi in methods:
            if methodi in ['gradcam','gradcampp']:
                if GRADCAM_IMPL == 'pytorch_grad_cam': 
                    extra_for_cam = {}
                    # import pdb;pdb.set_trace()
                    augmentations.requires_grad_(True)
                    # import pdb;pdb.set_trace()    

                    cams,scores,probs = gradcam.gradcam(augmentations,target=target_id,model_type='imagenet',scale=gradcam_scale_cams,cnn=cnn)
                    probs = torch.ones_like(probs);print('setting probs to ones')
                    assert probs.ndim==1,'expected probs to have only 1 dim'
                    # import pdb;pdb.set_trace()
                elif GRADCAM_IMPL == 'libre_cam':
                    # import ipdb;ipdb.set_trace() 
                    from gradcam2.libre_gradcam import my_GradCAM_multi
                    #==================================================
                    # model, moveto a setup function, eval and cuda
                    #TODO: normalize tensor
                    CAM_CLASS = my_GradCAM_multi(cnn)
                    #==================================================
                    # return a, grad_cam, grad_cam_pp,scores,probs
                    """
                    if dataset == 'imagenet':
                        normalized_augmentations = normalize_tensor(augmentations)
                    elif dataset == 'pascal':
                        #TODO: instead of DBG_CORRECT_PASCAL_NORMALIZATION, keep the normalization ON by default
                        normalized_augmentations = normalize_pascal(augmentations)
                    """
                    # import ipdb; ipdb.set_trace()
                    dutils.cipdb('DBG_PASCAL_NORMALIZATION')
                    _,cams,cams_pp,scores_ignore,probs_ignore = CAM_CLASS(normalized_augmentations, class_idx=target_id)
                    # import ipdb; ipdb.set_trace()
                    if methodi == 'gradcampp':
                        # import ipdb;ipdb.set_trace()
                        cams = cams_pp
                    if cams.ndim == 2:
                        cams = cams[None,...]
                    cams,scores_ignore,probs_ignore = copy.deepcopy(cams),copy.deepcopy(scores_ignore.detach()),copy.deepcopy(probs_ignore.detach())
                    out[methodi] = (cams,scores,probs,is_top_class)
                    dutils.cipdb('DBG_AVG_SALIENCY')
                    printd('gradcam done')
                
            elif methodi == 'relevancecam':
                from gradcam2.libre_gradcam import my_RelevanceCAM
                #from gradcam2.Multi_cam import multi_CAM 
                # from gradcam2.relevance_cam_utils.LRP_util import GradCAM_multi
                #==================================================
                # model, moveto a setup function, eval and cuda
                #TODO: normalize tensor
                device = augmentations.device
                CAM_CLASS = my_RelevanceCAM(cnn)

                """
                assert augmentations.min() >= 0 - 1e-5
                assert augmentations.max() <= 1 + 1e-5
                if dataset == 'imagenet':
                    normalized_augmentations = normalize_tensor(augmentations)
                elif dataset == 'pascal':
                    normalized_augmentations = normalize_pascal(augmentations)

                """
                # assert cnn(normalized_augmentations).argmax() == target_id
                # import ipdb; ipdb.set_trace()
                dutils.cipdb('DBG_AVG_SALIENCY')
                cams = CAM_CLASS(normalized_augmentations, class_idx=target_id)
                """
                if cams.ndim ==3:
                    probs = torch.ones(cams.shape[0],device=device) 
                    scores = torch.ones(cams.shape[0],device=device)
                else:
                    assert False, 'relevance cam should not be returning 2d cams, but 3d cams'
                    # probs = 1
                    # scores = 1
                """
                # import ipdb;ipdb.set_trace()
                print('TODO: how about using relevancecam instead of gradcam?')
                # assert cams.__class__ 
                assert not isinstance(cams,torch.Tensor)
                # import ipdb;ipdb.set_trace()
                if cams.ndim == 4:
                    cams = cams[:,0,...]
                if cams.ndim == 2:
                    cams = cams[None,...]
                out[methodi] = (cams,scores,probs,is_top_class)
                printd('relevancecam done')
        # import ipdb;ipdb.set_trace()
            elif methodi == 'cameras':
                from benchmark.pascal_run_competing_saliency_librecam import run_cameras
                if 'vgg' in str(cnn.__class__).lower():
                    #todo hardcoded for vgg16
                    modelname = 'vgg16'
                elif 'resnet' in str(cnn.__class__).lower():
                    #todo hardcoded for resnet50
                    modelname = 'resnet50'
                """
                assert augmentations.min() >= 0 - 1e-5
                assert augmentations.max() <= 1 + 1e-5
                if dataset == 'imagenet':
                    normalized_augmentations = normalize_tensor(augmentations)
                elif dataset == 'pascal':
                    normalized_augmentations = normalize_pascal(augmentations)

                """
                cams = []
                for i in range(normalized_augmentations.shape[0]):
                    cams_i = run_cameras(cnn,normalized_augmentations[i:i+1],target_id,modelname=modelname,base_method=None,device=None)
                    cams.append(cams_i['saliency'])
                cams = torch.stack(cams,dim=0)
                if isinstance(cams,torch.Tensor):
                    cams = cams.detach().cpu().numpy()
                assert cams.ndim == 3
                # saliency_dict = run_cameras(cnn,normalized_augmentations,target_id,modelname=modelname,base_method=None,device=None)
                # cams = saliency_dict['saliency']
                # probs = torch.ones(cams.shape[0],device=device) 
                # scores = torch.ones(cams.shape[0],device=device)
                out[methodi] = (cams,scores,probs,is_top_class)
                if False:
                    dutils.img_save(cams,f'gpnn-cameras.png')
                    dutils.img_save(augmentations,f'ref-gpnn-cameras.png')
                # import ipdb; ipdb.set_trace()
            elif methodi == 'layercam':
                from benchmark.pascal_run_competing_saliency_librecam import run_layercam
                if 'vgg' in str(cnn.__class__).lower():
                    #todo hardcoded for vgg16
                    modelname = 'vgg16'
                elif 'resnet' in str(cnn.__class__).lower():
                    #todo hardcoded for resnet50
                    modelname = 'resnet50'            
                """
                assert augmentations.min() >= 0 - 1e-5
                assert augmentations.max() <= 1 + 1e-5
                if dataset == 'imagenet':
                    normalized_augmentations = normalize_tensor(augmentations)
                elif dataset == 'pascal':
                    normalized_augmentations = normalize_pascal(augmentations)

                """
                saliency_dict = run_layercam(cnn,normalized_augmentations,target_id,modelname=modelname,base_method=None,device=None)
                dutils.cipdb('DBG_AVG_SALIENCY')
                
                cams = saliency_dict['saliency']
                assert cams.ndim == 4
                cams = cams.squeeze(1)
                # probs = torch.ones(cams.shape[0],device=device) 
                # scores = torch.ones(cams.shape[0],device=device)            
                if isinstance(cams,torch.Tensor):
                    cams = cams.detach().cpu().numpy()
                out[methodi] = (cams,scores,probs,is_top_class)
                if False:
                    dutils.img_save(cams,f'gpnn-layercam.png')
                    dutils.img_save(augmentations,f'ref-gpnn-layercam.png')            
                # import ipdb; ipdb.set_trace()
                printd('layercam done')
            else:
                assert False
    
    return out
#
def permute(indicators,patch_aggregation_info,output_shape,patch_size=(1,1),stride=1):
    if patch_aggregation_info['patch_aggregation'] == 'uniform':
        return permute_using_I(indicators,
                               patch_aggregation_info['I'],
                               output_shape,patch_size=patch_size,
                               patch_aggregation = patch_aggregation_info['patch_aggregation'],
                               stride=1)        
    if patch_aggregation_info['patch_aggregation'] == 'distance-weighted':
        return permute_using_I(indicators,
                               patch_aggregation_info['I'],
                               output_shape,patch_size=patch_size,
                               patch_aggregation = patch_aggregation_info['patch_aggregation'],
                               stride=1,W=patch_aggregation_info['weights'])
    raise NotImplementedError


def permute_using_I(indicators,I,output_shape,patch_size=(1,1),
                    patch_aggregation = None,stride=1,W=None):
    required_numel_I = np.prod([indicators.shape[0]] + [os-2*(ps//2) for os,ps in zip(output_shape,patch_size)])
    # import ipdb;ipdb.set_trace()
    try:
        assert I.shape[0] == required_numel_I
    except AssertionError:
        import ipdb; ipdb.set_trace()
    batch_size = indicators.shape[0]
    patches = torch.stack([extract_patches(di.permute(1,2,0),patch_size,
                                                # model.PATCH_SIZE, 
                                                stride) for di in indicators],dim=0)

    # I1 = I.view(model.batch_size,I.shape[0]//model.batch_size,*I.shape[1:])
    '''
    permuted_patches = torch.cat([patches[i][Ii.T]  for i,Ii in zip( range(batch_size),I.view(batch_size,I.shape[0]//batch_size,*I.shape[1:]))],dim=0)
    '''
    
    assert I.shape[-1] == 1
    permuted_patches = torch.cat([ torch.index_select(patches[i],0,Ii.squeeze(-1)).unsqueeze(0)  for i,Ii in zip( range(batch_size),I.view(batch_size,I.shape[0]//batch_size,*I.shape[1:]))],dim=0)

    assert permuted_patches.ndim == 5,'1,npatches,nchan,7,7'
    '''
    permuted_bhwc = torch.stack([combine_patches(v, 
                                                # model.PATCH_SIZE, 
                                                patch_size,
                                                stride,output_shape,as_np=False,use_divisor=True) for v in permuted_patches],dim=0)
    '''
    permuted_bhwc = []
    for i,v in enumerate(permuted_patches):
        Wi = W[i] if W is not None else None
        # import pdb;pdb.set_trace()
        permuted_bhwc_i = combine_patches(v, 
        # model.PATCH_SIZE, 
        patch_size,
        stride,output_shape,as_np=False,patch_aggregation=patch_aggregation,use_divisor=True,weights=Wi)['combined'] 
        permuted_bhwc.append(permuted_bhwc_i)
    permuted_bhwc = torch.stack(permuted_bhwc,dim=0)
    permuted = permuted_bhwc.permute(0,3,1,2).contiguous()

    return permuted
# """
def unpermute_map_old(map,aggregation_info,patch_size,permuted_shape=None,indicators_shape = None):
    device = map.device
    print('TODO: convert map shape to bchw')
    print(map.shape)
    full_shape = map.shape[-2:]
    '''
    # some jugglery on the cam shape at this point
    # better to send it from get_saliency in a cropped format or not
    if aggregation_info['patch_aggregation'] in ['uniform','distance-weighted']:
        I = aggregation_info['I']
        
    if permuted_shape is None:
        permuted_shape = valid_shape_for_ps1 = full_shape[0] - 2*(patch_size[0]//2),full_shape[1] - 2*(patch_size[1]//2)
    elif permuted_shape == 'fit':
        permuted_shape = valid_shape_for_ps1 = full_shape[0] - 6,full_shape[1] - 6
    else:
        valid_shape_for_ps1 = permuted_shape
    '''
    valid_shape_for_ps1 = permuted_shape = full_shape
    # permuted_shape = valid_shape_for_ps1 = full_shape[0] ,full_shape[1] 
    print('TODO: remove crop_margin, the calling function is responsible for that')
    crop_margin = tuple([(fs-ps)//2 for ps,fs in zip(permuted_shape,full_shape)])
    # indicators = torch.ones(map.shape[0],*valid_shape_for_ps1,1).to(device).requires_grad_(True)
    indicators = torch.ones(map.shape[0],1,*valid_shape_for_ps1).to(device).requires_grad_(True)
    permuted_indicators = permute(indicators,aggregation_info,permuted_shape+(1,),patch_size=patch_size)
    # import ipdb; ipdb.set_trace()
    cropped_map = map[...,crop_margin[0]:(-(crop_margin[0]) if crop_margin[0] >0 else None),crop_margin[1]:(-(crop_margin[0]) if crop_margin[0] >0 else None)]
    if not torch.isclose(permuted_indicators.max(),permuted_indicators.mean()):
        print('permuted indicators max and mean not same')
        if os.environ.get('PERMUTE_ERROR',"1") != "0":
            import pdb;pdb.set_trace()
        
    '''
    assert torch.isclose(permuted_indicators.mean(),torch.ones_like(permuted_indicators.mean()))
    assert torch.isclose(permuted_indicators.std(),torch.zeros_like(permuted_indicators.std()))
    '''
    (permuted_indicators * cropped_map).sum().backward()
    unpermuted_map = indicators.grad
    return unpermuted_map
# """
"""
def unpermute_map(map,aggregation_info,patch_size,permuted_shape=None,indicators_shape=None):
    device = map.device
    print('TODO: convert map shape to bchw')
    print(map.shape)
    full_shape = map.shape[-2:]
    '''
    # some jugglery on the cam shape at this point
    # better to send it from get_saliency in a cropped format or not
    if aggregation_info['patch_aggregation'] in ['uniform','distance-weighted']:
        I = aggregation_info['I']
        
    if permuted_shape is None:
        permuted_shape = valid_shape_for_ps1 = full_shape[0] - 2*(patch_size[0]//2),full_shape[1] - 2*(patch_size[1]//2)
    elif permuted_shape == 'fit':
        permuted_shape = valid_shape_for_ps1 = full_shape[0] - 6,full_shape[1] - 6
    else:
        valid_shape_for_ps1 = permuted_shape
    '''
    if indicators_shape is None:
        valid_shape_for_ps1 = permuted_shape = full_shape
    else:
        valid_shape_for_ps1 = permuted_shape = indicators_shape
    # permuted_shape = valid_shape_for_ps1 = full_shape[0] ,full_shape[1] 
    print('TODO: remove crop_margin, the calling function is responsible for that')
    crop_margin = tuple([(fs-ps)//2 for ps,fs in zip(permuted_shape,full_shape)])
    # indicators = torch.ones(map.shape[0],*valid_shape_for_ps1,1).to(device).requires_grad_(True)
    indicators = torch.ones(map.shape[0],1,*valid_shape_for_ps1).to(device).requires_grad_(True)
    permuted_indicators = permute(indicators,aggregation_info,permuted_shape+(1,),patch_size=patch_size)
    
    cropped_map = map[...,crop_margin[0]:(-(crop_margin[0]) if crop_margin[0] >0 else None),crop_margin[1]:(-(crop_margin[0]) if crop_margin[0] >0 else None)]
    if not torch.isclose(permuted_indicators.max(),permuted_indicators.mean()):
        print('permuted indicators max and mean not same')
        import pdb;pdb.set_trace()
        # import time;time.sleep(1)
        
    '''
    assert torch.isclose(permuted_indicators.mean(),torch.ones_like(permuted_indicators.mean()))
    assert torch.isclose(permuted_indicators.std(),torch.zeros_like(permuted_indicators.std()))
    '''
    if False:
        ignore_margins = True
        if ignore_margins:
            permuted_indicators  = permuted_indicators[...,6:-6,6:-6]
            cropped_map = cropped_map[...,6:-6,6:-6]
            print('ignoring margins')
    print('interploate syntax')
    (torch.nn.functional.interpolate(permuted_indicators,size=cropped_map.shape[-2:],mode='bilinear') * cropped_map).sum().backward()
    unpermuted_map = indicators.grad
    return unpermuted_map
"""

'''
def get_flow(augmentations,I,patch_size,H,W):
    assert False,'aggregation info not impllemented'
    device = augmentations.device
    X,Y = torch.meshgrid(torch.arange(W,device=device),torch.arange(H,device=device),indexing='xy')
    X,Y = X[patch_size[0]//2:-(patch_size[0]//2),patch_size[1]//2:-(patch_size[1]//2)],Y[patch_size[0]//2:-(patch_size[0]//2),patch_size[1]//2:-(patch_size[1]//2)]
    X,Y = X[None,None,...],Y[None,None,...]
    tgt_X = unpermute_map(X,aggregation_info,(1,1))
    tgt_Y = unpermute_map(Y,aggregation_info,(1,1))
    mass =  unpermute_map(torch.ones_like(X),I,patch_size)
    assert False,'remove clamp'
    tgt_X = tgt_X/mass.clamp(1,None)
    tgt_Y = tgt_Y/mass.clamp(1,None)
    flow_X = tgt_X - X
    flow_Y = tgt_Y - Y
    return flow_Y,flow_X
'''
def get_saliency(augmentations,aggregation_results,patch_size,target_id,saliency_method='gradcam',cnn=None,cnns=[],dataset='imagenet',running_saliency=None,running_saliencys=[],ref=None):
    if len(cnns) == 0:
        cnns= [cnn]
    if len(running_saliencys) == 0:
        running_saliencys = [running_saliency]
    device = augmentations.device
    gradcam_scale_cams = False
    assert patch_size[0] == patch_size[1]
    # import ipdb;ipdb.set_trace()
    
    # cams,scores,probs
    indicators_shape=(256,256)
    if os.environ.get('SCALE_AND_CROP',False) == '1' and not os.environ.get('NO_SCALE_AND_CROP',False) == '1':
        n_aug = 5
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
                    scale = np.random.rand(1).item()*0.5 + 0.75
                else:
                    print(colorful.red('^^^^^^^^only high scales^^^^^^^^^'))
                    scale = np.random.rand(1).item()*0. + 1
                a1 = torch.nn.functional.interpolate(a,scale_factor=scale,mode='bilinear',align_corners=False)
                scales.append(scale)
                # sample top left for random crop
                crop_tl_y = 0
                if max(0,a1.shape[-2] - a.shape[-2]) > 0: 
                    crop_tl_y = np.random.randint(0,max(0,a1.shape[-2] - a.shape[-2]))
                crop_tl_x = 0
                if max(0,a1.shape[-1] - a.shape[-1]) > 0:
                    crop_tl_x = np.random.randint(0,max(0,a1.shape[-1] - a.shape[-1]))
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
    # import ipdb; ipdb.set_trace()
    # cam_outs = []
    outs = []
    # import ipdb; ipdb.set_trace()
    for cnn,running_saliency in zip(cnns,running_saliencys):
        cam_out = get_cams(augmentations_for_cam,target_id,method=saliency_method,gradcam_scale_cams=gradcam_scale_cams,cnn=cnn,dataset=dataset,
                       aggregation_results = aggregation_results,
                       patch_size = patch_size,
                       indicators_shape = indicators_shape,
                       ref= ref,
                       )
        if os.environ.get('SCALE_AND_CROP',False) == '1' and not os.environ.get('NO_SCALE_AND_CROP',False) == '1':       
            # extract the regions from the cams and place them back in their position in the original image
            for camname in cam_out:
                cams,scores,probs,is_top_class = cam_out[camname]
                # if camname == 'relevancecam':
                    
                #     assert (1 - cams.view(cams.shape[0],-1).max(-1)[0]).abs().sum() <1e-4
                #     assert (cams.view(cams.shape[0],-1).min(-1)[0]).abs().sum() <1e-4
                # cams =cams/cams.max( -1,keepdim=True)[0].max( -2,keepdim=True)[0]
                assert cams.ndim == 3
                cams = torch.tensor(cams,device=device).unsqueeze(1)
                cams0 = cams
                cams2 = []
                masks = []
                for ci,cami in enumerate(cams):
                    cami = cami[None]
                    assert not cami.requires_grad
                    scale = scales[ci]
                    cam2_ = torch.nn.functional.interpolate(cami[:,:,b_crops_tl[ci][0]:b_crops_tl[ci][1],b_crops_tl[ci][2]:b_crops_tl[ci][3] ],size=cami.shape[-2:],mode='bilinear',align_corners=False)
                    # cam2_ = cam2_[:,:,:cami.shape[2],:cami.shape[3]]
                    cam2 = torch.zeros_like(cami)
                    cam2 = torch.nn.functional.interpolate(cam2,scale_factor=max(scale,1),mode='bilinear',align_corners=False)
                    cam2[:,:,crops_tl[ci][0]:crops_tl[ci][1],crops_tl[ci][2]:crops_tl[ci][3]] = cam2_
                    cam2 = torch.nn.functional.interpolate(cam2,size=cami.shape[-2:],mode='bilinear',align_corners=False)
                    cams2.append(cam2)
                    
                    window = torch.nn.functional.interpolate(torch.ones_like(cami)[:,:,b_crops_tl[ci][0]:b_crops_tl[ci][1],b_crops_tl[ci][2]:b_crops_tl[ci][3]],size=cami.shape[-2:],mode='bilinear',align_corners=False)
                    # window = window[:,:,:cami.shape[2],:cami.shape[3]]
                    mask = torch.zeros_like(cami)
                    mask = torch.nn.functional.interpolate(mask,scale_factor=max(scale,1),mode='bilinear',align_corners=False)
                    mask[:,:,crops_tl[ci][0]:crops_tl[ci][1],crops_tl[ci][2]:crops_tl[ci][3]] = window
                    mask = torch.nn.functional.interpolate(mask,size=cami.shape[-2:],mode='bilinear',align_corners=False)
                    masks.append(mask)
                                
                cams = torch.cat(cams2,dim=0)
                masks = torch.cat(masks,dim=0)

                if False:
                    print(colorful.red("<<<<<<<<<<<<<<cams by cams max>>>>>>>>>>>>>"))
                    cams_max = (cams.view(*cams.shape[:-2],-1).max(dim=-1)[0])
                    cams_min = (cams.view(*cams.shape[:-2],-1).min(dim=-1)[0])
                    if True:
                        denom_nrmz = cams_max-cams_min
                        denom_nrmz = (denom_nrmz  + (denom_nrmz ==0).float())
                        cams = (cams - cams_min[...,None,None])/(denom_nrmz[...,None,None])
                    else:
                        denom_nrmz = cams_max
                        denom_nrmz = (denom_nrmz  + (denom_nrmz ==0).float())
                        cams = (cams)/(denom_nrmz[...,None,None])
                    if True:
                        cams = cams * probs[:,None,None,None]
                    assert cams.max() <= 1.0
                    assert cams.min() >= 0.0

                masked = cams*masks
                masked = masked.view(augmentations.shape[0],n_aug,*masked.shape[1:])
                masks = masks.view(augmentations.shape[0],n_aug,*masks.shape[1:])

                is_top_class1 = is_top_class.float().view(augmentations.shape[0],-1)#.mean(dim=1)
                denom = (masks*is_top_class1[...,None,None,None]).sum(dim=1)
                cams = (masked*is_top_class1[...,None,None,None]).sum(dim=1)/(denom +(denom == 0).float())

                denom2 = is_top_class1.sum(dim=1)
                probs0 = probs
                scores0 = scores
                probs1 = (probs.view(augmentations.shape[0],-1)*is_top_class1).sum(dim=1)/(denom2 +(denom2 == 0).float())
                scores1 = (scores.view(augmentations.shape[0],-1)*is_top_class1).sum(dim=1)/(denom2 +(denom2 == 0).float())
                
                cams = cams.cpu().numpy()[:,0,...]
                cam_out[camname] = (cams,scores1,probs1)
                if False and (camname == 'relevancecam'):
                    import ipdb; ipdb.set_trace()
                    #=======================================
                    ll = 3
                    mm = n_aug*ll + min(0,n_aug-1)
                    cams2t = torch.cat(cams2)
                    dutils.img_save(cams[ll],'c.png',cmap='jet')
                    # dutils.img_save(saliency[ll],'s.png',cmap='jet')
                    dutils.img_save(augmentations[ll],'a.png')
                    dutils.img_save(augmentations_for_cam[mm],'a2.png')
                    dutils.img_save(cams2t[mm],'c2.png',cmap='jet')
                    dutils.img_save(cams0[mm],'c0.png',cmap='jet')
                    print(probs0[mm],scores0[mm],is_top_class[mm])
                    # dutils.img_save(augmentations[ll],'s.png',cmap='jet')
                    #=======================================                        
        # cam_outs.append(cam_out)
        out = {}
        USE_BACKPROP = os.environ.get('USE_BACKPROP',False) == '1'
        EARLY_FUSION = os.environ.get('EARLY_FUSION',False) == '1'
        # MYCAM = os.environ.get('gpnn-mycam',False) == '1'
        # MYCAM =  method in ['gpnn-mycam']
        if USE_BACKPROP:        
            assert running_saliency is not None
            stacked_cams = []
            for k in cam_out.keys():
                cams,scores,probs,is_top_class = cam_out[k]
                # cams_t = torch.tensor(cams)[:,None,...].to(device)
                stacked_cams.append(torch.tensor(cams)[:,None,...].to(device))
                if k not in running_saliency:
                    running_saliency[k] = {}
                if 'running_saliency_backprop' not in running_saliency[k]:
                    assert cams.ndim == 3
                    running_saliency[k]['running_saliency_backprop'] = torch.zeros(1,1,*cams.shape[-2:]).to(device).requires_grad_(True)
            # running_saliency['optimizer'] 
            # if 'optimizer' not in running_saliency[k] :
            if True:
                optimizer = torch.optim.Adam([running_saliency[k]['running_saliency_backprop'] for k in running_saliency],lr=1e-2)
                for k in running_saliency:
                    running_saliency[k]['optimizer'] = optimizer
            else:
                optimizer = running_saliency[k]['optimizer']
            # running_saliency['optimizer'].param_groups[0]['lr'] = 1e0
            # permute running_saliency[k] according to the aggregation_info
            # cams_t = torch.cat([torch.tensor(cam_out[k][0])[:,None,...].to(device) for k in cam_out.keys()],dim=0)
            stacked_cams = torch.cat(stacked_cams,dim=0)
            NORMALIZED_SALIENCY = True
            if NORMALIZED_SALIENCY:
                stacked_cams = stacked_cams/stacked_cams.flatten(1).max(1,keepdim=True)[0].unsqueeze(1).unsqueeze(1)
            MEDIAN_ON_CAMS = True
            if MEDIAN_ON_CAMS:
                print(colorful.orange("<<<<<<<<<<<<median filtering the saliency>>>>>>>>>>>>>"))
                # saliency = weighted_median_filter(saliency, 11, sigma, padding=0) 
                median_k = 11
                m_batch_size = 10
                for mi in range(0,stacked_cams.shape[0],m_batch_size):
                    stacked_cams[mi:mi+m_batch_size] = weighted_median_filter(stacked_cams[mi:mi+m_batch_size], median_k, None, padding=median_k//2)
                # stacked_cams = weighted_median_filter(stacked_cams, median_k, None, padding=median_k//2)
                assert not stacked_cams.isnan().any() 
                assert not stacked_cams.isinf().any()                 
            # stacked_running_saliency = torch.cat(cams_t,dim=0)
            for inneri in range(5):
                stacked_permuted_saliency = []
                for k in cam_out.keys():
                    permuted_running_saliency_k = permute(
                    running_saliency[k]['running_saliency_backprop'].repeat(cams.shape[0],1,1,1),aggregation_results,cams.shape[-2:]+(1,),patch_size=patch_size)
                    stacked_permuted_saliency.append(permuted_running_saliency_k)
                stacked_permuted_saliency = torch.cat(stacked_permuted_saliency,dim=0)
                # Sp = permute_using_I(torch.tile(S,(10,1,1,1,)),aggregation_results['I'],

                #                         S.shape[-2:]+(1,),patch_size=(config['patch_size'],config['patch_size']),

                #                         patch_aggregation = aggregation_results['patch_aggregation'],

                #                         stride=1,W=aggregation_results['weights'])
                loss = (stacked_cams - stacked_permuted_saliency)**2
                # loss = (stacked_cams - stacked_permuted_saliency).abs()
                loss = loss.mean(dim=0).sum()
                print(loss)
                loss.backward()

                optimizer.step()            
                if NORMALIZED_SALIENCY:
                    for k in cam_out.keys():
                        running_saliency[k]['running_saliency_backprop'].data.copy_(
                            running_saliency[k]['running_saliency_backprop'].clip(0,)
                        )
                # permuted_indicators = permute(indicators,aggregation_info,permuted_shape+(1,),patch_size=patch_size)
            # import ipdb; ipdb.set_trace()
            for k in cam_out.keys():
                out[k] = dict(saliency=None,raw_saliency=None,avg_saliency=None,
            std_saliency=None,cams=cam_out[k][0],mass_sum=None)
        else:
            unpermute_map = unpermute_map_old;print('setting unpermute_map to unpermute_map_ol')
            for k in cam_out.keys():
                cams,scores,probs = cam_out[k][0],cam_out[k][1],cam_out[k][2]
                if EARLY_FUSION:
                    assert cams.ndim == 3
                    cams = torch.tensor(cams).unsqueeze(1).to(device) #1,1,333,500
                    indicators_shape=(256,256)
                    patch_size1 = patch_size
                    n_augmentations = augmentations.shape[0]
                    if cams.shape[0] != n_augmentations:
                        for_mass = torch.cat([torch.ones_like(cams) for _ in range(n_augmentations)],dim=0)
                    else:
                        for_mass =torch.ones_like(cams)
                    mass =  unpermute_map(for_mass,aggregation_results,patch_size1,permuted_shape='fit',indicators_shape=indicators_shape)                
                    mass_sum = mass.sum(dim=0,keepdim=True)
                    out[k] = dict(saliency=cams,raw_saliency=None,avg_saliency=cams,
            std_saliency=None,cams=cams,mass_sum=mass_sum)
                    if True:
                        out[k] = {kk:(tensor_to_numpy(v) if isinstance(v,torch.Tensor) else v) for kk,v in out[k].items()}                
                    continue
                # import ipdb; ipdb.set_trace()
                
                if isinstance(cams,torch.Tensor):
                    # layercam_map = tensor_to_numpy(layercam_map)
                    cams = tensor_to_numpy(cams)
                    
                assert cams.ndim == 3
                cams = torch.tensor(cams).unsqueeze(1).to(device) #1,1,333,500
                # dummy = torch.ones_like(cams).requires_grad_(True)
                # debug.stop_at_combine = True
                if True:
                    patch_size1 = patch_size
                else:
                    patch_size1 = (1,1)
                    cams = cams[:,:,patch_size[0]//2:- (patch_size[0]//2),patch_size[1]//2:-(patch_size[1]//2)]
                    aggregation_results['patch_aggregation'] = 'uniform'
                    print('hardcoding patch_aggregation to uniform')
                if aggregation_results['patch_aggregation'] == 'distance-weighted':
                    assert patch_size1 == patch_size, "you can't change patch-size when working with distance-weighted. (the weights will no longer make sense)"
                print('TODO:hardcoded indicators_shape')
                indicators_shape=(256,256)
                # import ipdb;ipdb.set_trace()
                
                unpermuted_saliency = unpermute_map(cams,aggregation_results,patch_size1,permuted_shape='fit',indicators_shape=indicators_shape)
                unpermuted_saliency2 = unpermute_map(cams**2,aggregation_results,patch_size1,permuted_shape='fit',indicators_shape=indicators_shape)
                # import pdb;pdb.set_trace()
                mass =  unpermute_map(torch.ones_like(cams),aggregation_results,patch_size1,permuted_shape='fit',indicators_shape=indicators_shape)
                if False and 'smoothen then divide':
                    '''
                    earlier implementation, keep this for checking
                    '''
                    # due to noisy flow, saliency will have some noise in it
                    # assert False,'miht not need to permute while doing weighted median'
                    smoothed_saliency = weighted_median_filter(unpermuted_saliency, patch_size[0], sigma=None, padding=patch_size[0]//2)
                    smoothed_mass = weighted_median_filter(mass, patch_size[0], sigma=None, padding=patch_size[0]//2)
                    # if a patch is repeated, its saliency will be the sum of each of the repetitions
                    saliency = smoothed_saliency/(smoothed_mass + (smoothed_mass==0).float())
                    if probs.ndim >1:
                        probs = probs.mean(-1)
                    print('see if shapes of prob etc are aligned')
                    # import pdb;pdb.set_trace()
                    if True:
                        weights = probs[:,None,None,None] * (smoothed_mass != 0).float()
                        denom = weights.sum(dim=0,keepdim=True)
                        denom = denom + (denom ==0).float()
                        avg_saliency = (weights * saliency).sum(dim=0,keepdim=True)/denom
                        print('saliency std not implemented')        
                        std_saliency = torch.zeros_like(avg_saliency)
                    else:
                        selector_for_avg_saliency = 1 - (smoothed_mass == 0).float()
                        denom_for_avg_saliency = (selector_for_avg_saliency * (probs[:,None,None,None])).sum(dim=0,keepdim=True)
                        denom_for_avg_saliency = denom_for_avg_saliency + (denom_for_avg_saliency==0).float()
                        avg_saliency = (probs[:,None,None,None] * saliency).sum(dim=0,keepdim=True)/denom_for_avg_saliency
                        print('saliency std not implemented')        
                        std_saliency = torch.zeros_like(avg_saliency)            
                else:
                    'divide and then smoothen'
                    assert unpermuted_saliency.shape == mass.shape
                    # import pdb;pdb.set_trace()
                    saliency = unpermuted_saliency/(mass + (mass==0).float())

                    if dutils.cipdb('DBG_MEM'):
                        import ipdb; ipdb.set_trace()
                    if True:
                        print(colorful.orange("<<<<<<<<<<<<median filtering the saliency>>>>>>>>>>>>>"))
                        # saliency = weighted_median_filter(saliency, 11, sigma, padding=0) 
                        median_k = 11
                        saliency = weighted_median_filter(saliency, median_k, None, padding=median_k//2)
                        assert not saliency.isnan().any() 
                        assert not saliency.isinf().any() 
                    # import pdb;pdb.set_trace()
                    if False:
                        saliency = weighted_median_filter(saliency, patch_size[0], sigma=None,
                        padding=patch_size[0]//2)
                    else:print('skipping post median filter')
                    if probs.ndim >1:
                        probs = probs.mean(-1)
                    print('see if shapes of prob etc are aligned')
                    print('is this a correct way of calculating avg_saliency?')
                    #import pdb;pdb.set_trace()
                    print('an alternate is to get avg_saliency from raw_salincy and then smoothen')
                    assert probs.ndim == 1
                    if False:
                        '''
                        earlier version that wasnt working, TODO remove
                        '''
                        selector_for_avg_saliency = 1 - (mass == 0).float()
                        unselected = (mass.sum(dim=0,keepdim=True) == 0)
                        denom_for_avg_saliency = (torch.ones_like(mass) * (probs[:,None,None,None])).sum(dim=0,keepdim=True)
                        assert denom_for_avg_saliency.shape == unselected.shape
                        denom_for_avg_saliency[unselected] = 1
                        avg_saliency = (probs[:,None,None,None] * saliency).sum(dim=0,keepdim=True)/denom_for_avg_saliency
                        avg_saliency = weighted_median_filter(avg_saliency, patch_size[0], sigma=None,
                            padding=patch_size[0]//2)
                        pass
                    else:
                        selected = (mass != 0)
                        weights = selected * probs[:,None,None,None]
                        # if (probs !=1).any():
                        #     dutils.cipdb('DBG_AVG_SALIENCY3');os.environ['DBG_AVG_SALIENCY'] = '1'
                        
                        denom_for_avg_saliency = weights.sum(dim=0,keepdim=True)
                        # assert denom_for_avg_saliency.shape == selected.shape
                        denom_for_avg_saliency = denom_for_avg_saliency + (denom_for_avg_saliency == 0).float()
                        avg_saliency = (weights * saliency).sum(dim=0,keepdim=True)/denom_for_avg_saliency
                        
                        # os.environ['SCORE_CAM_WEIGHTS'] = '1'
                        import aniket_utils as au
                        au.fpdb()
                        if os.environ.get('SCORE_CAM_WEIGHTS',False) == '1':
                            # import ipdb; ipdb.set_trace()
                            """
                            if dataset == 'imagenet':
                                normalized_augmentations = normalize_tensor(augmentations)

                            elif dataset == 'pascal':
                                #TODO: instead of DBG_CORRECT_PASCAL_NORMALIZATION, keep the normalization ON by default
                                # import ipdb; ipdb.set_trace()
                                normalized_augmentations = normalize_pascal(augmentations)

                            """
                            saliency_max = saliency.amax(dim=(1,2,3,))[:,None,None,None] + 1e-5
                            saliency_min = saliency.amin(dim=(1,2,3,))[:,None,None,None] + 1e-5
                            if False:
                                saliency1 = saliency/saliency_max
                            else:
                                saliency1 = (saliency - saliency_min)/(saliency_max - saliency_min)
                            saliency1.amax(dim=(-1,-2))
                            # this is unpermuted saliency, it needs to be multiplied with the original red
                            if dataset == 'imagenet':
                                normalized_ref = normalize_tensor(ref)

                            elif dataset == 'pascal':
                                normalized_ref = normalize_pascal(ref)
                            
                            maskout = cnn(saliency1*normalized_ref)
                            if maskout.ndim == 4:
                                maskout = maskout.mean(dim=(-1,-2))
                                # import ipdb; ipdb.set_trace()
                            maskprob = torch.softmax(maskout,dim=-1)
                            maskprob = maskprob[:,target_id]

                            reout = cnn(normalized_ref)
                            if reout.ndim == 4:
                                reout = reout.mean(dim=(-1,-2))
                            probre = torch.softmax(reout,dim=-1)
                            probre = probre[:,target_id]
                            """
                            assert False, 'ref is not normalized'

                            """
                            # import ipdb; ipdb.set_trace()
                            if True and "scorecam equation":
                                print(colorful.green('scorecam equation'))
                                avg_saliency = (maskprob[:,None,None,None] * saliency).sum(dim=0,keepdim=True)
                                # avg_saliency = (maskprob[:,None,None,None] * saliency1).sum(dim=0,keepdim=True)
                            elif False and "relative change from ref prob":
                                print(colorful.green('using relative change from ref prob'))
                                weights2= (maskprob-probre)/(1-probre)
                                weights2= torch.nn.functional.relu(weights2)
                                avg_saliency2 = (weights2[:,None,None,None] * saliency).sum(dim=0,keepdim=True)
                                # dutils.img_save(avg_saliency2/avg_saliency2.max(),'sr2.png',cmap='jet')
                            # import ipdb; ipdb.set_trace()


                        
                        # if k == 'relevancecam':
                        #     import ipdb;ipdb.set_trace()
                        if False:
                            avg_saliency = weighted_median_filter(avg_saliency, patch_size[0], sigma=None,padding=patch_size[0]//2)
                        else:print('skipping post median filter')
                        
                    if False and not any( [avg_saliency.max() <= 1,avg_saliency.min() >= 0] ):
                        print('WARNING: saliency max and min not within limits')
                        avg_saliency = (avg_saliency - avg_saliency.min())/(avg_saliency.max() - avg_saliency.min())
                    # import pdb;pdb.set_trace()
                    if False and 'incomplete std, applied on top of smoothed':
                        square_E_of_x = saliency.sum(dim=0,keepdim=True)**2
                        assert False,'smooth mass is missing'
                        E_of_x_square = saliency_2
                        saliency_std = (probs * saliency).sum(dim=0,keepdim=True)/(probs).sum(dim=0,keepdim=True)
                    else:
                        '''std from unpermuted_saliency then smoothen'''
                        mass_sum = mass.sum(dim=0,keepdim=True)
                        denom = mass_sum  + (mass_sum ==0).float()
                        E_raw_saliency = unpermuted_saliency.sum(dim=0,keepdim=True)/denom
                        square_E_of_x = E_raw_saliency**2
                        # square_E_of_x = square_E_of_x/denom
                        E_of_x_square = unpermuted_saliency2.sum(dim=0,keepdim=True)/denom
                        std_raw_saliency = (E_of_x_square - square_E_of_x)
                        if not (std_raw_saliency >= 0).all():
                            print(f'WRONG!!: std is negative,min:{std_raw_saliency.min()}')
                            std_raw_saliency = std_raw_saliency.clamp(0,None)
                        std_raw_saliency[mass_sum==0] = std_raw_saliency.max()
                            # import pdb;pdb.set_trace()
                        std_saliency = weighted_median_filter(std_raw_saliency, patch_size[0], sigma=None,
                        padding=patch_size[0]//2)#[0].permute(1,2,0)
                        #================================================
                        # sampling
                        if False:
                            assert 'sampling not defined'
                            sampling = denom/indicators_for_denominator.grad.shape[0]
                            sampling = sampling/sampling.max()
                out[k] = dict(saliency=saliency,raw_saliency=unpermuted_saliency,avg_saliency=avg_saliency,
            std_saliency=std_saliency,cams=cams,mass_sum=mass_sum)
                if True:
                    out[k] = {kk:tensor_to_numpy(v) for kk,v in out[k].items()}

            # out = \
            # dict(saliency=saliency,raw_saliency=unpermuted_saliency,avg_saliency=avg_saliency,
            # std_saliency=std_saliency,cams=cams,mass_sum=mass_sum)
            # import ipdb;ipdb.set_trace()
        outs.append(out)

    return outs
