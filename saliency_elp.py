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
def get_cams(augmentations,imagenet_target,method=None,gradcam_scale_cams = False,patch_aggregation_info=None,patch_size=None):
    device = augmentations.device
    # model = resnet50(pretrained=True)
    # import IPython;IPython.embed()
    # resnet50()
    if method == 'elp':
        # from extremal_perturbation import *
        import extremal_perturbation
        cnn = resnet50(pretrained=True).to(device)
        cnn.eval()        
        cams,history = extremal_perturbation.extremal_perturbation(cnn,
                          augmentations,
                          imagenet_target,
                          areas=[0.1],
                          perturbation=extremal_perturbation.BLUR_PERTURBATION,
                          max_iter=800,
                          num_levels=8,
                          step=7,
                          sigma=21,
                          jitter=True,
                          variant=extremal_perturbation.PRESERVE_VARIANT,
                          print_iter=None,
                          debug=False,
                          reward_func=extremal_perturbation.simple_reward,
                          resize=False,
                          resize_mode='bilinear',
                          smooth=0,
                          patch_aggregation_info= patch_aggregation_info,
                          patch_size=patch_size)
        nsamples = augmentations.shape[0]        
        probs,scores = torch.ones( nsamples,device=device).float(),torch.ones( nsamples,device=device).float()
        assert cams.ndim == 4
        cams = cams.squeeze(1)
        cams = tensor_to_numpy(cams)
    elif method == 'gradcam':
        
        extra_for_cam = {}
        import ipdb;ipdb.set_trace()
        augmentations.requires_grad_(True)
        # import pdb;pdb.set_trace()    

        cams,scores,probs = gradcam.gradcam(augmentations,target=imagenet_target,model_type='imagenet',scale=gradcam_scale_cams)
        probs = torch.ones_like(probs);print('setting probs to ones')
        assert probs.ndim==1,'expected probs to have only 1 dim'
        # import pdb;pdb.set_trace()
    
    if method == 'gradients':
        '''
        cnn = resnet50(pretrained=True).to('cuda')
        saliency = Saliency(cnn)
        # Computes saliency maps for class 3.
        if False and 'bhwc':
            cams = saliency.attribute(augmentations.permute(0,3,1,2), target=imagenet_target)
        else:
            cams = saliency.attribute(augmentations, target=imagenet_target)
        cams = cams.abs()
        cams = cams.mean(dim = 1); print('averaging the gradients in channel')
        # cams = cams.max(dim = 1); print('averaging the gradients in channel')
        cams = tensor_to_numpy(cams)
        scores = torch.ones(cams.shape[0]).to('cuda'); print('prob = 1')
        probs = torch.ones(cams.shape[0]).to('cuda'); print('scores = 1')
        '''
        cnn = resnet50(pretrained=True).to(device)
        cnn.eval()
        if False and 'replace avgpool layer':
            cnn.avgpool = torch.nn.AvgPool2d(7, stride=1)
            new_fc = torch.nn.Conv2d(2048, 1000, 1,device=device)
            new_fc.weight.data.copy_(cnn.fc.weight[:,:,None,None])
            assert torch.allclose(new_fc.weight,cnn.fc.weight[:,:,None,None])
            new_fc.bias.data.copy_(cnn.fc.bias)
            cnn.fc = new_fc
        
        augmentations1 = augmentations.detach().clone()
        from gradcam import normalize_tensor
        augmentations1 = normalize_tensor(augmentations1)
        augmentations1 = augmentations1.requires_grad_(True)
        # assert False,'preprocessing?'
        scores = cnn(augmentations1)
        probs = torch.softmax(scores,dim=1)
        scores_tgt = scores[:,imagenet_target]
        probs_tgt = probs[:,imagenet_target]
        reward = scores_tgt.sum()
        reward.backward()
        g = augmentations1.grad
        if False and 'simonyan':
            cams = g.abs().max(dim=1)[0]
            cams = cams/cams.max(dim=1,keepdim=True)[0].max(dim=2,keepdim=True)[0]
            cams = tensor_to_numpy(cams)
            scores,probs = scores_tgt,probs_tgt
        elif False and  'mean':
            cams = g.mean(dim=1,keepdim=True)
            cams = tensor_to_numpy(cams)
            scores,probs = scores_tgt,probs_tgt
            probs = torch.ones_like(probs);print('setting probs to ones')
            cams = cams.squeeze(1)
        elif True and  'mask across channels':
            print('using mask across channels mode')
            cams = (g*torch.sign(augmentations1)).mean(dim=1,keepdim=True)
            cams =cams.clamp(0,None)
            cams = cams/cams.max(dim=-2,keepdim=True)[0].max(dim=-1,keepdim=True)[0]
            cams = tensor_to_numpy(cams)
            scores,probs = scores_tgt,probs_tgt
            # probs = torch.ones_like(probs);print('setting probs to ones')
            cams = cams.squeeze(1)
    if method == 'cams for sanity check':
        assert False,'model.* will not work'
        assert cams.ndim == 3, 'n_cams,H,W'
        sanity_cams0 = torch.tile(torch.linspace(0.1,1.,valid_shape_for_ps1[0])[None,:,None],
                                (cams.shape[0],1,valid_shape_for_ps1[-1])).to('cuda')
        # assert sanity_cams.shape == cams.shape
        sanity_cams_values = torch.stack([extract_patches(di,  
                                                        (1,1),
                                                        #  model.PATCH_SIZE, 
                                                        model.STRIDE) for di in sanity_cams0.unsqueeze(-1)],dim=0)
        # import pdb;pdb.set_trace()
        # mask_keys_flat = mask_keys.reshape((mask_keys.shape[0], -1)).contiguous()
        sanity_cams_values = torch.cat([sanity_cams_values[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(model.batch_size,I.shape[0]//model.batch_size,*I.shape[1:]))],dim=0)
        # sanity_cams_values = torch.cat([sanity_cams_values[i][Ii.T]  for i,Ii in zip( range(model.batch_size),I.view(I.shape[0]//model.batch_size,model.batch_size,*I.shape[1:]).permute(1,0,2))],dim=0)
        # dummy_values = dummy_values.squeeze(0)
        # dummy_values = dummy_values.reshape(model.y_pyramid[0].shape[0],
        #                         dummy_values.shape[0]//model.y_pyramid[0].shape[0],*dummy_values.shape[1:])
        assert sanity_cams_values.ndim == 5,'1,npatches,nchan,7,7'
        # import pdb;pdb.set_trace()
        sanity_cams = torch.stack([combine_patches(v,
                                                (1,1),
                                                #   model.PATCH_SIZE, 
                                                model.STRIDE, 
                                                valid_shape_for_ps1 +(1,),as_np=False,use_divisor=True) for v in sanity_cams_values],dim=0)
        sanity_cams = sanity_cams.permute(0,3,1,2).contiguous()

        assert sanity_cams.max() <= 1
        assert sanity_cams.min() >= 0.1
        sanity_cams = sanity_cams.squeeze(1)
        # if cams.shape != sanity_cams.shape:
        #     import pdb;pdb.set_trace()
        cams = sanity_cams
        cams = tensor_to_numpy(cams)
        # cams =  np.ones(cams.shape)
   
    return cams,scores,probs 
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
    assert I.shape[0] == required_numel_I
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
    
    cropped_map = map[...,crop_margin[0]:(-(crop_margin[0]) if crop_margin[0] >0 else None),crop_margin[1]:(-(crop_margin[0]) if crop_margin[0] >0 else None)]
    if not torch.isclose(permuted_indicators.max(),permuted_indicators.mean()):
        print('permuted indicators max and mean not same')
        if os.environ.get('PERMUTE_ERROR',"1") != "0":
            import pdb;pdb.set_trace()
        # import time;time.sleep(1)
        
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
def get_saliency(augmentations,aggregation_results,patch_size,target,saliency_method='gradcam'):
    device = augmentations.device
    gradcam_scale_cams = False
    assert patch_size[0] == patch_size[1]
    cams,scores,probs = get_cams(augmentations,target,method=saliency_method,gradcam_scale_cams=gradcam_scale_cams,patch_aggregation_info=aggregation_results,patch_size=patch_size)
    assert not isinstance(cams,torch.Tensor)
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
    unpermute_map = unpermute_map_old;print('setting unpermute_map to unpermute_map_ol')
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
            denom_for_avg_saliency = weights.sum(dim=0,keepdim=True)
            # assert denom_for_avg_saliency.shape == selected.shape
            denom_for_avg_saliency = denom_for_avg_saliency + (denom_for_avg_saliency == 0).float()
            avg_saliency = (weights * saliency).sum(dim=0,keepdim=True)/denom_for_avg_saliency
            # import pdb;pdb.set_trace()
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


    out = \
    dict(saliency=saliency,raw_saliency=unpermuted_saliency,avg_saliency=avg_saliency,
    std_saliency=std_saliency,cams=cams,mass_sum=mass_sum)
    if True:
        out = {k:tensor_to_numpy(v) for k,v in out.items()}
    return out
