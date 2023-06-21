import torch
from collections import defaultdict
from model.my_gpnn import gpnn
import numpy as np
import skimage.io
import copy
from model.utils import *
from aggregation import combine_patches_multiple_images
from matplotlib import pyplot as plt
import blur
TODO = None
# order = 4
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def get_cnn(datasetname,modelname):
    if datasetname == 'voc2007':
        import architectures.vgg16
        # model = architectures.vgg16.vgg16(pretrained=True)
        model = architectures.get_model(arch='vgg16',
                dataset='voc',
                convert_to_fully_convolutional=True)
        model.eval()
        print('TODO:is it the maxpool layer for vgg16? ')
        print('TODO: move loading the model to outside (global')
    elif datasetname == 'imagenet':
        if modelname == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            model.eval()
    print('')
    return model
def normalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    return out
def combine_using_visibility(saliency_patches_A,saliency_patches_B,patches_A,patches_B,order,
            patch_size,img_shape,stride,patch_aggregation):
    assert (saliency_patches_A <= 1).all()
    assert (saliency_patches_A >= 0).all()
    assert (saliency_patches_B <= 1).all()
    assert (saliency_patches_B >= 0).all()
    
    visibility_denom = saliency_patches_A**order + saliency_patches_B**order
    all_bad = (visibility_denom == 0.)
    saliency_patches_A[all_bad] = saliency_patches_A[all_bad] + 1
    saliency_patches_B[all_bad] = saliency_patches_B[all_bad] + 1
    visibility_denom = saliency_patches_A**order + saliency_patches_B**order
    still_bad = (visibility_denom == 0.)
    assert not still_bad.any()
    # visibility_denom = visibility_denom + (visibility_denom==0.).float()
    # import ipdb;ipdb.set_trace()
    visibility_A = saliency_patches_A**order/visibility_denom
    visibility_B = saliency_patches_B**order/visibility_denom
    assert torch.allclose(visibility_A + visibility_B,torch.ones_like(visibility_A) )
    interpolated_patches = patches_B*visibility_B + patches_A*visibility_A
    # interpolated = [TODO.combine_patches]
    patch_size = interpolated_patches.shape[-2:]
    assert len(patch_size) == 2
    interpolated = combine_patches_multiple_images(
        interpolated_patches, 
            patch_size, stride, img_shape+(3,),
            as_np=False,
            patch_aggregation=patch_aggregation,
            distances=None,I=TODO)['combined'][0]
    # import pdb;pdb.set_trace()
    assert not interpolated.isnan().any()
    return interpolated

def run(config=None):
    
    config_inpainting_ = copy.deepcopy(config)
    print('TODO:we are not deleting the input_img in the hope that the hole part will not be considered by gpnn (we can also multiply it out later...)')
    # del config_inpainting_['input_img']
    print('TODO:check what elements need to be deleted from config_inpainting')
    config_inpainting_['task'] = 'inpainting'    
    #========================================================
    ref = config['input_img']    
    patch_size = config['patch_size']
    n_outer_iter = config['n_outer_iter'] = 1
    thresh = config['thresh'] = 0.5
    assert not hasattr(patch_size,'__len__')
    stride = config['stride']
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    n_inner_iter = config['n_inner_iter'] = 100
    patch_aggregation = config['patch_aggregation'] = 'uniform'
    visualize_every = config['visualize_every'] = 1
    masking_method = 'elp'
    #========================================================    
    if 'setup':
        os.system(f'rm -rf {config["out_dir"]}')
        os.makedirs(config["out_dir"])
        trends = defaultdict(list)
        gpnn_augmentation = gpnn(config)
        
        ref = torch.tensor(ref,device=device).float().unsqueeze(0).permute(0,3,1,2)
        from model.my_gpnn import extract_patches
        ref_patches = extract_patches(ref.permute(0,2,3,1), (patch_size,patch_size), stride)
        assert ref_patches.ndim == 4, 'like 169,3,7,7'
        
        img_shape = ref.shape[-2:]
        n_patches = np.prod([img_shape[0] - 2*(patch_size//2),img_shape[1] - 2*(patch_size//2)])
        if masking_method == 'saliency_patches':
            n_saliency_channels = 1
            unfolded_shape = n_patches,n_saliency_channels,patch_size,patch_size
            print('scalin saliency patches')
            saliency_patches_ = (0 + 1 * torch.rand(n_patches,1,1,1,device=device)).requires_grad_(True)
            saliency_patches = torch.ones(unfolded_shape,device=device)*saliency_patches_
            # import pdb;pdb.set_trace()
            assert saliency_patches.min() >=0,saliency_patches.max()<=1
            lr = config['lr'] = 1e-1
            optimizer_saliency = torch.optim.Adam([saliency_patches_],lr=lr)
            
        elif masking_method == 'elp':
            from model.elp_mask import MaskGenerator
            step = 7
            sigma = 21
            shape = ref.shape[-2:]
            mask_generator = MaskGenerator(shape, step, sigma).to(device)
            h, w = mask_generator.shape_in
            saliency_ = (torch.rand(1, 1, h, w)**16).to(device).requires_grad_(True)
            saliency,_ = mask_generator.generate(saliency_)
            saliency_patches = extract_patches(saliency.permute(0,2,3,1), (patch_size,patch_size), stride)          
            lr = config['lr'] = 1e-2
            optimizer_saliency = torch.optim.Adam([saliency_],lr=lr)
        cnn = get_cnn(config['datasetname'],config['modelname']).to(device)
        print('STAGE:initial assets created')
        # print('view saliency assets')
        # import ipdb;ipdb.set_trace()
    #========================================================    

    print('TODO: how does evolving saliency effect the need for newer augmentations?')
    for i  in range(n_outer_iter):
        # n_patches  = np.prod(unfolded_shape[2:4])
        all_I = torch.arange(n_patches,device=device)[:,None]
        if masking_method == 'saliency_patches':
            
            saliency_noise = 0. * torch.rand(saliency_patches.shape).to(device)
            binary_saliency_patches = (saliency_patches + saliency_noise)> thresh
            # import ipdb;ipdb.set_trace()
            print('TODO: is this compatible with extract_patches shape')
            # assert binary_saliency_patches.shape[:2] == (1,1)
            fg_mask = binary_saliency_patches[...,0,0]
            assert fg_mask.dtype == torch.bool
            # assert np.prod(fg_mask.shape[:4]) == n_patches
            assert fg_mask.ndim == 2, '127500,1'
            fg_I = all_I[fg_mask][:,None]
            bg_I = all_I[~fg_mask][:,None]
            # picks out bg patches using indicators ( alternate to using I like above)
            bg_saliency_indicator = binary_saliency_patches.clone().zero_()
            bg_saliency_indicator[bg_I.T] = 1
        '''
        fg_patches = saliency_patches * binary_saliency_patches
        bg_patches = saliency_patches * (1-binary_saliency_patches)
        '''
        #====================================================================
        if False and 'make the augmentations':
            augmentations = gpnn_augmentation.run(TODO,TODO.batch_size)
            pass
        else:
            print('using original image for the augmentation')
            augmentations = ref.detach().clone().to(device)
            distances = torch.zeros(1,1,augmentations.shape[-2]-6,augmentations.shape[-1]-6).float().to(device)
            #patches = torch.zeros((1,3,augmentations.shape[-2]-6,augmentations.shape[-1]-6,7,7))
            I = all_I

        #====================================================================  
        assert I.shape[-1] == 1
        
        if False and 'setup for hole-filling':
            print('TODO: remove the image tensor from the config to prevent memory leakage')
            config_inpainting = copy.deepcopy(config_inpainting_)
            #=========================================================
            print('Too many permutations going on. TODO: is this correct?')
            ref_patches_augmentation = ref_patches[I.T]
            # ref_patches_augmentation = ref_patches_augmentation.unsqueeze(0)
            print('TODO: why does ref_patches index dim0 while saliency_patches index dim 2')
            saliency_patches_augmentation = saliency_patches[I.T]
            # only_bg_saliency_patches_augmentations = saliency_patches_augmentation * bg_saliency_indicator[I.T]
            if False and 'ypeError: list indices must be integers or slices, not list':
                assert all([item.shape[[0,1,3,4]] == ref_patches_augmentation.shape[[0,1,3,4]] for item in [ref_patches_augmentation,saliency_patches_augmentation,only_bg_saliency_patches_augmentations]])
            #=========================================================         
            # create inpainting mask
            '''
            """
            this didnt work, it was causing too many values to either be ON or OFF while making the pyramid
            """
            config_inpainting['mask'] =  combine_patches_multiple_images(only_bg_saliency_patches_augmentations, 
            (patch_size,patch_size), stride, img_shape+(1,),
            as_np=False,
            patch_aggregation=patch_aggregation,
            distances=None,I=TODO)['combined'][0]
            '''
            config_inpainting['mask'] =  combine_patches_multiple_images(saliency_patches_augmentation, 
            (patch_size,patch_size), stride, img_shape+(1,),
            as_np=False,
            patch_aggregation=patch_aggregation,
            distances=None,I=TODO)['combined'][0]            
            config_inpainting['mask'] = (config_inpainting['mask'] < 0.5)
            import pdb;pdb.set_trace()
            assert config_inpainting['mask'].ndim == 3
            assert config_inpainting['mask'].__class__ == torch.Tensor
            tensor_to_numpy = lambda t:t.detach().cpu().numpy()
            print('should the holes be 1 and non-holes 0 , or will it be done inside gpnn?')
            config_inpainting['mask'] = tensor_to_numpy(config_inpainting['mask'][:,:,0])
            gpnn_inpainting = gpnn(config_inpainting)
            # hole_filled_results,TODO.others = gpnn_inpainting(config_inpainting)
            holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
            print('copy and compare parameters with hole_filling.py')
            ref_patches_holefilled = ref_patches[hole_filled_results['I'].T]
            saliency_patches_holefilled = saliency_patches[hole_filled_results['I'].T]
            

        for j in range(n_inner_iter):
            #==========================================================
            if True and 'setup for hole-filling':
                print('TODO: remove the image tensor from the config to prevent memory leakage')
                config_inpainting = copy.deepcopy(config_inpainting_)
                #=========================================================
                print('Too many permutations going on. TODO: is this correct?')
                ref_patches_augmentation = ref_patches[I.T]
                # ref_patches_augmentation = ref_patches_augmentation.unsqueeze(0)
                print('TODO: why does ref_patches index dim0 while saliency_patches index dim 2')
                saliency_patches_augmentation = saliency_patches[I.T]
                # only_bg_saliency_patches_augmentations = saliency_patches_augmentation * bg_saliency_indicator[I.T]
                if False and 'ypeError: list indices must be integers or slices, not list':
                    assert all([item.shape[[0,1,3,4]] == ref_patches_augmentation.shape[[0,1,3,4]] for item in [ref_patches_augmentation,saliency_patches_augmentation,only_bg_saliency_patches_augmentations]])
                #=========================================================     
                # create inpainting mask
                if masking_method == 'elp':
                    saliency, ignore = mask_generator.generate(saliency_)
                    saliency_patches = extract_patches(saliency.permute(0,2,3,1), (patch_size,patch_size), stride)
                    saliency = saliency[0,0].unsqueeze(-1)
                    
                elif masking_method == 'saliency_patches':
                    saliency =  combine_patches_multiple_images(saliency_patches_augmentation, 
                    (patch_size,patch_size), stride, img_shape+(1,),
                    as_np=False,
                    patch_aggregation=patch_aggregation,
                    distances=None,I=TODO)['combined'][0]
                    # saliency = blur.blur(saliency.unsqueeze(1), window_size=11, sigma=2).squeeze(1)
                print('view saliency')
                # import ipdb;ipdb.set_trace()                    
                config_inpainting['mask'] = saliency
                # convert to binary
                fg_mask = (config_inpainting['mask'] >= thresh).float()
                config_inpainting['mask'] = fg_mask
                assert config_inpainting['mask'].ndim == 3
                assert config_inpainting['mask'].__class__ == torch.Tensor
                tensor_to_numpy = lambda t:t.detach().cpu().numpy()
                print('GPNN expects that regions to be filled are 1')
                config_inpainting['mask'] = tensor_to_numpy(config_inpainting['mask'][:,:,0])
                # import ipdb;ipdb.set_trace()
                if False:
                    print('hardcoding holefilling mask to bottom right')
                    config_inpainting['mask'] = np.zeros(config_inpainting['mask'].shape)
                    config_inpainting['mask'][...,-100:,-100:] = 1                
                # remove regions so that keys dont contain them
                if False:
                    config_inpainting['input_img'] = (1 - config_inpainting['mask'][...,None]) * config['input_img']

                assert config_inpainting['mask'].sum() > 0,'there is no hole to be filled, everything is bg'
                trends['hole_size'].append(config_inpainting['mask'].sum().item())
                
                gpnn_inpainting = gpnn(config_inpainting)   
                # gpnn_inpainting.keys_to_keep = bg_I.squeeze()
                # hole_filled_results,TODO.others = gpnn_inpainting(config_inpainting)
                holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
                # import ipdb;ipdb.set_trace()
                print('copy and compare parameters with hole_filling.py')
                ref_patches_holefilled = ref_patches[hole_filled_results['I'].T]
                saliency_patches_holefilled = saliency_patches[hole_filled_results['I'].T]            
                print('TODO:once holes are filled see the I, they should be different only in the holes region')
                # import ipdb;ipdb.set_trace()
            #==========================================================
            print('TODO: check if all arguments are correct')
            print('TODO:check bfore interpolation')
            assert  saliency_patches_augmentation.shape == saliency_patches_holefilled.shape
            assert ref_patches_augmentation.shape == ref_patches_holefilled.shape
            order = 2;print(f'setting order to {order}')
            interpolated = combine_using_visibility(
                saliency_patches_augmentation,
                saliency_patches_holefilled,
                ref_patches_augmentation,
                ref_patches_holefilled,
                order,
                patch_size,img_shape,stride,patch_aggregation)
            assert interpolated.ndim == 3
            assert interpolated.shape[-1] == 3
            interpolated = interpolated.permute(2,0,1).unsqueeze(0)
            #=======================================================
            #NOTE: only the interpolated1 image has the optimizable quantity
            interpolated1 = normalize_tensor(interpolated)
            if True and 'forward-backward step':
                scores = cnn(interpolated1)     
                loss = - scores[:,config['target_class']].sum()
                optimizer_saliency.zero_grad()
                loss.backward()
                trends['loss'].append(loss.item())
                optimizer_saliency.step()                
                # recreate the saliency patches
                
                print('TODO:1st experiment without smoothness')
                if masking_method ==  'saliency_patches':
                    saliency_patches_.data.copy_(saliency_patches_.clamp(0,1))
                    saliency_patches = torch.ones(unfolded_shape,device=device)* saliency_patches_
                elif masking_method ==  'elp':
                    saliency_.data.copy_(saliency_.clamp(0,1))
                    saliency,_ = mask_generator.generate(saliency_)
                    saliency_patches = extract_patches(saliency.permute(0,2,3,1), (patch_size,patch_size), stride)          
            #=======================================================
            if (i*j) % visualize_every == 0:    
                # print('TODO: it is the holefilled image')
                # img_bg = TODO.combine_patches()
                if masking_method == 'elp':
                    saliency, ignore = mask_generator.generate(saliency_)
                    saliency = saliency[0]
                elif masking_method == 'saliency_patches':
                    saliency = combine_patches_multiple_images([saliency_patches], 
                    (patch_size,patch_size), stride, img_shape+(1,),
                    as_np=False,
                    patch_aggregation=patch_aggregation,
                    distances=None,I=TODO)['combined'][:,:,:,0]
                    # Test the function
                
                # saliency = blur.blur(saliency.unsqueeze(1), window_size=11, sigma=2).squeeze(1)

                trends['augmentations'] = tensor_to_numpy(augmentations)
                # import ipdb;ipdb.set_trace()
                trends['interpolated'] = tensor_to_numpy(interpolated)
                trends['holefilled']  = tensor_to_numpy(holefilled_images)
                trends['saliency'] = tensor_to_numpy(saliency)
                trends['inpainting_input'] = [config_inpainting['input_img'].transpose(2,0,1)]
                DD = np.abs((trends['inpainting_input'][0] - trends['holefilled'][0])).sum()
                # import ipdb;ipdb.set_trace()
                trends['diff_interpolated_augmentation'] = tensor_to_numpy((interpolated-augmentations).abs()[:,0])
                trends['diff_holefilled_augmentation'] = tensor_to_numpy((holefilled_images-augmentations).abs()[:,0])
                
                                                                    
                trends['avg_diff_interpolated_augmentation'].append((interpolated-augmentations).abs().sum(dim=(1,2,3)).mean(0).item()
                                                                ) 
                trends['avg_diff_holefilled_augmentation'].append((holefilled_images-augmentations).abs().sum(dim=(1,2,3)).mean(0).item())
                # '''
                if config["out_dir"] is not None:
                    for k in ['interpolated','augmentations','holefilled','saliency','diff_holefilled_augmentation','diff_interpolated_augmentation','inpainting_input']:
                        for ii,item in enumerate(trends[k]):
                            if k not in ['saliency','diff_holefilled_augmentation','diff_interpolated_augmentation']:
                                item = item.transpose(1,2,0)
                            img_save(
                                item, 
                                os.path.join(config['out_dir'],k + '.png') 
                                )
                    
                    for k in ['loss','hole_size','avg_diff_holefilled_augmentation','avg_diff_interpolated_augmentation']:
                        save_plot(trends[k],k,os.path.join(config['out_dir'],f'{k}.png'))
                # assert False
    # import ipdb;ipdb.set_trace()    
    print('TODO: what will this return? saliency?')                
if __name__ == '__main__':
    # original_imname = 'images/372.png'; imagenet_target=372
    # original_imname = 'images/ILSVRC2012_val_00000013.JPEG'; imagenet_target=370
    # original_imname = 'database/balloons.png'
    # original_imname = 'database/volacano.png';imagenet_target = None
    # print('is it class 6 for voc?')
    original_imname = 'images/cars.png'; imagenet_target = 751#829#751#6#
    # original_imname = 'images/vulture.jpeg'; imagenet_target = 23
    # original_imname = 'images/n01443537_16.JPEG'; imagenet_target = 1
    # original_imname = 'images/data/feature_inversion/building.jpg'; imagenet_target = 538
    im = skimage.io.imread(original_imname)
    if im.max() > 1:
        im = im/255.
    if im.ndim == 3 and im.shape[-1] > 3:
        im = im[...,:-1]
    target_size = 256
    overshoot = min(im.shape[:2])/target_size
    resize_aspect_ratio = 1/overshoot
    print(im.shape)

    im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))


    config = {
        #---------------------------------------------
        'target_class' : imagenet_target,
        'datasetname':'imagenet',
        'modelname':'resnet50',
        #---------------------------------------------
        'out_dir':'gpnn-saliency-output',
        'iters':10,
        # 'iters':1,#10
        'coarse_dim':14,#
        # 'coarse_dim':28,
        # 'coarse_dim':100,#
        'out_size':0,
        'patch_size':7,
        # 'patch_size':15,
        'stride':1,
        'pyramid_ratio':4/3,
        # 'pyramid_ratio':2,
        'faiss':True,
        # 'faiss':False,
        'no_cuda':False,
        #---------------------------------------------
        'in':None,
        'sigma':4*0.75,
        # 'sigma':0.3*0.75,
        'alpha':0.005,
        'task':'random_sample',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':im,
        'batch_size':10,
        #---------------------------------------------
        'implementation':'gpnn',#'efficient-gpnn','gpnn'
        'init_from':'zeros',#'zeros','target'
        'keys_type':'single-resolution',#'multi-resolution','single-resolution'
        #---------------------------------------------
        'use_pca':True,
        'n_pca_components':30,
        #---------------------------------------------
        'patch_aggregation':'uniform',#'uniform','distance-weighted','median'
        }    
    print('TODO:check if config is correct for hole filling')
    run(config=config)
