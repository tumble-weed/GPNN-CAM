#! /opt/conda/bin/python
#%%
import os
import numpy as np
# os.chdir('/root/evaluate-saliency-5/GPNN')
#%%
# faiss doesnt work without gpu
# !python random_sample.py -in database/balloons.png --faiss
import skimage.io
from matplotlib import pyplot as plt
from model.utils import Timer

from saliency_for_gpnn_gradcam import get_saliency,permute_using_I
from model.my_gpnn  import extract_patches,combine_patches
import torch
import gradcam
from model.utils import *
import skimage.io
import skimage.transform
from torch.nn.functional import fold,unfold
from model.my_gpnn import gpnn
import debug
from termcolor import colored
from saliency_for_gpnn_gradcam import get_cams
from torch.nn.functional import unfold
def augmentation_attribution(config=None,cnn=None):
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    model = gpnn(config)
    #========================================================
    ref = config['input_img']
    patch_size = config['patch_size']
    stride = config['stride']
    if not isinstance(ref,torch.Tensor):
        ref = torch.tensor(ref,device=device).float().unsqueeze(0).permute(0,3,1,2)
    else:
        assert ref.ndim == 4
        assert ref.shape[1] == 3,'only tested for 3 channel inputs'
    imagenet_target = config['imagenet_target']
    #========================================================
    with Timer('model run'):
        augmentations,aggregation_results = model.run(to_save=False)
    augmentations = augmentations[:,:3]
    ref = ref[:,:3]
    # ref = config['input_img']
    # patch_size = config['patch_size']
    #========================================================
    #========================================================
    # get initial gradcam
    if False and 'original image':
        from saliency import get_cams
        cam0,_,_ = get_cams(ref[:,:3],imagenet_target,method='gradcam')
        cam0 = torch.tensor(cam0).unsqueeze(1).to(device)
        # config['input_img'] = config['input_img']* tensor_to_numpy(cam0[0,0][...,None]);print('setting input_image to masked')
        # debug.cam0 = torch.ones_like(cam0);print('setting cam0 to ones')
        from torch.nn.functional import unfold
        cam0u = unfold(cam0, kernel_size=(patch_size,patch_size), dilation=(1, 1), stride=stride, padding=(0, 0))
        debug.cam0 = cam0u
        # add initial image to the augmentations
        ref = cam0 + torch.zeros_like(ref)
        augmentations = torch.cat([augmentations,ref],dim=0)
        
    torch.cuda.empty_cache()
    # output_im = skimage.io.imread(output_imname)
    saliency_dict = {}
    # import pdb;pdb.set_trace()
    if False and 'original image':
        for k in ['weights','I']:
            if k in aggregation_results:
                v = aggregation_results[k]
                if k == 'weights':
                    # new_v = torch.ones_like(v[:1])
                    new_v  = torch.ones_like(v[:1]) * (1/49.)
                elif k == 'I':
                    new_v = torch.arange( 
                                                        np.prod(
                                                            [
                                                            ref.shape[-2] - 2*(patch_size//2),
                                                            ref.shape[-1] - 2*(patch_size//2)
                                                            ]
                                                        ),device=device)
                    new_v = new_v.flatten()[:,None]
                                                        
                aggregation_results[k] = torch.cat([v,new_v],dim=0)
    # import pdb;pdb.set_trace()
    from saliency_for_scorecam import get_saliency
    saliency_dict = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method=config['base_method'],cnn=cnn)
    return augmentations,aggregation_results,saliency_dict,model
def main(config=None,cnn=None,save_results = False,more_returns={}):
    
    # for d in ['output','camoutput','unpermuted_camsoutput','maskoutput']:
    #     os.system(f'rm -rf {d}')
    # print(colored('brute force deleting gpnn-gradcam'))
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    os.system(f'rm -rf {config["out_dir"]}')
    if True and 'original image':
        ref = config['input_img']
        imagenet_target = config['imagenet_target']
        if not isinstance(ref,torch.Tensor):
            ref = torch.tensor(ref,device=device).float().unsqueeze(0).permute(0,3,1,2)
        else:
            # assert all([ref.ndim == 4, ref.shape[-1] == 3])
            # ref = ref.permute(0,3,1,2)
            
            assert all([ref.ndim == 4, ref.shape[1] == 3]) 
        cam0,_,_ = get_cams(ref[:,:3],imagenet_target,method='gradcam',cnn=cnn)
        cam0 = torch.tensor(cam0).unsqueeze(1).to(device)
        more_returns['cam0'] = tensor_to_numpy(cam0)[0,0]
        print('set cam0 in more_returns as non-tensor')
        # print('TODO:see if the correct class is predicted');import ipdb;ipdb.set_trace()
        # config['input_img'] = config['input_img']* tensor_to_numpy(cam0[0,0][...,None]);print('setting input_image to masked')
        # debug.cam0 = torch.ones_like(cam0);print('setting cam0 to ones')
        # add initial image to the augmentations
    #===============================================================================
    # HFLIP   
    HFLIP = False
    NOISE = True
    n_noisy = 10
    augmentations = ref
    import ipdb;ipdb.set_trace()
    flip_indicator = torch.zeros(augmentations.shape[0],device=device).bool()
    if HFLIP:
        naug_prev = augmentations.shape[0]
        augmentations = torch.cat([augmentations,augmentations.flip(2)],dim=0)
        new_flip_indicator = torch.zeros(augmentations.shape[0])
        new_flip_indicator[naug_prev:] = 1
        flip_indicator = new_flip_indicator
    if NOISE:
        # NOISE
        augmentations = augmentations.unsqueeze(1) + 0.1*torch.randn(1,n_noisy,*ref.shape[1:],device=device)
        flip_indicator = flip_indicator.unsqueeze(1) * torch.ones((1,n_noisy),device=device)
        flip_indicator = flip_indicator.bool()
        augmentations = augmentations.flatten(start_dim=0,end_dim=1)
        flip_indicator = flip_indicator.flatten()
        assert flip_indicator.shape[0] == augmentations.shape[0]
        #============================================================================
    cams_,_,_ = get_cams(augmentations,imagenet_target,method='gradcam',cnn=cnn)
    cams = torch.tensor(cams_).unsqueeze(1).to(device)
    cams[flip_indicator] = cams[flip_indicator].flip(2)
    avg_saliency = cams.mean(dim=0,keepdim=True)
    #dutils.img_save(tensor_to_numpy(avg_saliency)[0,0],'trivial_saliency.png')

    return avg_saliency
if __name__ == '__main__':    
    if True:
        # from model.gpnn import gpnn

        # original_imname = 'images/372.png'; imagenet_target=372
        # original_imname = 'images/ILSVRC2012_val_00000013.JPEG'; imagenet_target=370
        # original_imname = 'database/balloons.png'
        # original_imname = 'database/volacano.png';imagenet_target = None
        # print('is it class 6 for voc?')
        original_imname = 'images/cars.png'; imagenet_target = 751#829#751#6#
        # original_imname = 'images/vulture.jpeg'; imagenet_target = 23
        # original_imname = 'images/n01443537_16.JPEG'; imagenet_target = 1
        # original_imname = 'images/data/feature_inversion/building.jpg'; imagenet_target = 538

        output_imname = os.path.join('output',os.path.basename(original_imname))
        output_imname_root,ext = output_imname.split('.')
        output_imname = output_imname_root + '_random_sample' +'.png'
        original_im = skimage.io.imread(original_imname)
        # !python random_sample.py -in  {original_imname} --faiss
        # assert False
        im = skimage.io.imread(original_imname)
        if im.ndim == 3 and im.shape[-1] > 3:
            im = im[...,:-1]
        target_size = 256
        overshoot = min(im.shape[:2])/target_size
        resize_aspect_ratio = 1/overshoot
        print(im.shape)

            
        im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))
        # im = im[-256:,-256:];print('forcibly cropping image')
        print(im.shape)    
        
        # print(colored('brute force making gpnn-gradcam, change in utils?','red'))
        # import os;os.makedirs('gpnn-gradcam')
        config = {
        'out_dir':'gpnn-gradcam/output',
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
        'patch_aggregation':'distance-weighted',#'uniform','distance-weighted','median'
        'imagenet_target':imagenet_target,
        'n_super_iters': 10,
        }    
    main(config=config,save_results = True)
