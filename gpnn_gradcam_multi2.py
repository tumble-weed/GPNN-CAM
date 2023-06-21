#! /opt/conda/bin/python
#%%
import os
import numpy as np
import dutils
# os.chdir('/root/evaluate-saliency-5/GPNN')
#%%
# faiss doesnt work without gpu
# !python random_sample.py -in database/balloons.png --faiss
import skimage.io
from matplotlib import pyplot as plt
from model.utils import Timer
import colorful
from saliency_for_gpnn_gradcam_multi import get_saliency,get_cams
if os.environ.get('USE_GPNN_FAST',False) == '1':
    from model.my_gpnn_fast  import extract_patches,combine_patches    
    from model.my_gpnn_fast import gpnn
    assert False
elif os.environ.get('USE_GPNN_TIMING',False) == '1':
    from model.my_gpnn_with_timing  import extract_patches,combine_patches    
    from model.my_gpnn_with_timing import gpnn    
else:
    from model.my_gpnn  import extract_patches,combine_patches
    from model.my_gpnn import gpnn
import torch
import gradcam
from model.utils import img_save, tensor_to_numpy

import skimage.transform
from torch.nn.functional import fold,unfold

import debug
from termcolor import colored
# from saliency_for_gpnn_gradcam import get_cams

import blosc2
import importlib
import register_ipdb
importlib.reload(register_ipdb)
from torch.nn.functional import unfold
def toggled_img_save(*args,**kwargs):
    if False:
        img_save(*args,**kwargs)
#=============================================
SAVE_D_I = False
SAVE_AUGMENTATIONS = True
#=============================================
def augmentation_attribution(config=None,cnn=None,cnns=None,dataset='imagenet',running_saliency_dicts=[],ref=None,
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dreloaded2=None,
# Ireloaded=None,
# augmentations_reloaded=None,
# ref_patches=None,
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
loadgen_obj = None,
):
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
    if config['base_method'] in ['gpnn-loadgen'] or config['base_method'].startswith('gpnn-loadgen-mycam'):
        augmentations,aggregation_results = loadgen_obj.get_augmentations_batch()
        if False:
            if SAVE_AUGMENTATIONS:
                augmentations,aggregation_results = loadgen_obj.get_augmentations()
                if False:
                    aggregation_results = {}
                    augmentations = augmentations_reloaded.permute(0,3,1,2)
                    aggregation_results['combined'] = augmentations
                    pass
            elif SAVE_D_I:
                augmentations,aggregation_results = loadgen_obj.get_augmentations(config,cnns)
                if False:
                    from aggregation import combine_patches_multiple_images
                    # Dreloaded = blosc2.load_array(os.path.join(config['blosc_dir'],'saved_gpnn_D.bl2'))            
                    # Ireloaded = blosc2.load_array(os.path.join(config['blosc_dir'],'saved_gpnn_I.bl2'))                                    
                    # Dreloaded = torch.tensor(Dreloaded,device=device).float()
                    # Ireloaded = torch.tensor(Ireloaded,device=device).long()

                    # Dreloaded2 = Dreloaded.reshape(config['batch_size'],Dreloaded.shape[0]//config['batch_size'],1,1,1)
                    # import ipdb; ipdb.set_trace()
                    permuted_patches = [torch.index_select(ref_patches,0,Ireloaded[j].squeeze()).unsqueeze(0) for j in range(config['batch_size'])]
                    permuted_patches = torch.cat(permuted_patches)
                    permuted_patches1 = permuted_patches
                    # assert permuted_patches.shape[0] ==1
                    # permuted_patches1 = permuted_patches[0].reshape(config['batch_size'],permuted_patches.shape[1]//config['batch_size'],permuted_patches.shape[2],permuted_patches.shape[3],permuted_patches.shape[4])
                    """
                    aggregation_results = combine_patches(permuted_patches[0][Dre2.shape[1]*aix :(Dre2.shape[1])*(aix+1) ], 
                                        model.PATCH_SIZE, 
                                        model.STRIDE, 
                                        model.input_img.shape,
                                        as_np=False,
                                        patch_aggregation=model.PATCH_AGGREGATION,
                                        distances=Dre2[aix],
                                        I=Ire2[Dre2.shape[1]*aix :(Dre2.shape[1])*(aix+1) ])
                    """
                    # import ipdb; ipdb.set_trace()
                    Ireloaded = Ireloaded.flatten(start_dim=0,end_dim=1)
                    # import ipdb; ipdb.set_trace()
                    aggregation_results = combine_patches_multiple_images(
                        permuted_patches1, 
                        config['patch_size'], config['stride'], config['input_img'].shape[-2:] + (config['input_img'].shape[1],),
                        as_np=False,
                        patch_aggregation=config['patch_aggregation'],
                        distances_bp111=Dreloaded2,I_bp1=Ireloaded) 
                    # import ipdb; ipdb.set_trace()
                    # augmentations,aggregation_results = 
                    augmentations = aggregation_results['combined']
                    augmentations = augmentations.permute(0,3,1,2)
                    aggregation_results['D'] = Dreloaded2
                    aggregation_results['I'] = Ireloaded
    else:
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

    saliency_dicts = [{} for _ in cnns]
    if config['base_method'] != 'gpnn-onlygen-save':
        # import ipdb; ipdb.set_trace()
        saliency_dicts = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method=config['base_method'],cnns=cnns,dataset=dataset,running_saliencys=running_saliency_dicts,ref=ref)
    # import ipdb; ipdb.set_trace()
    return augmentations,aggregation_results,saliency_dicts,model
def main(config=None,cnn=None,cnns=None,save_results = False,more_returns=[],dataset='imagenet'):
    if config['base_method'] == 'gpnn-loadgen':
        config['base_method'] = 'gpnn-loadgen-mycam'

    
    if 'mycam' in config['base_method']: 
        def use_my_cam_saliency():
            from saliency_for_gpnn_gradcam_mycam import get_saliency,get_cams
            globals()['get_saliency'] = get_saliency
            
            globals()['get_cams'] = get_cams
        use_my_cam_saliency()

    if os.environ.get('DBG_MANY_AUG',False) == '1':
        config['n_super_iters'] = 20

    if config['base_method'] == 'gpnn-onlygen-save':

        config['batch_size'] =  config['batch_size']
        config['n_super_iters'] = config['n_super_iters']
    #====================================================
    if config['base_method'] == 'gpnn-loadgen':
        from loadgen_D_I import GPNN_loadgen_D_I
        loadgen_obj = GPNN_loadgen_D_I(config)
    if config['base_method'].startswith( 'gpnn-loadgen'):
        from loadgen_augmentations import GPNN_loadgen_augmentations
        loadgen_obj = GPNN_loadgen_augmentations(config)
    if config['base_method'] == 'gpnn-onlygen-save':
        if SAVE_AUGMENTATIONS:
            from loadgen_augmentations import GPNN_onlygen_save_augmentations
            savegen_obj = GPNN_onlygen_save_augmentations(config)
        if SAVE_D_I:
            assert False,'not implemented'
    #====================================================
    USE_BACKPROP = os.environ.get('USE_BACKPROP',False) == '1'
    print(colorful.yellow(f"~~~~setting base method to {config['base_method']}~~~~"))
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    os.system(f'rm -rf {config["out_dir"]}')
    if True and 'original image':
        ref = config['input_img']
        imagenet_target = config['imagenet_target']
        if not isinstance(ref,torch.Tensor):
            ref = torch.tensor(ref,device=device).float().unsqueeze(0).permute(0,3,1,2)
        else:
            
            assert all([ref.ndim == 4, ref.shape[1] == 3]) 

        out0_per_model = []
        for cnn in cnns:
            more_returns.append({})
            if False:
                out0 = get_cams(ref[:,:3],imagenet_target,method=config['base_method'],cnn=cnn,dataset=dataset)
                out0_per_model.append(out0)
                for k in out0:
                    cam0 = out0[k][0]
                    cam0 = torch.tensor(cam0).unsqueeze(1).to(device)
                    more_returns[-1][k] = { 'cam0' : tensor_to_numpy(cam0)[0,0]}
            

    n_super_iters = config['n_super_iters']
    import copy
    running_saliency_dicts = [{} for _ in cnns]
    if config['base_method'].startswith('gpnn-loadgen'):
        
        loadgen_obj.load(device)


        
    for i in range(n_super_iters):
            
        config_i = copy.deepcopy(config)
        config_i['out_dir']  = config_i['out_dir'] + str(i)

        if i > 0:
            for running_saliency_dict,saliency_dict in zip(running_saliency_dicts,saliency_dicts):

                if True:
                    # import ipdb;ipdb.set_trace()
                    for k in saliency_dict.keys():
                        saliency_dict_k = saliency_dict[k]
                        if k not in running_saliency_dict:
                            running_saliency_dict[k] = {'mass_sum':0,'avg_saliency':0}
                            
                        running_saliency_dict_k = running_saliency_dict[k]
                        
                        previous_saliency_dict = saliency_dict_k
                        previous_avg_saliency = previous_saliency_dict['avg_saliency']
                        print('see what the input_img is like and how to attach the saliency_dict?')
                        
                        if not isinstance(config_i['input_img'],torch.Tensor):
                            assert config_i['input_img'].ndim == 3
                        
                        try:
                            assert previous_avg_saliency.ndim == 4
                            # assert previous_avg_saliency.max() <= 1
                            assert previous_avg_saliency.min() >= 0
                        except Exception as e:
                            # import ipdb;ipdb.set_trace()
                            print(colorful.red('ignoring error in saliency'))
                            # import time;time.sleep(5)
                            pass
                        # running_saliency_dict['avg_saliency'] = running_saliency_dict
                        try:
                            if not USE_BACKPROP:
                                running_mass_sum = running_saliency_dict_k['mass_sum']
                                previous_mass_sum = previous_saliency_dict['mass_sum']
                                running_avg_saliency = running_saliency_dict_k['avg_saliency']
                                total_mass = running_mass_sum + previous_mass_sum
                                # import ipdb;ipdb.set_trace()
                                running_saliency_dict_k['avg_saliency'] =  (running_avg_saliency * running_mass_sum + previous_avg_saliency * previous_mass_sum )/(total_mass + (total_mass==0).astype(np.float32))
                                running_saliency_dict_k['mass_sum'] = total_mass 
                                if np.isnan(running_saliency_dict_k['avg_saliency']).any() or np.isinf(running_saliency_dict_k['avg_saliency']).any():
                                    #import pdb;pdb.set_trace()
                                    print(colorful.red('ignoring error in saliency'))
                                    # import time;time.sleep(5)
                            else:
                                # running_saliency_dict_k['avg_saliency'] = previous_saliency_dict['running_saliency_backprop']
                                # running_saliency_dict_k['running_saliency_backprop'] = previous_saliency_dict['running_saliency_backprop']
                                pass
                        except KeyError as ek:
                            print(ek)
                            if 'mycam' in config['base_method']:
                                pass
                        except Exception as ee:
                            print(ee)
                            
                            import ipdb; ipdb.set_trace()    
                            raise(ee)
                        if False and 'using running saliency':
                            config_i['input_img']  = np.concatenate([config_i['input_img'],running_saliency_dict['avg_saliency'][0,0][...,None]],axis=-1)
                        

                
        if os.environ.get('USE_GPNN_TIMING',False) == "1":
            print('check what keys are there in running_saliency_dict, and how we can communicate the running_saliency to augmentation_attribution (dutils)')
            import ipdb;ipdb.set_trace()
        if config['base_method'] in ['gpnn-loadgen','gpnn-loadgen-mycam'] or config['base_method'].startswith('gpnn-loadgen-mycam'):
            augmentations,aggregation_results,saliency_dicts,model =   augmentation_attribution(config=config_i,cnns=cnns,dataset=dataset,running_saliency_dicts=running_saliency_dicts,ref=ref,loadgen_obj=loadgen_obj)
        else:
            augmentations,aggregation_results,saliency_dicts,model =   augmentation_attribution(config=config_i,cnns=cnns,dataset=dataset,running_saliency_dicts=running_saliency_dicts,ref=ref)

        if config['base_method'] == 'gpnn-onlygen-save':
            if SAVE_D_I:
                Dnp = aggregation_results['D'].detach().cpu().numpy()
                Inp = aggregation_results['I'].detach().cpu().numpy()
                savegen_obj.add(Dnp,Inp)                
            elif SAVE_AUGMENTATIONS:
                augmentations_np = aggregation_results['combined'].detach().cpu().numpy()
                savegen_obj.add(augmentations_np)
                
        if i > 0:
            # remove the saliency layer
            augmentations = augmentations[:,:3]
        #==============================================
        if True:
            for running_saliency_dict,saliency_dict in zip(running_saliency_dicts,saliency_dicts):
                if False:
                    if os.environ.get('USE_BACKPROP',False) == '1':
                        for ik,k in enumerate(running_saliency_dict):
                            sal = running_saliency_dict[k]['running_saliency_backprop']
                            dutils.img_save(tensor_to_numpy(sal/sal.max())[0],f'Sp{k}.png')
                            sal = saliency_dict[k]['cams']
                            dutils.img_save((sal/sal.max())[0],f'{k}.png')
                        import ipdb; ipdb.set_trace()
                else:
                    if True:
                        # import ipdb;ipdb.set_trace()
                        for k in saliency_dict.keys():
                            saliency_dict_k = saliency_dict[k]
                            if k not in running_saliency_dict:
                                running_saliency_dict[k] = {'mass_sum':0,'avg_saliency':0}
                                
                            running_saliency_dict_k = running_saliency_dict[k]
                            
                            previous_saliency_dict = saliency_dict_k
                            previous_avg_saliency = previous_saliency_dict['avg_saliency']
                            print('see what the input_img is like and how to attach the saliency_dict?')
                            
                            if not isinstance(config_i['input_img'],torch.Tensor):
                                assert config_i['input_img'].ndim == 3
                            
                            try:
                                assert previous_avg_saliency.ndim == 4
                                # assert previous_avg_saliency.max() <= 1
                                assert previous_avg_saliency.min() >= 0
                            except Exception as e:
                                # import ipdb;ipdb.set_trace()
                                print(colorful.red('ignoring error in saliency'))
                                # import time;time.sleep(5)
                                pass
                            # running_saliency_dict['avg_saliency'] = running_saliency_dict
                            try:
                                if not USE_BACKPROP:
                                    running_mass_sum = running_saliency_dict_k['mass_sum']
                                    previous_mass_sum = previous_saliency_dict['mass_sum']
                                    running_avg_saliency = running_saliency_dict_k['avg_saliency']
                                    total_mass = running_mass_sum + previous_mass_sum
                                    # import ipdb;ipdb.set_trace()
                                    running_saliency_dict_k['avg_saliency'] =  (running_avg_saliency * running_mass_sum + previous_avg_saliency * previous_mass_sum )/(total_mass + float(total_mass==0))
                                    running_saliency_dict_k['mass_sum'] = total_mass 
                                    if np.isnan(running_saliency_dict_k['avg_saliency']).any() or np.isinf(running_saliency_dict_k['avg_saliency']).any():
                                        #import pdb;pdb.set_trace()
                                        print(colorful.red('ignoring error in saliency'))
                                        # import time;time.sleep(5)
                                else:
                                    # running_saliency_dict_k['avg_saliency'] = previous_saliency_dict['running_saliency_backprop']
                                    # running_saliency_dict_k['running_saliency_backprop'] = previous_saliency_dict['running_saliency_backprop']
                                    pass
                            except KeyError as ek:
                                print(ek)
                                if 'mycam' in config['base_method']:
                                    pass
                            except Exception as ee:
                                print(ee)
                                
                                import ipdb; ipdb.set_trace()    
                                raise(ee)
                            if False and 'using running saliency':
                                config_i['input_img']  = np.concatenate([config_i['input_img'],running_saliency_dict['avg_saliency'][0,0][...,None]],axis=-1)
                    else:
                        assert False,'not tested with multicam'
                        print(colorful.red_on_grey('using gradient descent for avg_saliency'))
                        S = running_saliency_dict['avg_saliency']
                        if not isinstance(S,torch.Tensor):
                            S = (cam0).clone().detach().requires_grad_(True)
                            running_saliency_dict['avg_saliency'] = S
                            optS = torch.optim.Adam([S],lr=1e-2)
                            running_saliency_dict['optS'] = optS
                        optS = running_saliency_dict['optS']
                        

                
            # print(colored('weighing saliency by 10','red'))
        
        #==============================================

    if config['base_method'] == 'gpnn-onlygen-save':
        savegen_obj.dump()
    dutils.cipdb('STOP_AT_GPNN_RETURN')
    """
    #=========================================================
    def save_method(running_saliency_dicts,method):
        s = running_saliency_dicts[0][method]['avg_saliency']
        dutils.img_save(s/s.max(),f'{method}_saliency.png',cmap='jet')
    save_method(running_saliency_dicts,'gradcam')
    save_method(running_saliency_dicts,'gradcampp')
    save_method(running_saliency_dicts,'relevancecam')
    save_method(running_saliency_dicts,'layercam')
    dutils.img_save(config['input_img'],'ref.png')
    #=========================================================    
    """
    import aniket_utils as au
    au.fpdb()
    
    if os.environ.get('DBG_DIFFICULT_IMAGES',False) == '1':
        dutils.img_save(running_saliency_dicts[0]['gpnn-mycam']['avg_saliency'],'saliency.png')
        dutils.img_save(config['input_img'],'ref.png')
        # import sys;sys.exit()
        # dutils.cipdb('DBG_DIFFICULT_IMAGES')
    #=============================================================================
    if os.environ.get('DBG_EACH',False) == '1':
        dutils.img_save(running_saliency_dicts[0]['gpnn-mycam']['avg_saliency'],'saliency.png')
        dutils.img_save(config['input_img'],'ref.png')        
        #============================================+
        # use torchvision utils to plot the augmentations in a grid
        import torchvision
        augmentations = loadgen_obj.augmentations_reloaded2[0].permute(0,3,1,2)
        im_grid = torchvision.utils.make_grid(augmentations,value_range = (-128,128),nrow= int(np.sqrt(augmentations.shape[0])) )
        dutils.img_save(im_grid,'augmentations.png')
        # print(scores[:,target_id].reshape((int(np.sqrt(augmentations.shape[0])),-1)))
        #============================================+    
        # import ipdb;ipdb.set_trace()
        dutils.cipdb('DBG_EACH')
        #=============================================================================
    return running_saliency_dicts#['avg_saliency']
import snoop
# if os.environ.get('TRACE_STEP_BY_STEP','') != '':
#     import snoop
#     main = snoop(main)
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
