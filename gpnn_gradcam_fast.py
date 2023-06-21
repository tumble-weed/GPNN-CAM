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
import colorful
from saliency_for_gpnn_gradcam import get_saliency,permute_using_I
if os.environ.get('USE_GPNN_FAST',False) == '1':
    from model.my_gpnn_fast  import extract_patches,combine_patches    
    from model.my_gpnn_fast import gpnn
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
from saliency_for_gpnn_gradcam import get_cams
from torch.nn.functional import unfold
def toggled_img_save(*args,**kwargs):
    if False:
        img_save(*args,**kwargs)
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
        cam0,_,_ = get_cams(ref[:,:3],imagenet_target,method=config['base_method'],cnn=cnn)
        cam0 = torch.tensor(cam0).unsqueeze(1).to(device)
        more_returns['cam0'] = tensor_to_numpy(cam0)[0,0]
        print('set cam0 in more_returns as non-tensor')
        # print('TODO:see if the correct class is predicted');import ipdb;ipdb.set_trace()
        # config['input_img'] = config['input_img']* tensor_to_numpy(cam0[0,0][...,None]);print('setting input_image to masked')
        # debug.cam0 = torch.ones_like(cam0);print('setting cam0 to ones')
        # add initial image to the augmentations
        

    # n_super_iters = config['n_super_iters']
    n_super_iters  = 2
    config['batch_size'] = 100
    import copy
    running_saliency_dict = {'mass_sum':0,'avg_saliency':0}
    
    for i in range(n_super_iters):
        config_i = copy.deepcopy(config)
        config_i['out_dir']  = config_i['out_dir'] + str(i)
        if  i ==0:
            cam0_ = cam0[0,0]
            cam0_ = cam0_/cam0_.max()
            # import ipdb;ipdb.set_trace()
            toggled_img_save(tensor_to_numpy(cam0_), os.path.join(config_i['out_dir'],'unpermuted_cams','original.png' ))
            #import pudb;pudb.set_trace()
        if i > 0:
            if False:
                previous_saliency_dict = saliency_dict
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
                running_mass_sum = running_saliency_dict['mass_sum']
                previous_mass_sum = previous_saliency_dict['mass_sum']
                running_avg_saliency = running_saliency_dict['avg_saliency']
                total_mass = running_mass_sum + previous_mass_sum
                # import ipdb;ipdb.set_trace()
                running_saliency_dict['avg_saliency'] =  (running_avg_saliency * running_mass_sum + previous_avg_saliency * previous_mass_sum )/(total_mass + (total_mass==0).astype(np.float32))
                running_saliency_dict['mass_sum'] = total_mass 
                if np.isnan(running_saliency_dict['avg_saliency']).any() or np.isinf(running_saliency_dict['avg_saliency']).any():
                    #import pdb;pdb.set_trace()
                    print(colorful.red('ignoring error in saliency'))
                    # import time;time.sleep(5)
                    
                if False and 'using running saliency':
                    config_i['input_img']  = np.concatenate([config_i['input_img'],running_saliency_dict['avg_saliency'][0,0][...,None]],axis=-1)
                
            else:
                print(colorful.red_on_grey('using gradient descent for avg_saliency'))
                S = running_saliency_dict['avg_saliency']
                if not isinstance(S,torch.Tensor):
                    S = (cam0).clone().detach().requires_grad_(True)
                    running_saliency_dict['avg_saliency'] = S
                    optS = torch.optim.Adam([S],lr=1e-1)
                    running_saliency_dict['optS'] = optS
                optS = running_saliency_dict['optS']
                from saliency_for_gpnn_gradcam import permute_using_I
                



                cams = torch.tensor(saliency_dict['raw_saliency'],device=device)
                trendss = defaultdict(list)
                for inneri in range(2):
                    Sp = permute_using_I(torch.tile(S,( config['batch_size'],1,1,1,)),aggregation_results['I'],

                                            S.shape[-2:]+(1,),patch_size=(config['patch_size'],config['patch_size']),

                                            patch_aggregation = aggregation_results['patch_aggregation'],

                                            stride=1,W=aggregation_results['weights'])
                    loss = (cams - Sp)**2
                    loss = loss.mean(dim=0).sum()
                    optS.zero_grad()
                    loss.backward()
                    optS.step()
                    trendss['loss'].append(loss.item())
                # _ = running_saliency_dict['avg_saliency'].data.copy_(S.data)
                for ii,Spi in enumerate(Sp):
                    dutils.img_save(tensor_to_numpy(Spi)[0],f'Sp{ii}.png')

            break
            # print(colored('weighing saliency by 10','red'))
        # import ipdb;ipdb.set_trace()
        augmentations,aggregation_results,saliency_dict,model =   augmentation_attribution(config=config_i,cnn=cnn)
        # print('TODO:visualize augmentations,saliency after 1 iter');import ipdb;ipdb.set_trace()
        if i > 0:
            # remove the saliency layer
            augmentations = augmentations[:,:3]
        #==============================================
        
        if save_results:
            # save
            # augmentations

            for ii,aug in enumerate(augmentations):
                img_save(tensor_to_numpy(aug.permute(1,2,0)), model.out_file[:-len('.png')] + str(ii) + '.png' )
            # continue
            #==============================================
            if True:
                def save_graph(y,title,filename):
                    plt.figure()
                    plt.plot(y)
                    plt.title(title)
                    plt.draw()
                    plt.savefig(filename)
                    plt.close()
                def save_im(y,title,filename):
                    plt.figure()
                    plt.imshow(y)
                    plt.colorbar()
                    plt.title(title)
                    plt.draw()
                    plt.savefig(filename)
                    plt.close()   
                # for ii in reversed(range(11)):
                #     save_graph(model.trends['D',ii],'D',
                #             os.path.join(config_i['out_dir'],f'D{ii}.png'))
                #     save_graph(model.trends['change_in_I',ii],'change_in_I',
                #             os.path.join(config_i['out_dir'],f'change_in_I{ii}.png'))
                #     save_graph(model.trends['diversity',ii],'diversity',
                #             os.path.join(config_i['out_dir'],f'diversity{ii}.png'))
                #     save_graph(model.trends['change_in_image',ii],'change_in_image',
                #             os.path.join(config_i['out_dir'],f'change_in_image{ii}.png'))
                #     # import pdb;pdb.set_trace()
                '''
                distances_ = tensor_to_numpy(debug.distances)
                distances_ = distances_/distances_.max()
                save_im(distances_,'distances')
                # plt.figure()
                # plt.imshow(distances_.reshape((augmentations.shape[-2] - 6,augmentations.shape[-1]  - 6 )))
                # plt.colorbar()
                # plt.draw()
                # plt.savefig('distances.png')
                '''
            #==============================================
            # cams
            cams = saliency_dict['cams']
            print('see the shape of cams')
            #import pdb;pdb.set_trace()
            assert cams.ndim == 4,'cams.ndim == 4'
            assert cams.shape[1] == 1
            for ii,ci in enumerate(cams):
                img_save(ci[0], os.path.join(config_i['out_dir'],'cam',str(ii) + '.png' ))
            #==============================================
            #unpermuted cams
            unpermuted_cams = saliency_dict['saliency']
            print('see the shape of cams')
            assert unpermuted_cams.ndim == 4,'cams.ndim == 4'
            assert unpermuted_cams.shape[1] == 1
            for ii,ci in enumerate(unpermuted_cams):
                img_save(ci[0], os.path.join(config_i['out_dir'],'unpermuted_cams', str(ii) + '.png' ))
                
                
            #====================================================================
            # std
            std_cam_ = saliency_dict['std_saliency']
            std_cam_ = std_cam_ / (std_cam_.max())
            assert std_cam_.ndim == 4
            assert std_cam_.shape[1] == 1
            print('see the shape of std')
            print('not saving std')
            if True:
                img_save(std_cam_[0,0], os.path.join(config_i['out_dir'],'unpermuted_cams','std' + '.png' ))

            avg_saliency_ = saliency_dict['avg_saliency']
            assert avg_saliency_.ndim == 4
            assert avg_saliency_.shape[1] == 1
            img_save(avg_saliency_[0,0], os.path.join(config_i['out_dir'],'unpermuted_cams', 'avg_saliency' + '.png' ))
            if i>0:
                running_avg_saliency_ = running_saliency_dict['avg_saliency']
                normalized = running_avg_saliency_/running_avg_saliency_.max()
                img_save(normalized[0,0], os.path.join(config_i['out_dir'],'unpermuted_cams', 'running_avg_saliency' + '.png' ))
            
            

            if False:
                img_save(tensor_to_numpy(sampling), 
                    os.path.join(os.path.join(config_i['out_dir'],'output','sampling.png') ))
            #%%
            print('TODO!!: how about using unnormalized maps from gradcam?')
            print('TODO!!: median filter in aggregating?')
            print('TODO!!: median filter on optical flow (only on the final iteration)?')
            print('TODO!!: not using probs to aggregate?')
            # import IPython;IPython.embed()`
    import ipdb;ipdb.set_trace()
    # import sys;sys.exit()
    return running_saliency_dict['avg_saliency']
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
