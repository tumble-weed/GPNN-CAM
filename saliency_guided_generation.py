#! /opt/conda/bin/python
#%%
import register_ipdb
import os
import numpy as np

# os.chdir('/root/evaluate-saliency-5/GPNN')
#%%
# faiss doesnt work without gpu
# !python random_sample.py -in database/balloons.png --faiss
import skimage.io
from matplotlib import pyplot as plt
from model.utils import Timer

# from saliency_for_gpnn_gradcam import get_saliency,permute_using_I
# from model.my_gpnn_for_guided  import extract_patches,combine_patches
import torch
import gradcam
from model.utils import *
import skimage.io
import skimage.transform
from torch.nn.functional import fold,unfold
from model.my_gpnn_for_guided import gpnn
import debug
from termcolor import colored
# from saliency_for_gpnn_gradcam import get_cams
from torch.nn.functional import unfold
from benchmark import settings
import dutils
import ipdb
import glob
import pickle
import colorful
import lzma
from collections import defaultdict
IMAGENET_ROOT = '/root/bigfiles/dataset/imagenet'
PASCAL_ROOT = '/root/bigfiles/dataset/VOCdevkit/VOC2007/'
global global_stats
global_stats = defaultdict(lambda :defaultdict(int))
# global_stats['n'] = 0
def augmentation_attribution(config=None,cnn=None,stats=None):
    global global_stats
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
    torch.cuda.reset_peak_memory_stats()
    with Timer('model run') as model_timer:
        augmentations_4chan,aggregation_results = model.run(to_save=False,stats=stats)
    if False:
        stats.append(model_timer.elapsed)
        memory_stats = torch.cuda.memory_stats()
        peak_bytes_requirement = memory_stats["allocated_bytes.all.peak"]
    for scale_and_iter in stats:
        for metric in stats[scale_and_iter]:
            global_stats[scale_and_iter][metric] += stats[scale_and_iter][metric]
        global_stats[scale_and_iter]['n'] += 1
    # import ipdb; ipdb.set_trace()
    dutils.cipdb('DBG_MEMORY')
    augmentations = augmentations_4chan
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
    #saliency_dict = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method=config['base_method'],cnn=cnn)
    return augmentations,aggregation_results,model
def main(config=None,cnn=None,save_results = False,more_returns={},use_saliency=True,stats=None):
    
    # for d in ['output','camoutput','unpermuted_camsoutput','maskoutput']:
    #     os.system(f'rm -rf {d}')
    # print(colored('brute force deleting gpnn-gradcam'))
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    os.system(f'rm -rf {config["out_dir"]}')
    if use_saliency:
        config['input_img']  = np.concatenate([config['input_img'],config['saliency_factor']*config['saliency'][0,...,None]],axis=-1)
    else:
        config['input_img']  = np.concatenate([config['input_img'],0*config['saliency'][0,...,None]],axis=-1)
    # cam0 = config['saliency']
    # import ipdb;ipdb.set_trace()
    
    augmentations_4chan,aggregation_results,model =   augmentation_attribution(config=config,cnn=cnn,stats=stats)
    # print('TODO:visualize augmentations,saliency after 1 iter');import ipdb;ipdb.set_trace()
    # remove the saliency layer
    augmentations = augmentations_4chan[:,:3]
    augmentations_saliency = augmentations_4chan[:,-1]
    # import ipdb;ipdb.set_trace()
    return augmentations,augmentations_saliency
# if __name__ == '__main__':    
def _1_image(im,saliency,target_id,USE_SALIENCY,aggregation,stats=None):
        #===================================================================
        print("TODO: merge this with main")

        print(im.shape)    

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
        'patch_aggregation':aggregation,#'uniform','distance-weighted','median'
        'imagenet_target':target_id,
        'n_super_iters': 10,
        #---------------------------------------------
        'saliency':saliency,
        'index_type':'ivf',
        #---------------------------------------------
        'saliency_factor': 2,
        }    
        augmentations, augmentations_saliency = main(config=config,save_results = True,use_saliency=USE_SALIENCY,stats=stats)
        return augmentations, augmentations_saliency

        
def run_multiple_images(dataset="imagenet",modelname = 'vgg16',device = 'cuda',methodname = 'gradcam',USE_SALIENCY=True,N_IMAGES = 100,aggregation=None,out_dir0=None,save_image=False,stats=None):

    global global_stats
    # results_dir = settings.RESULTS_DIR_librecam
    # results_dir = os.path.join(os.path.basename(settings.RESULTS_DIR_librecam),'saliency-guided')
    os.makedirs(out_dir0,exist_ok=True)
    print('TODO:move out_dir0 to outside?')
    #==================================================
    # load model
    
    def get_model(modelname):
        print('TODO:move me to outside')
        import libre_cam_models.relevance.vgg
        import libre_cam_models.relevance.resnet

        if modelname == 'vgg16':
            model = libre_cam_models.relevance.vgg.vgg16_bn(pretrained = True).to(device)
        else:
            model = libre_cam_models.relevance.resnet.resnet50(pretrained = True).to(device)        
        return model
    model = get_model(modelname)
    model.eval()        
    # ipdb.set_trace()
    ##############################################################################################
    # image_paths = sorted(glob.glob(os.path.join(imagenet_root,'images','val','*.JPEG')))
    # size=(224,224)
    # image_paths = list((image_paths))
    # # import pdb;pdb.set_trace()
    # if end is not None:
    #     image_paths = image_paths[:end]
    # image_paths = image_paths[start:]
    # print(start,end)
    # assert False
    ##############################################################################################   
    
    image_paths = list(sorted(glob.glob(
                os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodname}-{modelname}','*/')
            )))
    assert len(image_paths)
    ##############################################################################################       
    avg_prob_across_dataset = []
    dataset_results = []
    
    for i,impath in enumerate(image_paths):
        # import ipdb; ipdb.set_trace()
        # impaths
        # 2: lassie
        # 3:cup
        # 4: baby
        # 5: snake
        # 6: hedgehog
        # 7: macroon
        # 8: mousetrap
        """
        if i < 2:
            print(colorful.red("coontinuing for impath < 2"))
            continue
        """ 
        if i > N_IMAGES:
            break
        imroot = os.path.basename(impath.rstrip(os.path.sep))
        #############################################        
        if dataset  == 'imagenet':
            im = skimage.io.imread(os.path.join(IMAGENET_ROOT,'images','val',imroot+'.JPEG'))
        elif dataset  == 'pascal':
            im = skimage.io.imread(os.path.join(PASCAL_ROOT,'JPEGImages',imroot+'.jpg'))
        if im.ndim == 2:
            
            print(colorful.red('continuing on 2d image'))
            continue
        #############################################################################
        out_dir = os.path.join(out_dir0,f'{dataset}-{methodname}-{modelname}-{imroot}')
        os.makedirs(out_dir,exist_ok=True)
        methoddir =  os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodname}-{modelname}')
        im_results_dir = os.path.join(methoddir,imroot)
        
        try:
            pklnames = glob.glob(os.path.join(im_results_dir,'*.pkl'))
            assert len(pklnames) == 1
            pklname = pklnames[0]
            
            with open(pklname,'rb') as f:
                loaded = pickle.load(f)
        except AssertionError as e:
            pklnames = glob.glob(os.path.join(im_results_dir,'*.xz'))
            assert len(pklnames) == 1
            pklname = pklnames[0]
            with lzma.open(pklname,'rb') as f:
                loaded = pickle.load(f)            
            

        saliency = loaded['saliency']
        if saliency.ndim==4 and saliency.shape[:2] == (1,1):
            saliency = saliency[:,0,...]
        M = saliency.max()
        M = M + float((M ==0))
        saliency = saliency/M
        # saliency = 1 - saliency
        saliency = 1 * saliency
        target_id = loaded['target_id']
            # ref = loaded['ref']
        #assert ref.min() >0 and ref.max() < 1
        #===================================================================
        im = skimage.transform.resize(im,saliency.shape[1:])    
        # ipdb.set_trace()    
        #############################################################################
    
        import imagenet_localization_parser
        from benchmark.synset_utils import get_synset_id,synset_id_to_imagenet_class_ix
        if dataset == 'imagenet':
            bbox_info = imagenet_localization_parser.get_voc_label(
            root_dir = os.path.join(IMAGENET_ROOT,'bboxes','val'),
            x = imroot)
            synset_id = bbox_info['annotation']['object'][0]['name']
            target_id = synset_id_to_imagenet_class_ix(synset_id)
            import imagenet_synsets
            classname = imagenet_synsets.synsets[target_id]['label']
            classname = classname.split(',')[0]
            if False:
                classname = '_'.join(classname)
            else:
                classname = classname.replace(' ','_')     
        elif dataset == 'pascal':
            from benchmark.voc_utils import class_names
            bbox_info = imagenet_localization_parser.get_voc_label(
            root_dir = os.path.join(PASCAL_ROOT,'Annotations'),
            x = imroot)
            classnames = [ obji['name'] for obji in (bbox_info['annotation']['object'])]
            # classname = bbox_info['annotation']['object'][0]['name']
            
            
            target_ids = [class_names.index(classname) for classname in classnames]
            classnames = [classname.replace(' ','_') for classname in classnames]
            target_id = target_ids[0]
            classname = classnames[0]
    #     print(bbox_info)
        
    #     print(synset_id)
        
    #     print(target_id)

        ##############################################################################################               
        if isinstance(stats,dict):
            # dataset="imagenet",modelname = 'vgg16',device = 'cuda',methodname = 'gradcam',USE_SALIENCY=True,N_IMAGES = 100,aggregation=None,out_dir0=None,save_image=False,stats=None
            stats[impath,USE_SALIENCY,aggregation] = {}
        augmentations, augmentations_saliency = _1_image(im,saliency,target_id,USE_SALIENCY=USE_SALIENCY,aggregation=aggregation,stats=stats[impath,USE_SALIENCY,aggregation])
        #==================================================
        # ipdb.set_trace()
        # from cnn import get_vgg_transform
        # vgg_transform = get_vgg_transform(size)
        # if im_.ndim == 2:
        #     im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
        # im_pil = Image.fromarray(im_)
        # ref = vgg_transform(im_pil).unsqueeze(0)
        # return ref        
        def normalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                        vgg_std=[0.229, 0.224, 0.225]):
            device = t.device
            out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
            return out            
        # import ipdb;ipdb.set_trace()
        ref = normalize_tensor(augmentations,vgg_mean=[0.485, 0.456, 0.406],
                        vgg_std=[0.229, 0.224, 0.225])
        ref = ref.contiguous()
        scores = model(ref)
        probs = torch.softmax(scores,dim=1)
        prob = probs[:,target_id]
        score = scores[:,target_id]
        avg_prob_across_dataset.append(tensor_to_numpy(prob))
        print(prob.mean())
        #==================================================
        original_im_tensor = torch.tensor(im,device=device).permute(2,0,1).unsqueeze(0).float()
        original_im_tensor = normalize_tensor(original_im_tensor)
        original_scores = model(original_im_tensor)
        original_probs = torch.softmax(original_scores,dim=1)
        original_prob = original_probs[:,target_id]
        original_score = original_scores[:,target_id]        
        #==================================================
        increase_in_prob = (prob - original_prob).clamp(min=0)
        relative_increase_in_prob = increase_in_prob/original_prob
        
        decrease_in_prob = ( original_prob - prob).clamp(min=0)        
        relative_decrease_in_prob = decrease_in_prob/original_prob
        
        results = {
            'prob':tensor_to_numpy(prob),'score':tensor_to_numpy(score),
            'original_prob':tensor_to_numpy(original_prob),'original_score':tensor_to_numpy(original_score),
            'increase_in_prob':tensor_to_numpy(increase_in_prob),
            'relative_increase_in_prob':tensor_to_numpy(relative_increase_in_prob),
            'decrease_in_prob':tensor_to_numpy(decrease_in_prob),
            'relative_decrease_in_prob':tensor_to_numpy(relative_decrease_in_prob),
                   }
        dataset_results.append(results)
        if save_image:
            with open(os.path.join(out_dir,f'results.pkl'),'wb') as f:
                pickle.dump(results,f)
            # import ipdb;ipdb.set_trace()
            for i in range(augmentations.shape[0]):
                dutils.cipdb('DBG_SAVE_IMAGE')
                dutils.img_save(tensor_to_numpy(augmentations[i].permute(1,2,0)),os.path.join(out_dir,f'im{i}.png'))
                dutils.img_save(tensor_to_numpy(augmentations_saliency[i]),os.path.join(out_dir,f'saliency{i}.png'))  
    
    #------------------------------------
    if False:
        dataset_results = calculate_dataset_avg(dataset_results)
        
    # avg_prob_across_dataset = np.mean(avg_prob_across_dataset)
    # print(avg_prob_across_dataset)
    # import time;time.sleep(4)
    # return avg_prob_across_dataset
    return dataset_results
    #------------------------------------
def calculate_dataset_avg(dataset_results):
    avg_prob_dataset =  np.mean([el['prob'] for el in  dataset_results])
    avg_increase_in_prob_dataset = np.mean([el['increase_in_prob'] for el in  dataset_results])
    
    avg_decrease_in_prob_dataset = np.mean([el['decrease_in_prob'] for el in  dataset_results])
    
    avg_relative_increase_in_prob_dataset = np.mean([el['relative_increase_in_prob'] for el in  dataset_results])
    avg_relative_decrease_in_prob_dataset = np.mean([el['relative_decrease_in_prob'] for el in  dataset_results])
    avg_increase_indicator = np.mean([el['increase_indicator'] for el in  dataset_results])
    avg_decrease_indicator = np.mean([el['decrease_indicator'] for el in  dataset_results])
    avg_original_prob = np.mean([el['original_prob'] for el in  dataset_results])
    dataset_results = dict( 
        avg_prob_dataset = avg_prob_dataset,
        avg_increase_in_prob_dataset = 
        avg_increase_in_prob_dataset,
        avg_decrease_in_prob_dataset = avg_decrease_in_prob_dataset,
        avg_relative_increase_in_prob_dataset = avg_relative_increase_in_prob_dataset,
        avg_relative_decrease_in_prob_dataset = avg_relative_decrease_in_prob_dataset,
        avg_original_prob = avg_original_prob,
        avg_increase_indicator = avg_increase_indicator,
        avg_decrease_indicator = avg_decrease_indicator,
        )
    return dataset_results

def aggregate_time_and_memory(stats):
    out = {}
    # n = stats['n']
    for scale_and_iter in stats:
        if scale_and_iter == 'n':
            continue
        out[scale_and_iter] = {}
        n = stats[scale_and_iter]['n']
        for metric in stats[scale_and_iter]:
            if metric == 'n':
                out[scale_and_iter][metric] = stats[scale_and_iter][metric]
                continue
            gross = stats[scale_and_iter][metric]
            avg = gross/n
            out[scale_and_iter][metric] = avg
    return out
#run_multiple_images()
def run_1_setting(save_image=True,NIMAGES = 1000,
         aggregations = ['distance-weighted','uniform'],
         USE_SALIENCYs = [True,False],
         root_out_dir = f'/root/bigfiles/other/gpnn-gradcam/saliency_guided_generation/',
         dataset= 'imagenet'):
    
    avg_probs = []
    results = []
    
    for aggregation in aggregations:
        for USE_SALIENCY in USE_SALIENCYs:
            worm = f'{"with_saliency" if USE_SALIENCY else "without_saliency"}-{aggregation}'
            out_dir0 = os.path.join(root_out_dir,worm)       
            stats = {}     
            dutils.cipdb('DBG_SAVE_IMAGE')
            dataset_results = run_multiple_images(dataset=dataset,modelname = 'vgg16',device = 'cuda' if torch.cuda.is_available() else 'cpu',methodname = 'gpnn-gradcam',USE_SALIENCY=USE_SALIENCY,N_IMAGES = NIMAGES,aggregation=aggregation,
                                       out_dir0=out_dir0,save_image=save_image,stats=stats)
            # avg_probs.append(avg_prob_across_dataset)
            
            results.append(dataset_results)
            avg_global_stats = aggregate_time_and_memory(global_stats)
            print(colorful.red("print out avg_global_stats"))
            import ipdb; ipdb.set_trace()
                
    # import ipdb; ipdb.set_trace()
    if save_image:
        with open(os.path.join(root_out_dir,f'guided-results.pkl'),'wb') as f:
            pickle.dump(results,f)
    print(results)
def collect_results(method_dirs):
    # method_dir = os.path.join(root_out_dir,
    all_method_results = []
    for method_dir in method_dirs:
        method_results = []
        for d in glob.glob(os.path.join(method_dir,'*/')):
            if not os.path.exists(os.path.join(d,'results.pkl')):
                continue
            with open(os.path.join(d,f'results.pkl'),'rb') as f:
                results = pickle.load(f)
                prob = results['prob']
                original_prob = results['original_prob']
                # import ipdb;ipdb.set_trace()
                del results['relaive_decrease_in_prob']
                del results['score']
                # import ipdb;ipdb.set_trace()
                results.update({ 
                           'increase_in_prob':np.clip(prob-original_prob,0,None),
                           'decrease_in_prob':np.clip(original_prob-prob,0,None),
                           'relative_increase_in_prob':np.clip(prob-original_prob,0,None)/((1 - original_prob) + float(original_prob == 1)),
                           'relative_decrease_in_prob':np.clip(original_prob-prob,0,None)/(original_prob + float(original_prob == 0)),
                           'increase_indicator':((prob-original_prob) >= 0).mean(),
                            'decrease_indicator': ((prob-original_prob) < 0).mean(),
                })
                assert all([(v >= 0).all() for v in results.values()])
            method_results.append(results)
        method_results = calculate_dataset_avg(method_results)
        all_method_results.append(method_results)
    print(all_method_results)
        
def main_guided():
    # assert False, 'add parameters to run_1_setting by looking at main_timing'
    root_out_dir = os.path.join(f'/root/bigfiles/other/gpnn-gradcam/saliency_guided_generation/')
    # aggregations = ['distance-weighted','uniform']
    # USE_SALIENCYs = [True,False]
    aggregations = ['distance-weighted']
    USE_SALIENCYs = [False]    
    dataset = 'pascal'
    run_1_setting(
        dataset = dataset,
        save_image=True,
         NIMAGES = 50,
         aggregations = aggregations,
         USE_SALIENCYs = USE_SALIENCYs,
         root_out_dir = root_out_dir
         )    
    out_dirs = []
    for aggregation in aggregations:
        for USE_SALIENCY in USE_SALIENCYs:
            worm = f'{"with_saliency" if USE_SALIENCY else "without_saliency"}-{aggregation}'
            out_dir0 = os.path.join(root_out_dir,worm)            
            out_dirs.append(out_dir0)
    collect_results(out_dirs)        
    if False:
        run_1_setting()    
    if False:
        dataset = 'imagenet'
        IMAGENET_ROOT = '/root/bigfiles/dataset/imagenet'
        root_out_dir = os.path.join(f'/root/bigfiles/other/gpnn-gradcam/saliency_guided_generation/{dataset}')
        NIMAGES = 1000
        avg_probs = []
        results = []
        out_dirs = []
        for aggregation in ['distance-weighted','uniform']:
            for USE_SALIENCY in [True,False]:
                worm = f'{"with_saliency" if USE_SALIENCY else "without_saliency"}-{aggregation}'
                out_dir0 = os.path.join(f'/root/bigfiles/other/gpnn-gradcam/saliency_guided_generation/',worm)            
                out_dirs.append(out_dir0)
        collect_results(out_dirs)
def main_timing():
    root_out_dir = os.path.join(f'/root/bigfiles/other/gpnn-gradcam/timing-study')
    aggregations = ['distance-weighted']
    USE_SALIENCYs = [False]
    run_1_setting(save_image=True,
         NIMAGES = 50,
         aggregations = aggregations,
         USE_SALIENCYs = USE_SALIENCYs,
         root_out_dir = root_out_dir
         )    
    # IMAGENET_ROOT = '/root/bigfiles/dataset/imagenet'
    avg_probs = []
    results = []
    out_dirs = []
    for aggregation in aggregations:
        for USE_SALIENCY in USE_SALIENCYs:
            worm = f'{"with_saliency" if USE_SALIENCY else "without_saliency"}-{aggregation}'
            out_dir0 = os.path.join(root_out_dir,worm)            
            out_dirs.append(out_dir0)
    collect_results(out_dirs)    
def main_noisy():
    os.environ['ADD_SCALE_NOISE'] = '1'
    root_out_dir = os.path.join(f'/root/bigfiles/other/gpnn-gradcam/noise-study')
    aggregations = ['distance-weighted']
    USE_SALIENCYs = [False]
    run_1_setting(save_image=True,
         NIMAGES = 50,
         aggregations = aggregations,
         USE_SALIENCYs = USE_SALIENCYs,
         root_out_dir = root_out_dir
         )    
    IMAGENET_ROOT = '/root/bigfiles/dataset/imagenet'
    avg_probs = []
    results = []
    out_dirs = []
    for aggregation in aggregations:
        for USE_SALIENCY in USE_SALIENCYs:
            worm = f'{"with_saliency" if USE_SALIENCY else "without_saliency"}-{aggregation}'
            out_dir0 = os.path.join(root_out_dir,worm)            
            out_dirs.append(out_dir0)
    collect_results(out_dirs)    
    
if __name__ == '__main__':
    main_guided()
    # main_timing()
    # main_noisy()
    '''
    
    [
        {'avg_prob_dataset': 0.62634146, 'avg_increase_in_prob_dataset': 0.04637048, 'avg_decrease_in_prob_dataset': 0.06869661, 'avg_relative_increase_in_prob_dataset': 0.2592189, 'avg_relative_decrease_in_prob_dataset': 0.16668226, 'avg_original_prob': 0.64866745}, 
        {'avg_prob_dataset': 0.53425455, 'avg_increase_in_prob_dataset': 0.027831146, 'avg_decrease_in_prob_dataset': 0.14224415, 'avg_relative_increase_in_prob_dataset': 0.13625854, 'avg_relative_decrease_in_prob_dataset': 0.27131695, 'avg_original_prob': 0.64866745}, 
        
        {'avg_prob_dataset': 0.59000486, 'avg_increase_in_prob_dataset': 0.03982725, 'avg_decrease_in_prob_dataset': 0.0984899, 'avg_relative_increase_in_prob_dataset': 0.18280624, 'avg_relative_decrease_in_prob_dataset': 0.20620432, 'avg_original_prob': 0.64866745}, 
        
        {'avg_prob_dataset': 0.49457595, 'avg_increase_in_prob_dataset': 0.025972309, 'avg_decrease_in_prob_dataset': 0.18006386, 'avg_relative_increase_in_prob_dataset': 0.113460764, 'avg_relative_decrease_in_prob_dataset': 0.3233722, 'avg_original_prob': 0.64866745}]

[
    {'avg_prob_dataset': 0.62634146, 'avg_increase_in_prob_dataset': 0.04637048, 'avg_decrease_in_prob_dataset': 0.06869661, 'avg_relative_increase_in_prob_dataset': 0.2592189, 'avg_relative_decrease_in_prob_dataset': 0.16668226, 'avg_original_prob': 0.64866745, 'avg_increase_indicator': 0.514649033570702, 'avg_decrease_indicator': 0.4853509664292981}, 
    {'avg_prob_dataset': 0.53425455, 'avg_increase_in_prob_dataset': 0.027831146, 'avg_decrease_in_prob_dataset': 0.14224415, 'avg_relative_increase_in_prob_dataset': 0.13625854, 'avg_relative_decrease_in_prob_dataset': 0.27131695, 'avg_original_prob': 0.64866745, 'avg_increase_indicator': 0.33865717192268563, 'avg_decrease_indicator': 0.6613428280773143}, 
    {'avg_prob_dataset': 0.59000486, 'avg_increase_in_prob_dataset': 0.03982725, 'avg_decrease_in_prob_dataset': 0.0984899, 'avg_relative_increase_in_prob_dataset': 0.18280624, 'avg_relative_decrease_in_prob_dataset': 0.20620432, 'avg_original_prob': 0.64866745, 'avg_increase_indicator': 0.4091556459816887, 'avg_decrease_indicator': 0.5908443540183113}, 
    {'avg_prob_dataset': 0.49457595, 'avg_increase_in_prob_dataset': 0.025972309, 'avg_decrease_in_prob_dataset': 0.18006386, 'avg_relative_increase_in_prob_dataset': 0.113460764, 'avg_relative_decrease_in_prob_dataset': 0.3233722, 'avg_original_prob': 0.64866745, 'avg_increase_indicator': 0.29196337741607326, 'avg_decrease_indicator': 0.7080366225839267}]

    
    A 2x 2 table in latex describing the results
    \begin{table}[h]
    \centering
    \begin{tabular}{|c|c|c|}
    \hline
    & \textbf{With Saliency} & \textbf{Without Saliency} \\ \hline
    \textbf{Distance Weighted} & 0.62634146 & 0.53425455 \\ \hline
    \textbf{Uniform} & 0.59000486 & 0.49457595 \\ \hline
    \end{tabular}
    \caption{Measuring the effect of saliency guided generation on the probability of the generated image. The table shows the average probability of the generated image, for the 2 aggreation schemes: distance-weighted and uniform, and for the 2 genration schemes: with and without using the saliency channel.Saliency Guided Generation with distance-weighted aggregation scheme retains more objectness than others. For comparison the  average probability of the original dataset is 0.64866745. }
    \label{tab:label}
    \end{table}
    
    A 2x 2x3 table in latex describing the results.make this table be half column
    \begin{table}[h]
    \centering
    \begin{tabular}{|c|c|c|c|c|c|c|}
    \hline
    & \multicolumn{3}{c|}{\textbf{With Saliency}} & \multicolumn{3}{c|}{\textbf{Without Saliency}} \\ \hline
    & \textbf{Avg Probability} & \textbf{Avg Increase Indicator} & \textbf{Avg Decrease Indicator} & \textbf{Avg Probability} & \textbf{Avg Increase Indicator} & \textbf{Avg Decrease Indicator} \\ \hline
    \textbf{Distance Weighted} & 0.62634146 & 0.514649033570702 & 0.4853509664292981 & 0.53425455 & 0.33865717192268563 & 0.6613428280773143 \\ \hline
    \textbf{Uniform} & 0.59000486 &  0.4091556459816887 & 0.5908443540183113 & 0.49457595 & 0.29196337741607326 & 0.7080366225839267 \\ \hline
    \end{tabular}
    \caption{{Measuring the effect of saliency guided generation on the objectness of the generated image. The table shows the average probability of the generated image& and whether it was larger or smaller than the original image. Compared are the 2 aggreation schemes: distance-weighted and uniform, and for the 2 genration schemes: with and without using the saliency channel.Saliency Guided Generation with distance-weighted aggregation scheme retains more objectness than others. For comparison the  average probability of the original dataset is 0.64866745.  }
    \label{tab:label}
    \end{table}
    
    make a table be of width of 1 column of a 2 column page
    \begin{table}[h]
    \centering
    \begin{tabular}{|c|c|c|c|c|c|c|}
    \hline
    \
    '''
    
# %%

