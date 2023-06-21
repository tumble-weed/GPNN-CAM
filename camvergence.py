#%%
# import os
# os.chdir('/root/evaluate-saliency-4/jigsaw')    
# python -m camvergence --modelname vgg16 --methodnames gradcam --dataset imagenet --skip false
import register_ipdb
import dutils
import torch
import torchvision
import numpy as np
from benchmark.benchmark_utils import ChangeDir,AddPath
import skimage.io
from PIL import Image
import pickle
import glob
import os
import colorful
# '/root/evaluate-saliency-4/jigsaw/imagenet'


#from pydoc import importfile
#%% gradcam
from benchmark.benchmark_utils import create_im_save_dir
import pascal_localization_parser
from benchmark.voc_utils import class_names
import os
import pickle
from benchmark import settings

import dutils
import builtins

builtins.dutils = dutils
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
builtins.tensor_to_numpy = tensor_to_numpy
DISABLED = False
def get_image_tensor(impath,size=(224,),dataset=None):
    im_ = skimage.io.imread(impath)
    if im_.ndim == 2:
        im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
    im_pil = Image.fromarray(im_)
    if dataset == 'imagenet':
        from cnn import get_vgg_transform
        vgg_transform = get_vgg_transform(size)
        ref = vgg_transform(im_pil).unsqueeze(0)
        return ref
    elif dataset in ['pascal','voc']:
        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229, 0.224, 0.225)
        bgr_mean = [103.939, 116.779, 123.68]
        mean = [m / 255. for m in reversed(bgr_mean)]
        std = [1 / 255.] * 3
        
        vgg_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size),
                torchvision.transforms.Normalize(mean=mean,std=std),
                ]
            )
        ref = vgg_transform(im_pil).unsqueeze(0)
        # import ipdb;ipdb.set_trace()
        return ref
    else:
        assert False
        
##############################################################################################
##############################################################################################
# GPNN-GRADCAM

from trivial_gradcam import main as trivial_gradcam_main
from hacky_scorecam import main as hacky_scorecam_main
from hacky_relevancecam import main as hacky_relevancecam_main
import torch
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
def denormalize_tensor(t,dataset= None):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    if dataset == 'imagenet':
        vgg_mean=[0.485, 0.456, 0.406]
        vgg_std=[0.229, 0.224, 0.225]
        out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
        return out
    elif dataset in ['pascal','voc']:
        # pass
        bgr_mean = [103.939, 116.779, 123.68]
        mean = [m / 255. for m in reversed(bgr_mean)]
        std = [1 / 255.] * 3
        out = (t * torch.tensor(std).to(device)[None,:,None,None]) + torch.tensor(mean).to(device)[None,:,None,None]
        return out
    else:
        assert False

def run_relevance_cam(model,ref,target_id,base_method='gradcam',device=None):
    # return

    print(ref.__class__)
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='relevancecam',gradcam_scale_cams=None,cnn=model)    
    print('TODO:save original image, cam saliency and avg saliency')
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'relevancecam',
                        #'cam0':cam0,
                        'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
                        }
    return saliency_data

def run_libra_cam(model,ref,target_id,base_method='gradcam',device=None):
    # return
    ref = denormalize_tensor(ref)
    print(ref.__class__)
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='libracam',gradcam_scale_cams=None,cnn=model)    
    print('TODO:save original image, cam saliency and avg saliency')
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'relevancecam',
                        #'cam0':cam0,
                        'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
                        }
    return saliency_data
def run_score_cam(model,ref,target_id,base_method='gradcam',device=None):
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='scorecam',gradcam_scale_cams=None,cnn=model)
    print('TODO:save original image, cam saliency and avg saliency')
    return {
            'saliency':avg_saliency,
            'target_id':target_id,
            'method':'scorecam',
            #'cam0':cam0,
            'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
            }

def run_gradcam(model,ref,target_id,base_method='gradcam',device=None):
    # import ipdb;ipdb.set_trace()
    '''
    for vgg16, the max pools are at:
    13,23,33,43
    '''
    arg_target_layer = 43
    if 'vgg' in str(model.__class__):
        arg_target_layer = 33
    
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='gradcam',gradcam_scale_cams=None,cnn=model,arg_target_layer=arg_target_layer)
    if True:
        from matplotlib import cm
        from collections import defaultdict
        masked = ref
        sequence= []
        n_iter = 50
        ecc = 0.5
        trends = defaultdict(list)
        for i in range(n_iter):
            avg_saliency,scores,probs = get_cams(masked,target_id,method='gradcam',gradcam_scale_cams=None,cnn=model,arg_target_layer=arg_target_layer)
            
            mask = torch.tensor(avg_saliency,device = ref.device).unsqueeze(1)
            assert mask.min() >= 0
            m = mask.min()
            # m = m + (m==0).float()
            M = mask.max()
            denom = (M-m)
            denom = denom + (denom==0).float()
            mask = (mask -m)/denom
            sequence.append(tensor_to_numpy(mask)[0,0])
            
            #print(mask.shape)
            soft_mask = mask*ecc + (1-ecc)*torch.ones_like(mask)
            masked = soft_mask * ref        
            if i > 0:
                # calculate spearman correlation between this and previous
                # shape of avg_saliency is (1,224,224)
                # previous entry in sequence is of shape (1,224,224)
                from scipy.stats import spearmanr
                corr = spearmanr(avg_saliency.flatten(),sequence[-2].flatten())
                trends['corr'].append(corr.correlation)
                print(f'corr {corr.correlation}')
            mask_ =tensor_to_numpy(mask)[0,0]
            mask_ = cm.jet(mask_)
            
            mask_avg_ = np.stack(sequence,axis=0)
            mask_avg_ = mask_avg_ * (0.9 ** np.arange(len(sequence))[::-1,None,None])
            mask_avg_ = mask_avg_.sum(axis=0)
            mask_avg_ = mask_avg_/mask_avg_.max()
            mask_avg_ = cm.jet(mask_avg_)
            dutils.img_save(mask_,f'camvergence{i}.png')
            dutils.img_save(mask_avg_,f'camvergence_avg{i}.png')
            dutils.img_save(tensor_to_numpy(masked.permute(0,2,3,1))[0],f'camvergence_masked{i+1}.png')    
            
    import ipdb;ipdb.set_trace()
    print('TODO:save original image, cam saliency and avg saliency')
    return {
            'saliency':avg_saliency,
            'target_id':target_id,
            'method':'gradcam',
            #'cam0':cam0,
            'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
            }
def run_gradcampp(model,ref,target_id,base_method='gradcam',device=None):
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='gradcampp',gradcam_scale_cams=None,cnn=model)
    assert not np.isnan(avg_saliency).any()
    print('TODO:save original image, cam saliency and avg saliency')
    return {
            'saliency':avg_saliency,
            'target_id':target_id,
            'method':'scorecam',
            #'cam0':cam0,
            'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
            }
def run_trivial_cam(model,ref,target_id,base_method='gradcam',device=None):
    # return
    assert False, 'this will actually run score_cam, correct this'
    ref = denormalize_tensor(ref)
    print(ref.__class__)
    import time;time.sleep(5)
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
        'input_img':ref,
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
        'imagenet_target':target_id,
        'n_super_iters':10,
        #---------------------------------------------
        'device':device,        
        'index_type':'ivf',
        'base_method':base_method,
        }    
    more_returns = {}
    # avg_saliency = trivial_gradcam_main(config,cnn=model,save_results=False,more_returns=more_returns)
    print('hacking trivial cam as hacky_score_cam')
    time.sleep(5)
    avg_saliency = hacky_scorecam_main(config,cnn=model,save_results=False,more_returns=more_returns)
    print('TODO:save original image, cam saliency and avg saliency')
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'trivial-gradcam',
                        #'cam0':cam0,
                        'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
                        }
    return saliency_data
def run_gpnn_gradcam(model,ref,target_id,dataset,base_method='gradcam',device=None):
    #=========================================================
    if os.environ.get('USE_GPNN_GRADCAM_FAST',False):
        # assert False
        if base_method == 'allcam':
            assert False,'wont work for allcam'
        from gpnn_gradcam_fast import main as gpnn_gradcam_main
    else:
        if base_method == 'allcam':
            from gpnn_gradcam_multi import main as gpnn_gradcam_main
        else:
            from gpnn_gradcam import main as gpnn_gradcam_main
    #=========================================================
    # return
    ref = denormalize_tensor(ref,dataset=dataset)
    print(ref.__class__)
    # import time;time.sleep(5)
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
        'input_img':ref,
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
        'imagenet_target':target_id,
        'n_super_iters':10,
        #---------------------------------------------
        'device':device,        
        'index_type':'ivf',
        'base_method':base_method,
        }    
    more_returns = {}
    
    multi_saliency = gpnn_gradcam_main(config,cnn=model,save_results=False,more_returns=more_returns)
    # import ipdb;ipdb.set_trace()
    out = {}
    for k,inner in multi_saliency.items():
        
        out[k] = {
            'saliency':inner['avg_saliency'],
            'target_id':target_id,
            'method':f'gpnn-{k}',
            }    
    return out

##############################################################################################
##############################################################################################
def main(    
    methodnames = None,
    skip=False,
    start = 0,
    end=None,
    images_common_to=None,
    
    modelname = 'vgg16',
    device = 'cuda',
    other_info={},
    dataset = None,
    skip_incorrect = False,
    ):

    print('TODO:shift this to pascal_root')
    pascal_root = "/root/bigfiles/dataset/VOCdevkit"
    imagenet_root = IMAGENET_ROOT 
    
    '''
    def get_model(modelname):
        print('TODO:move me to outside')
        print(colorful.red('mocking imagenet instead of pascal'))
        import time;time.sleep(2)
        import libre_cam_models.relevance.vgg
        import libre_cam_models.relevance.resnet

        if modelname == 'vgg16':
            model = libre_cam_models.relevance.vgg.vgg16_bn(pretrained = True).cuda()
        else:
            model = libre_cam_models.relevance.resnet.resnet50(pretrained = True).cuda()        
        return model
    '''
    def get_model(dataset,modelname,is_relevancecam,device=None):
        
        assert modelname in ['vgg16','resnet50','dummymodel'],f'{modelname} not recognized'
        rprop = False
        convert_to_fully_convolutional=True
        if is_relevancecam:
            # convert_to_fully_convolutional=False
            rprop = True
        # print('TODO:fix hardcode of voc')
        if modelname == 'dummymodel':
            class MockObject():
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # self.return_value = self                
                def __getattr__(self, name):
                    return MockObject()      
                def __call__(self,*args,**kwargs):
                    pass
            return MockObject()
        else:
            if dataset in ['imagenet','']:
                dataset = 'imagenet'
                import libre_cam_models.relevance.vgg
                import libre_cam_models.relevance.resnet
                # import ipdb;ipdb.set_trace()
                if modelname == 'vgg16':
                    model = libre_cam_models.relevance.vgg.vgg16_bn(pretrained = True).to(device)
                else:
                    model = libre_cam_models.relevance.resnet.resnet50(pretrained = True).to(device)                    
            elif dataset in ['voc','pascal']:
                dataset = 'voc'
                from benchmark.architectures import get_model as get_model_
                model = get_model_(
                            arch=modelname,
                            dataset=dataset,
                            rprop = rprop,
                            convert_to_fully_convolutional=convert_to_fully_convolutional,
                            
                        )
            else:
                assert False
            model.to(device)
            model.eval()
            model.dataset = dataset
        return model
    
    has_relevancecam = True if any( [uses_rprop in methodname for methodname in methodnames for uses_rprop in ['relevancecam','allcam']]) else None
    # import ipdb;ipdb.set_trace()
    if True:
        model = get_model(dataset,modelname,has_relevancecam,device=device)
        model.eval()
    else:
        print(colorful.black_on_white('lazy get model'))
    # impaths = ['/root/evaluate-saliency-4/cyclegan results/samoyed1.png'] 
    # class_name = 'samoyed'
    # image_paths = sorted(glob.glob(os.path.join(pascal_root,'images','val','*.JPEG')))
    print(colorful.orange('shift getting imagepaths from voc root to common module'))
    # import ipdb;ipdb.set_trace()
    if dataset in ['voc','pascal']:
        image_paths = sorted(glob.glob(os.path.join(pascal_root,'VOC2007','JPEGImages','*.jpg')))
    elif dataset in ['imagenet','']:
        image_paths = sorted(glob.glob(os.path.join(imagenet_root,'images','val','*.JPEG')))
    else:
        assert False
    size=(224,224)
    image_paths = list((image_paths))
    # import pdb;pdb.set_trace()
    if end is not None:
        image_paths = image_paths[:end]
    image_paths = image_paths[start:]
    print(start,end)
    # assert False
    if len(image_paths) == 0:
        print(colorful.orange('len image_paths is 0'))
        import ipdb;ipdb.set_trace()
        
    
    ##############################################################################################
    # print('before methodnames');
    # import pdb;pdb.set_trace()
    for methodname in methodnames:
        if images_common_to is not None:
            hack_for_im_save_dirs = get_hack_for_im_save_dirs(images_common_to,modelname)
            print('#'*50,'\n','hack_for_im_save_dirs','\n','#'*50) 
            # import pdb;pdb.set_trace()   
            image_paths = hack_for_im_save_dirs(image_paths)                
        for i,impath in enumerate(image_paths):
            
            if os.environ.get('BREAK_COMPETING',None):
                BREAK_COMPETING = int(os.environ['BREAK_COMPETING'])
                if i > BREAK_COMPETING:
                    print(colorful.red(f'breaking out of run_competing_librecam at {i}'))
                    import time;time.sleep(2)
                    break
            imroot = os.path.basename(impath).split('.')[0]
            if os.environ.get('DBG_COMPETING_HARDCODE',False) == '1':
                # imroot = 'ILSVRC2012_val_00002330'
                # imroot = 'ILSVRC2012_val_00000005'
                # imroot = 'ILSVRC2012_val_00002483'
                # imroot = 'ILSVRC2012_val_00000019'
                imroot = 'ILSVRC2012_val_00000036'
                skip_incorrect = False
                impath = os.path.join(os.path.dirname(impath),f'{imroot}.JPEG')
                print(colorful.red_on_blue(f'setting imroot to {imroot},skip_incorrect to {skip_incorrect}'))

            if dataset in ['pascal','voc']:
                bbox_info = pascal_localization_parser.get_voc_label(
                    root_dir = os.path.join(pascal_root,'VOC2007','Annotations'),
                    x = imroot)

                classnames = [ obji['name'] for obji in (bbox_info['annotation']['object'])]
                # classname = bbox_info['annotation']['object'][0]['name']
                
                
                target_ids = [class_names.index(classname) for classname in classnames]
                classnames = [classname.replace(' ','_') for classname in classnames]
                # classname = classnames[0]
                # target_id = target_ids[0]
                
                
                # import ipdb;ipdb.set_trace()
            elif dataset in ['imagenet','']:

                import imagenet_localization_parser
                from benchmark.synset_utils import get_synset_id,synset_id_to_imagenet_class_ix
                bbox_info = imagenet_localization_parser.get_voc_label(
                    root_dir = os.path.join(imagenet_root,'bboxes','val'),
                    x = imroot)
            #     print(bbox_info)
                # import ipdb;ipdb.set_trace()
                synset_id = bbox_info['annotation']['object'][0]['name']
            #     print(synset_id)
                target_id = synset_id_to_imagenet_class_ix(synset_id)
            #     print(target_id)
                import imagenet_synsets
                classname = imagenet_synsets.synsets[target_id]['label']
                classname = classname.split(',')[0]
                if False:
                    classname = '_'.join(classname)
                else:
                    classname = classname.replace(' ','_')                
                # assert False, 'untested for imagenet'
                # import ipdb;ipdb.set_trace()
                target_ids = [target_id]
                classnames = [classname]
                # import ipdb;ipdb.set_trace()
                # bbox_info['annotation']['segmented']
                # bbox_info['annotation']['object'][0].keys()
                # import ipdb;ipdb.set_trace()
                pass
            for target_id,classname in zip(target_ids,classnames):
                # if any([
                    
                #     '00000004' in impath,
                #     '00000015' in impath,
                #     '00000019' in impath,
                #     '00000022' in impath
                # ]):
                #     print(impath,classname,target_id)
                #     continue
                #-------------------------------------------------------------------
                if skip:
                    # assert False,'only skips if integrated-gradients'
                    methodnames = [methodname]
                    if methodname == 'gpnn-allcam':
                        methodnames = ['gpnn-gradcam','gpnn-gradcampp','gpnn-relevancecam']
                    continue_flags = {k:False for k in methodnames}
                    for methodnamei in methodnames:
                        imroot = os.path.split(impath)[-1].split('.')[0]
                        im_save_dir = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodnamei}-{modelname}',imroot)
                        pklname = os.path.join(im_save_dir,f'{classname}{target_id}.pkl')
                        # import pdb;pdb.set_trace()
                        if os.path.exists(pklname):
                            # import pdb;pdb.set_trace()
                            try:
                                with open(pklname,'rb') as f:
                                    loaded = pickle.load(f)
                                    saliency = loaded['saliency']
                                    assert not np.isnan(saliency).any()
                                print(f'{pklname} exists, skipping')
                                continue_flags[methodnamei] = True
                                # continue
                            except EOFError as e:
                                print(f'{pklname} corrupt,overwriting')
                            except AssertionError as e:
                                print(f'nan found{pklname}, overwriting')
                    print(continue_flags)
                    if any(continue_flags.values()):
                        continue
                    # import ipdb;ipdb.set_trace()
                #-------------------------------------------------------------------
                ref = get_image_tensor(impath,size=size,dataset=dataset).to(device)
                #-------------------------------------------------
                if not any([ref.min() < 0,ref.max() > 1]):
                    print(colorful.yellow_on_red('ref is not normalized'))
                    
                if False:
                    dutils.ref = ref
                    model = get_model(dataset,modelname,has_relevancecam,device=device)
                    model.eval()                    
                    print(colorful.black_on_white(f'lazy get model'))
                    
                output = model(ref)
                if output.ndim == 4:
                    output = output.mean(dim=(-1,-2))
                predicted_id = output.argmax(dim=1).item()
                incorrectly_predicted = predicted_id != target_id
                # import ipdb;ipdb.set_trace()
                #-------------------------------------------------
                if skip_incorrect:
                    if incorrectly_predicted:
                        print(f'predicted_id {predicted_id} != target_id {target_id}, skipping')
                        continue
                    # else:
                    #     print(colorful.red('hacky continue'))
                    #     continue
                    
                    

                '''
                from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage,ROADLeastRelevantFirstAverage
                '''
                #######################################################################################
                
                #.......................................................
                if methodname in ['gpnn-gradcam','gpnn-gradcampp','gpnn-relevancecam','gpnn-allcam']:
                    # print('before jigsaw');
                    # import ipdb;ipdb.set_trace()
                    if methodname == 'gpnn-gradcam':
                        base_method = 'gradcam'
                    elif methodname == 'gpnn-gradcampp':
                        base_method = 'gradcampp'
                    elif methodname == 'gpnn-relevancecam':
                        base_method = 'relevancecam'                    
                    elif methodname == 'gpnn-allcam':
                        base_method = "allcam"
                    else:
                        assert False,f'{methodname} not recognized'
                    saliency_data =run_gpnn_gradcam(model,ref,target_id,dataset,base_method=base_method,device=device)
                elif methodname in ['trivial-gradcam','trivial-gradcampp']:
                    if methodname == 'trivial-gradcam':
                        base_method = 'gradcam'
                    elif methodname == 'trivial-gradcampp':
                        base_method = 'gradcampp'
                    saliency_data = run_trivial_cam(model,ref,target_id,base_method=base_method,device=device)
                elif methodname == 'gradcam':
                    saliency_data = run_gradcam(model,ref,target_id,base_method='gradcam',device=device)
                elif methodname == 'gradcampp':
                    saliency_data = run_gradcampp(model,ref,target_id,base_method='gradcampp',device=device)                
                elif methodname == 'scorecam':
                    saliency_data = run_score_cam(model,ref,target_id,base_method='scorecam',device=device)
                elif methodname == 'relevancecam':
                    saliency_data = run_relevance_cam(model,ref,target_id,base_method='relevancecam',device=device)                
                elif methodname == 'libracam':
                    saliency_data = run_libra_cam(model,ref,target_id,base_method='libracam',device=device)                                
                    
                else:
                    assert False,f'{methodname} not recognized'
                #....................................................... 
                # import ipdb;ipdb.set_trace() 
                def save_saliency_data(saliency_data,dataset,methodname,impath,modelname,classname,target_id,incorrectly_predicted):        
                    for k,v in saliency_data.items():
                        if isinstance(v,torch.Tensor):
                            saliency_data[k] = tensor_to_numpy(v)
                    
                    saliency_data.update(
                        dict(
                            modelname = modelname,
                            impath = impath,
                            classname = classname,
                            incorrectly_predicted = incorrectly_predicted
                        )
                    )            
                    im_save_dir = create_im_save_dir(experiment_name=f'{dataset}-{methodname}-{modelname}',root_dir=settings.RESULTS_DIR_librecam,impath=impath)  
                    savename = os.path.join(im_save_dir,f'{classname}{target_id}.pkl')
                    # import ipdb;ipdb.set_trace()
                    with open(savename,'wb') as f:
                        pickle.dump(saliency_data,f)     
                if False:
                    if methodname != 'gpnn-allcam':
                        save_saliency_data(saliency_data,dataset,methodname,impath,modelname,classname,target_id,incorrectly_predicted)
                    else:
                        for k,one_method in saliency_data.items():
                            
                            save_saliency_data(one_method,dataset,saliency_data[k]['method'],impath,modelname,classname,target_id,incorrectly_predicted)
                        
    ##############################################################################################         
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methodnames',nargs='*',default = [  
                    # 'gradcam',
                    # 'smoothgrad',
                    # 'fullgrad',
                    # 'integrated-gradients',
                    # 'gradients',
                    #'inputXgradients',
                    # 'jigsaw-saliency',
                    # 'elp',
                    
                    ])
    parser.add_argument('--skip',type=lambda v: (v.lower()=='true'),default=False)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=None)
    parser.add_argument('--device',type=str,default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--images-common-to',type=str,default=None)
    parser.add_argument('--modelname',type=str,default='vgg16')
    parser.add_argument('--dataset',type=str,default='voc')
    parser.add_argument('--skip_incorrect',type=lambda v: (v.lower()=='true'),default=True)
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    # main(methodnames=args.methodnames,skip=args.skip)    
    DEBUG = False
    if len(args.methodnames) == 0:
        print(colorful.orange('found no methods to run'))
        import ipdb;ipdb.set_trace()
    main(skip=args.skip,start=args.start,end=args.end,device=args.device,
    images_common_to=args.images_common_to,methodnames=args.methodnames,modelname=args.modelname,dataset = args.dataset,skip_incorrect = args.skip_incorrect)
       
