from gpnn_gradcam import main as gpnn_gradcam_main
import torch
def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    return out

def run_gpnn_gradcam(model,ref,target_id,device=None):
    # return
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
        }    
    avg_saliency = gpnn_gradcam_main(config,cnn=model)
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'gpnn-gradcam',
                        }
    
# saliency_data =run_gpnn_gradcam(model,ref,target_id,device=device)
# print('bye')
