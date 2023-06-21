# ==============================
from main import part_discovery
# from main_latest import part_discovery
import numpy as np
import torch
import pickle
import os
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def scale_point_to_original_size(CY,CX,saliency_size,original_size):
    rel_CY,rel_CX = (CY*1./saliency_size[0],CX*1./saliency_size[1])
    full_sized_CY,full_sized_CX = rel_CY*original_size[0],rel_CX*original_size[1]
    return (full_sized_CY,full_sized_CX)
def get_center(patch_centers,patch_importances,cnn_input_size,original_image_size_WH):
    # assert isinstance(patch_importances,np.ndarray)
    # assert isinstance(patch_centers,np.ndarray)
    assert patch_centers.ndim == 2
    assert patch_centers.shape[-1] == 2
    if patch_importances.ndim == 2:
        patch_importances = patch_importances[:,12]
        print('hard coding patch_importances in get_center to 12th class')
    i_most_important = np.argsort(patch_importances)[-1]
    assert (patch_importances[i_most_important] >= patch_importances).all()
    y,x = patch_centers[i_most_important]
    original_image_W, original_image_H = original_image_size_WH
    yx1 = scale_point_to_original_size(
                    y,x,
                    cnn_input_size,
                    (original_image_H,original_image_W)
                    )

    xy1 = [yx1[1],yx1[0]]
    return xy1


def main_torchray(
                model=None,
                ref=None,
                target_id=None,
                OPTIONS=None,
                class_name=None,
                save_dir=None,
                experiment_executor=None,
                original_image_size_WH=None):  
    TODO = gpnn_saliency(TODO)
    '''
    trends,patch_theta0,importances = part_discovery(
        OPTIONS['patch_options'],
        OPTIONS['jigsaw_options'],
        ref,model,target_id,
        OPTIONS['niter1'],OPTIONS['niter2'],OPTIONS['lr'],OPTIONS['pixel_jitter'],
        # OPTIONS['patch_importance_momentum'],
        OPTIONS['patch_correction_mode'],
        OPTIONS['evaluation_options'],
        # OPTIONS['use_occlude'],
        OPTIONS['loss_options'],
        OPTIONS['importance_options'],
        OPTIONS['non_lin'],
        OPTIONS['patch_lim_mode'],
        experiment_type = OPTIONS['experiment_type'],
        importance_type = OPTIONS['importance_type'],
        importance_mode = OPTIONS['importance_mode'],        
        # filter_options = OPTIONS['filter_options'],
        device=OPTIONS['device']
        )
    info = dict(
        # gt = {'class_id':class_id,'label':y[0],'dataset':self.experiment.dataset},
        # scaled_point = pointYX_scaled_
        trends=trends,
        patch_theta0 = tensor_to_numpy(patch_theta0),
        importances = tensor_to_numpy(importances.importances),
        importance_std = tensor_to_numpy(importances.std),
        importance_noise_logit = tensor_to_numpy(importances.noise_logit),
        )
    #------------------------------------------------------------
    from model import patch_centers_,get_patch_theta
    patch_theta = get_patch_theta(
        patch_theta0,
        ref.shape[-2:],
        OPTIONS['patch_options']['span'],
        non_lin = OPTIONS['non_lin'],
        lim_mode='full')
    patch_centers = patch_centers_(
        patch_theta,
        ref.shape[-2:],
        OPTIONS['patch_options']['span'])
    info['point'] = get_center(
        tensor_to_numpy(patch_centers),
        tensor_to_numpy(importances.importances),ref.shape[-2:],original_image_size_WH)
    '''
    info['point'] = get_center(TODO,TODO,ref.shape[-2:],original_image_size_WH)
    # visualize.save_for_pointing_game(im_save_dir,info,self.OPTIONS,to_server=True)
    info['OPTIONS'] = OPTIONS
    from cnn import denormalize_vgg
    info['ref'] = tensor_to_numpy(ref)
    try:
        os.system(f'rm -rf {save_dir}')
    except FileNotFoundError:
        pass
    os.mkdir(save_dir)
    with open(os.path.join(save_dir,'info.pkl'),'wb') as f:
        pickle.dump(info,f)
    #------------------------------------------------------------
    if True:
        import gc;gc.collect()
        torch.cuda.empty_cache()        
    return info
