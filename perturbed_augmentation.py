from elp_perturbation import get_masked_input,PRESERVE_VARIANT
from model.my_gpnn import extract_patches
from aggregation import combine_patches
import torch
import copy
from model.my_gpnn import gpnn
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
def unperturb(ref,gpnn_perturbed,augmentations_perturbed,aggregation_results_perturbed):

    sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
    D = aggregation_results_perturbed['D']
    
    D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
    I = aggregation_results_perturbed['I']
    I = I.reshape(augmentations_perturbed.shape[0],I.shape[0]//augmentations_perturbed.shape[0],1)
    augmentation_unperturbed = []
    # for v in sharp_patches.unsqueeze(0):
    for i in range(augmentations_perturbed.shape[0]):
        
        augmentation_unperturbed.append(combine_patches(sharp_patches[I[i].squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, augmentations_perturbed.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[i],I=I[i])['combined'])

    augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)    
    return augmentation_unperturbed

def perturb_augment_unperturb(ref,saliency,config):
    perturbed_ref = get_masked_input(
                    ref,
                    torch.tensor(saliency).to(ref.device),
                    num_levels=8,
                    variant=PRESERVE_VARIANT)
    config_blurred = copy.deepcopy(config)
    config_blurred['input_img'] = tensor_to_numpy(perturbed_ref.permute(0,2,3,1)[0])
    gpnn_perturbed = gpnn(config_blurred)
    augmentations_perturbed,aggregation_results_perturbed = gpnn_perturbed.run(to_save=False)
    augmentations_unperturbed = unperturb(ref,gpnn_perturbed,augmentations_perturbed,aggregation_results_perturbed)
    return augmentations_perturbed,augmentations_unperturbed,aggregation_results_perturbed,gpnn_perturbed
    