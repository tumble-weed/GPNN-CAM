# IPython log file

get_ipython().run_line_magic('logstart', 'ipython_pertubation.py')
# from elp_perturbation import get_masked_input
# perturbed_ref = get_masked_input(
#                     ref,
#                     saliency_dict['avg_saliency'],
#                     num_levels=8,
#                     variant=PRESERVE_VARIANT)
#                     #variant = None)
from elp_perturbation import get_masked_input,PRESERVE_VARIANT
# perturbed_ref = get_masked_input(
#                     ref,
#                     saliency_dict['avg_saliency'],
#                     num_levels=8,
#                     variant=PRESERVE_VARIANT)
#                     #variant = None)
saliency_dict['avg_saliency'].__class__
perturbed_ref = get_masked_input(
                    ref,
                    torch.tensor(saliency_dict['avg_saliency']).to(ref.device),
                    num_levels=8,
                    variant=PRESERVE_VARIANT)
           import dutils
dutils.img_save(tensor_to_numpy(perturbed_ref.permute(0,2,3,1))[0],'perturbed.png')
# gpnn
# model
# gpnn_perturbed = gpnn(config)
config_blurred = copy.deepcopy(config)
# config_blurred['input_img'] = ref_perturbed
# config_blurred['input_img'] = perturbed_ref
# gpnn_perturbed = gpnn(config_blurred)
config_blurred['input_img'] = tensor_to_numpy(perturbed_ref.permute(0,2,3,1)[0])
gpnn_perturbed = gpnn(config_blurred)
augmentations_perturbed,aggregation_results_perturbed = gpnn_perturbed.run(to_save=False)
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
     1))[0],'perturbed.png')
# print('hello')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')

dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed_augmentation.png')
# dutils.img_save(tensor_to_numpy(perturbed_ref.permute(0,2,3,1))[0],'perturbed.png'aggregation_results_perturbation.keys()
aggregation_results_perturbed.keys()
aggregation_results_perturbed['I'].shape
from model.my_gpnn import extract_patches
# sharp_patches =  extract_patches(ref, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# ref.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# augmentation_unperturbed = torch.stack([combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'] for v in sharp_patches],dim=0)
# augmentation_unperturbed = [combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'] for v in sharp_patches]
# augmentation_unperturbed = []
# for v in sharp_patches:
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
    
# for v in sharp_patches:
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
# for v in sharp_patches:
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
# v.shape
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D,I=I)['combined'])
    
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=I)['combined'])
    
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=aggregation_results_perturbed['I'])['combined'])
    
# distances.shape
# D.shape
# aggregation_results_perturbed['D'].shape
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=aggregation_results_perturbed['I'])['combined'])
    
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentation_unperturbed.shape[0],D.shape[0]//augmentation_unperturbed.shape[0],1,1,1)
# D
# D = D.reshape(augmentation_unperturbed.shape[0],D.shape[0]//augmentation_unperturbed.shape[0],1,1,1)
# D = D.reshape(augmentation_perturbed.shape[0],D.shape[0]//augmentation_perturbed.shape[0],1,1,1)
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D,I=aggregation_results_perturbed['I'])['combined'])
    
# D.shape
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=aggregation_results_perturbed['I'])['combined'])
    
# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_perturbed.permute(0,2,3,1))[0],'unperturbed.png')
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed.permute(0,2,3,1))[0],'unperturbed.png')
# augmentation_unperturbed.shape
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed.png')

# I
# I = aggregation_results_perturbed['I']
# I
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=aggregation_results_perturbed['I'])['combined'])
    
# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed.png')
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
#      1))[1],'perturbed1.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
#      1))[1],'perturbed_augmentation1.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
#      1))[2],'perturbed_augmentation2.png')
# get_ipython().run_cell_magic('', '', '\n')
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=I)['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# sharp_patches.shape
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=I)['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# sharp_patches.shape
# v.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I[0]], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# v[I[0]].shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# v.shape
# D.shape
# D[0].shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# I = I.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# I = I.reshape(augmentations_perturbed.shape[0],I.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# I = I.reshape(augmentations_perturbed.shape[0],I.shape[0]//augmentations_perturbed.shape[0],1)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I[0].squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# get_ipython().run_line_magic('logstop', '')
# get_ipython().run_line_magic('logstart', 'ipython_pertubation.py')
# from elp_perturbation import get_masked_input
# perturbed_ref = get_masked_input(
#                     ref,
#                     saliency_dict['avg_saliency'],
#                     num_levels=8,
#                     variant=PRESERVE_VARIANT)
#                     #variant = None)
# from elp_perturbation import get_masked_input,PRESERVE_VARIANT
# perturbed_ref = get_masked_input(
#                     ref,
#                     saliency_dict['avg_saliency'],
#                     num_levels=8,
#                     variant=PRESERVE_VARIANT)
#                     #variant = None)
# saliency_dict['avg_saliency'].__class__
# perturbed_ref = get_masked_input(
#                     ref,
#                     torch.tensor(saliency_dict['avg_saliency']).to(ref.device),
#                     num_levels=8,
#                     variant=PRESERVE_VARIANT)
#                     #variant = None)
# import dutils
# dutils.img_save(tensor_to_numpy(perturbed_ref.permute(0,2,3,1))[0],'perturbed.png')
# gpnn
# model
# gpnn_perturbed = gpnn(config)
# config_blurred = copy.deepcopy(config)
# config_blurred['input_img'] = ref_perturbed
# config_blurred['input_img'] = perturbed_ref
# gpnn_perturbed = gpnn(config_blurred)
# config_blurred['input_img'] = tensor_to_numpy(perturbed_ref.permute(0,2,3,1)[0])
# gpnn_perturbed = gpnn(config_blurred)
# augmentations_perturbed,aggregation_results_perturbed = gpnn_perturbed.run(to_save=False)
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed_augmentation.png')
# dutils.img_save(tensor_to_numpy(perturbed_ref.permute(0,2,3,1))[0],'perturbed.png')
# aggregation_results_perturbation.keys()
# aggregation_results_perturbed.keys()
# aggregation_results_perturbed['I'].shape
# from model.my_gpnn import extract_patches
# sharp_patches =  extract_patches(ref, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# ref.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# augmentation_unperturbed = torch.stack([combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'] for v in sharp_patches],dim=0)
# augmentation_unperturbed = [combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'] for v in sharp_patches]
# augmentation_unperturbed = []
# for v in sharp_patches:
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
    
# for v in sharp_patches:
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
# for v in sharp_patches:
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
# v.shape
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
    
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D,I=I)['combined'])
    
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=I)['combined'])
    
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=aggregation_results_perturbed['I'])['combined'])
    
# distances.shape
# D.shape
# aggregation_results_perturbed['D'].shape
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=aggregation_results_perturbed['I'])['combined'])
    
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentation_unperturbed.shape[0],D.shape[0]//augmentation_unperturbed.shape[0],1,1,1)
# D
# D = D.reshape(augmentation_unperturbed.shape[0],D.shape[0]//augmentation_unperturbed.shape[0],1,1,1)
# D = D.reshape(augmentation_perturbed.shape[0],D.shape[0]//augmentation_perturbed.shape[0],1,1,1)
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D,I=aggregation_results_perturbed['I'])['combined'])
    
# D.shape
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=aggregation_results_perturbed['I'])['combined'])
    
# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_perturbed.permute(0,2,3,1))[0],'unperturbed.png')
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed.permute(0,2,3,1))[0],'unperturbed.png')
# augmentation_unperturbed.shape
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed.png')
# I
# I = aggregation_results_perturbed['I']
# I
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=aggregation_results_perturbed['I'])['combined'])
    
# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed.png')
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
#      1))[1],'perturbed1.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
#      1))[1],'perturbed_augmentation1.png')
# dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
#      1))[2],'perturbed_augmentation2.png')
# get_ipython().run_cell_magic('', '', '\n')
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=I)['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# sharp_patches.shape
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=I)['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# sharp_patches.shape
# v.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I[0]], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# v[I[0]].shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# v.shape
# D.shape
# D[0].shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# I = I.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape
# sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
# D = aggregation_results_perturbed['D']
# D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# I=aggregation_results_perturbed['I']
# I = I.reshape(augmentations_perturbed.shape[0],I.shape[0]//augmentations_perturbed.shape[0],1,1,1)
# augmentation_unperturbed = []
# for v in sharp_patches.unsqueeze(0):
#     augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

# augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
# dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
# I.shape

sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
I = I.reshape(augmentations_perturbed.shape[0],I.shape[0]//augmentations_perturbed.shape[0],1)
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v[I[0].squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
get_ipython().run_line_magic('logstop', '')
get_ipython().run_line_magic('logstart', 'ipython_pertubation.py append')
hi
get_ipython().run_line_magic('logstop', '')
