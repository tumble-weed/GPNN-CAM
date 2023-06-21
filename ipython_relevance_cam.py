get_ipython().run_line_magic('logstart', 'ipython_pertubation.py')
from elp_perturbation import get_masked_input
perturbed_ref = get_masked_input(
                    ref,
                    saliency_dict['avg_saliency'],
                    num_levels=8,
                    variant=PRESERVE_VARIANT)
                    #variant = None)
from elp_perturbation import get_masked_input,PRESERVE_VARIANT
perturbed_ref = get_masked_input(
                    ref,
                    saliency_dict['avg_saliency'],
                    num_levels=8,
                    variant=PRESERVE_VARIANT)
                    #variant = None)
saliency_dict['avg_saliency'].__class__
#[Out]# <class 'numpy.ndarray'>
perturbed_ref = get_masked_input(
                    ref,
                    torch.tensor(saliency_dict['avg_saliency']).to(ref.device),
                    num_levels=8,
                    variant=PRESERVE_VARIANT)
                    #variant = None)
import dutils
dutils.img_save(tensor_to_numpy(perturbed_ref.permute(0,2,3,1))[0],'perturbed.png')
gpnn
#[Out]# <class 'model.my_gpnn.gpnn'>
model
#[Out]# <model.my_gpnn.gpnn object at 0x7f37f46161d0>
gpnn_perturbed = gpnn(config)
config_blurred = copy.deepcopy(config)
config_blurred['input_img'] = ref_perturbed
config_blurred['input_img'] = perturbed_ref
gpnn_perturbed = gpnn(config_blurred)
config_blurred['input_img'] = tensor_to_numpy(perturbed_ref.permute(0,2,3,1)[0])
gpnn_perturbed = gpnn(config_blurred)
augmentations_perturbed,aggregation_results_perturbed = gpnn_perturbed.run(to_save=False)
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed.png')
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,1))[0],'perturbed_augmentation.png')
dutils.img_save(tensor_to_numpy(perturbed_ref.permute(0,2,3,1))[0],'perturbed.png')
aggregation_results_perturbation.keys()
aggregation_results_perturbed.keys()
#[Out]# dict_keys(['combined', 'weights', 'patch_aggregation', 'D', 'I'])
aggregation_results_perturbed['I'].shape
#[Out]# torch.Size([1127500, 1])
from model.my_gpnn import extract_patches
sharp_patches =  extract_patches(ref, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
ref.shape
#[Out]# torch.Size([1, 3, 256, 457])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
augmentation_unperturbed = torch.stack([combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'] for v in sharp_patches],dim=0)
augmentation_unperturbed = [combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'] for v in sharp_patches]
augmentation_unperturbed = []
for v in sharp_patches:
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, mask.shape[1:3]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
for v in sharp_patches:
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
for v in sharp_patches:
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
v.shape
#[Out]# torch.Size([3, 7, 7])
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False)['combined'])
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D,I=I)['combined'])
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=I)['combined'])
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=aggregation_results_perturbed['I'])['combined'])
distances.shape
D.shape
aggregation_results_perturbed['D'].shape
#[Out]# torch.Size([1127500, 1])
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=aggregation_results_perturbed['D'],I=aggregation_results_perturbed['I'])['combined'])
D = aggregation_results_perturbed['D']
D = D.reshape(augmentation_unperturbed.shape[0],D.shape[0]//augmentation_unperturbed.shape[0],1,1,1)
D
#[Out]# tensor([[7.6449e-06],
#[Out]#         [8.2739e-06],
#[Out]#         [9.8584e-06],
#[Out]#         ...,
#[Out]#         [2.0181e-13],
#[Out]#         [8.7534e-14],
#[Out]#         [1.7199e-13]])
D = D.reshape(augmentation_unperturbed.shape[0],D.shape[0]//augmentation_unperturbed.shape[0],1,1,1)
D = D.reshape(augmentation_perturbed.shape[0],D.shape[0]//augmentation_perturbed.shape[0],1,1,1)
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D,I=aggregation_results_perturbed['I'])['combined'])
D.shape
#[Out]# torch.Size([10, 112750, 1, 1, 1])
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=aggregation_results_perturbed['I'])['combined'])
augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_perturbed.permute(0,2,3,1))[0],'unperturbed.png')
dutils.img_save(tensor_to_numpy(augmentation_unperturbed.permute(0,2,3,1))[0],'unperturbed.png')
augmentation_unperturbed.shape
#[Out]# torch.Size([1, 256, 457, 3])
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed.png')
I
I = aggregation_results_perturbed['I']
I
#[Out]# tensor([[ 32972],
#[Out]#         [ 32973],
#[Out]#         [ 32974],
#[Out]#         ...,
#[Out]#         [112472],
#[Out]#         [112473],
#[Out]#         [112474]])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=aggregation_results_perturbed['I'])['combined'])
augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed.png')
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
     1))[1],'perturbed1.png')
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
     1))[1],'perturbed_augmentation1.png')
dutils.img_save(tensor_to_numpy(augmentations_perturbed.permute(0,2,3,
     1))[2],'perturbed_augmentation2.png')
get_ipython().run_cell_magic('', '', '\n')
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=I)['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
sharp_patches.shape
#[Out]# torch.Size([112750, 3, 7, 7])
I.shape
#[Out]# torch.Size([1127500, 1])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=I)['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v, gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
sharp_patches.shape
#[Out]# torch.Size([112750, 3, 7, 7])
v.shape
#[Out]# torch.Size([112750, 3, 7, 7])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v[I], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
I.shape
#[Out]# torch.Size([1127500, 1])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v[I[0]], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
v[I[0]].shape
#[Out]# torch.Size([1, 3, 7, 7])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
I.shape
#[Out]# torch.Size([1127500, 1])
v.shape
#[Out]# torch.Size([112750, 3, 7, 7])
D.shape
#[Out]# torch.Size([10, 112750, 1, 1, 1])
D[0].shape
#[Out]# torch.Size([112750, 1, 1, 1])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
I = I.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
I.shape
#[Out]# torch.Size([1127500, 1])
sharp_patches =  extract_patches(ref.permute(0,2,3,1), gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE)
D = aggregation_results_perturbed['D']
D = D.reshape(augmentations_perturbed.shape[0],D.shape[0]//augmentations_perturbed.shape[0],1,1,1)
I=aggregation_results_perturbed['I']
I = I.reshape(augmentations_perturbed.shape[0],I.shape[0]//augmentations_perturbed.shape[0],1,1,1)
augmentation_unperturbed = []
for v in sharp_patches.unsqueeze(0):
    augmentation_unperturbed.append(combine_patches(v[I.squeeze(-1)], gpnn_perturbed.PATCH_SIZE, gpnn_perturbed.STRIDE, ref.shape[-2:]+(3,),patch_aggregation=gpnn_perturbed.PATCH_AGGREGATION,as_np=False,distances=D[0],I=[I])['combined'])

augmentation_unperturbed = torch.stack(augmentation_unperturbed,dim=0)
dutils.img_save(tensor_to_numpy(augmentation_unperturbed[0]),'unperturbed_augmentation.png')
I.shape
#[Out]# torch.Size([10, 112750, 1, 1, 1])
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
hi
get_ipython().run_line_magic('cell', '')
get_ipython().run_cell_magic('cell', '', "print('hi')\n\n")
import multicam2
import multicam2
import multicam2
import multicam2
multicam2.relevance_cam(img:torch.Tensor,model_arch,target_layer,target_class)
multicam2.relevance_cam(ref,'vgg16','layer2',751)
import importlib
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
import multicam2
multicam2.relevance_cam(ref,'vgg16','layer2',751)
multicam2.relevance_cam(ref,'vgg16','2',751)
multicam2.relevance_cam(ref,'resnet50','2',751)
multicam2.relevance_cam(ref,'resnet50','layer2',751)
import relevance_cam_modules.resnet
importlib.reload(relevance_cam_modules.resnet)
#[Out]# <module 'relevance_cam_modules.resnet' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/relevance_cam_modules/resnet.py'>
multicam2.relevance_cam(ref,'resnet50','layer2',751)
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
multicam2.relevance_cam(ref,'resnet50','layer2',751)
import LRP_util
importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
multicam2.relevance_cam(ref,'resnet50','layer2',751)
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
multicam2.relevance_cam(ref,'resnet50','layer2',751)
l
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
rcam = multicam2.relevance_cam(ref,'resnet50','layer2',751)
rcam.shape
#[Out]# (224, 224)
dutils.img_save(rcam,'rcam.png')
importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
rcam,rcam_thresh = multicam2.relevance_cam(ref,'resnet50','layer2',751)
dutils.img_save(rcam_thresh,'rcam.png')
rcam,rcam_thresh = multicam2.relevance_cam(ref,'resnet50','layer2',751)
dutils.img_save(rcam_thresh,'rcam.png')
rcam_thresh.shape
#[Out]# (32, 58, 1)
importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
rcam,rcam_thresh = multicam2.relevance_cam(ref,'resnet50','layer2',751)
importlib.reload(multicam2)
#[Out]# <module 'multicam2' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/multicam2.py'>
rcam,rcam_thresh = multicam2.relevance_cam(ref,'resnet50','layer2',751)
get_ipython().run_line_magic('debug', '')
importlib.reload(multicam2);importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
importlib.reload(multicam2);importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
rcam,rcam_thresh = multicam2.relevance_cam(ref,'resnet50','layer2',751)
dutils.img_save(rcam_thresh,'rcam.png')
importlib.reload(multicam2);importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
importlib.reload(multicam2);importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
rcam,rcam_thresh = multicam2.relevance_cam(ref,'resnet50','layer2',751)
importlib.reload(multicam2);importlib.reload(LRP_util)
#[Out]# <module 'LRP_util' from '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/LRP_util.py'>
dutils.img_save(rcam_thresh,'rcam.png')
dutils.img_save(rcam_thresh*rcam,'rcam.png')
dutils.img_save(rcam,'rcam.png')
