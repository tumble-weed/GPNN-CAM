import numpy as np
import torch
from saliency import get_saliency
import skimage.io

# from model.gpnn import gpnn
original_imname = 'images/372.png'; imagenet_target=372
# original_imname = 'images/ILSVRC2012_val_00000013.JPEG'; imagenet_target=370
# original_imname = 'database/balloons.png'
# original_imname = 'database/volacano.png';imagenet_target = None
# print('is it class 6 for voc?')
# original_imname = 'images/cars.png'; imagenet_target = 829#751#6#
# original_imname = 'images/n01443537_16.JPEG'; imagenet_target = 1
# original_imname = 'images/data/feature_inversion/building.jpg'; imagenet_target = 538


def get_image(original_imname):

    im = skimage.io.imread(original_imname)
    if im.ndim == 3 and im.shape[-1] > 3:
        im = im[...,:-1]
    target_size = 256
    overshoot = min(im.shape[:2])/target_size
    resize_aspect_ratio = 1/overshoot
    print(im.shape)

        
    im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))
    return im    
device = 'cuda'
batch_size = 64
noise_std = 1
print('TODO: wrap this in a function, which takes arguments')
print('TODO: add code for saving the augmentations, cams and unpermuted')
im = get_image(original_imname)
assert im.max() < 255
assert im.ndim == 3
x = torch.tensor(im).permute(2,0,1).unsqueeze(0).to(device)
x = torch.ones((1,3,256,457))
# imagenet_target = 1
patch_size = (7,7)
patch_placement = x.shape[-2] - 2*(patch_size//2) , x.shape[-1] - 2*(patch_size//2)
I = np.prod(patch_placement)

placed = torch.arange(I).reshape(patch_placement)
augmentations = torch.tile(x,(batch_size,1,1,1))
I = torch.tile(I.unsqueeze(0),(batch_size,1)).flatten()
augmentations[:batch_size//2] = augmentations[:batch_size//2].fliplr()
augmentations_noise = augmentations + noise_std*torch.randn(augmentations.shape)

# augmentations = torch.cat([augmentations_noise,augmentation_flip],dim=0)
augmentations = augmentations_noise
aggregation_results = {'patch_aggregation':'uniform','I':I}
saliency_dict = get_saliency(augmentations,aggregation_results,patch_size,imagenet_target,saliency_method='gradcam')