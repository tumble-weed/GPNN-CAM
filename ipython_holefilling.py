# IPython log file

augmentations0 = saliency_dict['augmentations']
augmentations.shape
import dutils
dutils.img_save(augmentations[:1])
dutils.img_save(augmentations[:1],'augmentation.png')
import importlib
importlib.reload(dutils)
dutils.img_save(augmentations[:1],'augmentation.png')
dutils.img_save(tensor_to_numpy(augmentations[:1]),'augmentation.png')
dutils.img_save(tensor_to_numpy(augmentations[:1].permute(0,2,3,1)[0,0]),'augmentation.png')
dutils.img_save(tensor_to_numpy(augmentations[0,0]),'augmentation.png')
augmentations.shape
dutils.img_save(tensor_to_numpy(augmentations[:1].permute(0,2,3,1)[0]),'augmentation.png')
get_ipython().run_line_magic('output', '')
get_ipython().run_line_magic('logstart', '')
get_ipython().run_line_magic('logstop', '')
get_ipython().run_line_magic('logstart', 'ipython_holefilling.py')
import dutils
dutils.img_save(tensor_to_numpy(augmentations[:1].permute(0,2,3,1)[0]),'augmentation.png')
dutils.img_save(tensor_to_numpy(augmentations[:1].permute(0,2,3,1)[0]),'saliency.png')
saliency_dict['avg_saliency'].shape
dutils.img_save(tensor_to_numpy(saliency_dict['avg_saliency'][0,0]),'saliency.png')
dutils.img_save(saliency_dict['avg_saliency'][0,0],'saliency.png')
import gradcam
import importlib
importlib.reload(gradcam)
saliency_dict = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method='gradcam')
dutils.img_save(saliency_dict['avg_saliency'][0,0],'saliency.png')
importlib.reload(saliency)
import saliency
importlib.reload(saliency)
from saliency import get_saliency
saliency_dict = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method='gradcam')
dutils.img_save(saliency_dict['avg_saliency'][0,0],'saliency.png')
saliency.model
gradcam.model
del gradcam.model
from saliency import get_saliency
saliency_dict = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method='gradcam')
dutils.img_save(saliency_dict['avg_saliency'][0,0],'saliency.png')
with_holes = tensor_to_numpy(augmentations[0]) * saliency_dict['avg_saliency'][0,0][...,None]
with_holes = tensor_to_numpy(augmentations[0]) * saliency_dict['avg_saliency'][0,0][None,...]
with_holes.__class__
augmentation_with_holes = tensor_to_numpy(augmentations[0]) * saliency_dict['avg_saliency'][0,0][None,...]
saliency_dict['avg_saliency'].min()
saliency_dict['avg_saliency'].max()
get_ipython().system('pip install seaborn')
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure()
sns.kde_plot(saliency_dict['avg_saliency'].flatten(),fig=)
fig = plt.figure()
sns.kde_plot(saliency_dict['avg_saliency'].flatten(),fig=fig)
sns.kdeplot(saliency_dict['avg_saliency'].flatten(),fig=fig)
sns.kdeplot(saliency_dict['avg_saliency'].flatten(),ax = plt.gca())
plt.draw()
plt.savefig(os.path.join(dutils.ROOT_DIR,'avg_saliency_values.png'))
plt.close()
saliency_dict['avg_saliency']] > 0.8
(saliency_dict['avg_saliency'] > 0.8).sum()
(saliency_dict['avg_saliency'] > 0.8).sum()
avg_saliency_high = (saliency_dict['avg_saliency'] > 0.8).astype(np.float32) * saliency_dict['avg_saliency']
img_save(avg_saliency_high[0,0],'avg_saliency_high.png')
dutils.img_save(avg_saliency_high[0,0],'avg_saliency_high.png')
avg_saliency_high = (saliency_dict['avg_saliency'] > 0.6).astype(np.float32) * saliency_dict['avg_saliency']
dutils.img_save(avg_saliency_high[0,0],'avg_saliency_high.png')
avg_saliency_high = (saliency_dict['avg_saliency'] > 0.7).astype(np.float32) * saliency_dict['avg_saliency']
dutils.img_save(avg_saliency_high[0,0],'avg_saliency_high.png')
avg_saliency_low = (saliency_dict['avg_saliency'] < 0.2).astype(np.float32) * saliency_dict['avg_saliency']
dutils.img_save(avg_saliency_low[0,0],'avg_saliency_low.png')
def save_high_low_saliency(saliency,thresh_low,thresh_high):
    saliency_high = (saliency > thresh_high).astype(np.float32)
    saliency_low = (saliency < thresh_low).astype(np.float32)
    dutils.img_save(saliency_low[0,0],'avg_saliency_low.png')
    dutils.img_save(saliency_high[0,0],'avg_saliency_high.png')
    
save_high_low_saliency(saliency_dict['avg_saliency'],0.2,0.6)
def save_high_low_saliency(saliency,thresh_low,thresh_high):
    import dutils
    saliency_high = (saliency > thresh_high).astype(np.float32)
    saliency_low = (saliency < thresh_low).astype(np.float32)
    dutils.img_save(saliency_low[0,0],'avg_saliency_low.png')
    dutils.img_save(saliency_high[0,0],'avg_saliency_high.png')
    
save_high_low_saliency(saliency_dict['avg_saliency'],0.2,0.6)
save_high_low_saliency(saliency_dict['avg_saliency'],0.1,0.6)
save_high_low_saliency(saliency_dict['avg_saliency'],0.01,0.6)
dutils.img_save(saliency_dict['total_mass'],'total_mass')
saliency_dict.keys()
dutils.img_save(saliency_dict['mass_sum'],'mass_sum.png')
dutils.img_save(saliency_dict['mass_sum'][0,0],'mass_sum.png')
dutils.img_save((saliency_dict['mass_sum'][0,0]>0),'mass_sum.png')
dutils.img_save((saliency_dict['mass_sum'][0,0]>1),'mass_sum.png')
dutils.img_save((saliency_dict['mass_sum'][0,0]>2),'mass_sum.png')
dutils.img_save((saliency_dict['mass_sum'][0,0]>1),'mass_sum.png')
augmentation_with_holes = tensor_to_numpy(augmentations[0]) * (saliency_dict['avg_saliency'][0,0][None,...] > 0.6).astype(np.float32)
config
config_inpainting = copy.deepcopy(config)
config_inpainting['input_img'] = augmentation_with_holes
config_inpainting['task'] = 'inpainting'
config_inpainting['mask'] = (saliency_dict['avg_saliency'] > 0.6).astype(np.float32)[0,0]
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
augmentation_with_holes.shape
augmentation_with_holes = tensor_to_numpy(augmentations.permute(0,2,3,1)[0]) * (saliency_dict['avg_saliency'][0,0][...,None] > 0.6).astype(np.float32)
config_inpainting['input_img'] = augmentation_with_holes
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
config_inpainting['device']
config_inpainting['no_cuda'] = True
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
config_inpainting['no_cuda'] = True
gpnn_inpainting = gpnn(config_inpainting)
import model.my_gpnn
del model.my_gpnn.device
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
get_ipython().run_line_magic('debug', '')
gpnn_inpainting.input_img.shape
gpnn_inpainting.mask.shape
import os
os.envirom['CUDA_LAUNCH_BLOCKING']="1"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
get_ipython().run_line_magic('debug', '')
import model.my_gpnn
import importlib
importlib.reload(model.my_gpnn)
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
get_ipython().run_line_magic('debug', '')
get_ipython().run_line_magic('debug', '')
from model.my_gpnn import gpnn
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
holefilled_images.shape
dutils.img_save(tensor_to_numpy(holefilled_images.permute(0,2,3,1))[0],'holefilled.png')
config_inpainting['input_img'] = ref
gpnn_inpainting = gpnn(config_inpainting)
gpnn_inpainting = gpnn(config_inpainting)
ref.__class__
config_inpainting['input_img'] = tensor_to_numpy(ref.permute(0,2,3,1))[0]
config_inpainting['input_img'] = tensor_to_numpy(ref.permute(0,2,3,1))[0]

#============================================
def save_high_low_saliency(saliency,thresh_low,thresh_high):
    import dutils
    saliency_high = (saliency > thresh_high).astype(np.float32)
    saliency_low = (saliency < thresh_low).astype(np.float32)
    dutils.img_save(saliency_low[0,0],'avg_saliency_low.png')
    dutils.img_save(saliency_high[0,0],'avg_saliency_high.png')
save_high_low_saliency(saliency_dict['avg_saliency'],0.2,0.6)

config_inpainting = copy.deepcopy(config)
config_inpainting['input_img'] = augmentation_with_holes
config_inpainting['task'] = 'inpainting'
config_inpainting['mask'] = (saliency_dict['avg_saliency'] > 0.6).astype(np.float32)[0,0]
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)
augmentation_with_holes.shape
augmentation_with_holes = tensor_to_numpy(augmentations.permute(0,2,3,1)[0]) * (saliency_dict['avg_saliency'][0,0][...,None] > 0.6).astype(np.float32)
config_inpainting['input_img'] = augmentation_with_holes
gpnn_inpainting = gpnn(config_inpainting)
holefilled_images,hole_filled_results = gpnn_inpainting.run(to_save=False)

