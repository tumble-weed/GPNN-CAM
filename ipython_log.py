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
