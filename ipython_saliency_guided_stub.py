import glob
import skimage.io
import pickle
from benchmark.visualize_results import get_get_available_dirs
results_dir = settings.RESULTS_DIR_librecam
imagenet_root = '/root/bigfiles/dataset/imagenet'
ref_methodname = 'gradcam'
modelname = 'vgg16'
img_ix = 1
get_available_dirs = get_get_available_dirs(ref_methodname,modelname,results_dir=results_dir)
#save_dir = os.path.join(settings.RESULTS_DIR_librecam,'method-saliency-comparison')
image_paths = list(sorted(glob.glob(
        os.path.join(results_dir,f'{ref_methodname}-{modelname}','*/')
    )))
img_path = image_paths[img_ix]
imroot = os.path.basename(img_path.rstrip(os.path.sep))
im = skimage.io.imread(os.path.join(imagenet_root,'images','val',imroot+'.JPEG'))
methoddir =  os.path.join(results_dir,f'{ref_methodname}-{modelname}')
im_results_dir = os.path.join(methoddir,imroot)
pklnames = glob.glob(os.path.join(im_results_dir,'*.pkl'))
assert len(pklnames) == 1
pklname = pklnames[0]
with open(pklname,'rb') as f:
    loaded = pickle.load(f)
saliency = loaded['saliency']
target_id = loaded['target_id']
ref = loaded['ref']
#assert ref.min() >0 and ref.max() < 1

