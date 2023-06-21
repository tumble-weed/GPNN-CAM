from model.my_gpnn  import extract_patches,combine_patches
from model.my_gpnn import gpnn
import skimage.io
import torch
import torchvision
from PIL import Image
import register_ipdb
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # from model.gpnn import gpnn

    # original_imname = 'images/372.png'; imagenet_target=372
    # original_imname = 'images/ILSVRC2012_val_00000013.JPEG'; imagenet_target=370
    # original_imname = 'database/balloons.png'
    # original_imname = 'database/volacano.png';imagenet_target = None
    # print('is it class 6 for voc?')
    original_imname = 'images/cars.png'; imagenet_target = 751#829#751#6#
    # original_imname = 'images/vulture.jpeg'; imagenet_target = 23
    # original_imname = 'images/n01443537_16.JPEG'; imagenet_target = 1
    # original_imname = 'images/data/feature_inversion/building.jpg'; imagenet_target = 538
    

    
    original_im = skimage.io.imread(original_imname)
    im = skimage.io.imread(original_imname)
    if im.ndim == 3 and im.shape[-1] > 3:
        im = im[...,:-1]
    target_size = 256
    overshoot = min(im.shape[:2])/target_size
    resize_aspect_ratio = 1/overshoot
    print(im.shape)
    im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))
    im_pil = Image.fromarray((im*255).astype(np.uint8))
    # im = im[-256:,-256:];print('forcibly cropping image')
    print(im.shape)    
        # print(colored('brute force making gpnn-gradcam, change in utils?','red'))
        # import os;os.makedirs('gpnn-gradcam')
    
    transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(target_size),
                # torchvision.transforms.Normalize(mean=mean,std=std),
                ]
            )
    ref = transform(im).unsqueeze(0).float().to(device)
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
        'imagenet_target':imagenet_target,
        'n_super_iters':10,
        #---------------------------------------------
        'device':device,        
        'index_type':'ivf',
        }    
    # main(config=config,save_results = True)    
    model = gpnn(config)
    # device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    augmentations,aggregation_results = model.run(to_save=False)
    import ipdb;ipdb.set_trace()
main()