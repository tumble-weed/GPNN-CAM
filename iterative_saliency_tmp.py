import torch
from collections import defaultdict
import skimage.transform
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
TODO = None
def augmentation_attribution2(config=None):

    #========================================================
    ref = config['input_img']
    patch_size = config['patch_size']
    stride = config['stride']
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    #========================================================    
    from model.my_gpnn import gpnn
    gpnn_model = gpnn(config)
    ref = torch.tensor(ref,device=device).float().unsqueeze(0).permute(0,3,1,2)
    visibility = torch.ones_like(ref[:,:1]).float().requires_grad_(True)
    optimizer_visibility = torch.optim.Adam([visibility],lr=1e-3)
    trends = defaultdict(list)
    cnn = get_cnn(config['datasetname'],config['modelname']).to(device)
    print('STAGE:initial assets created')

    #========================================================    
    config['n_augmentation_epochs']=1;print('TODO:hardcoding n_augmentation_epochs')
    for i in range(config['n_augmentation_epochs']):
        print('TODO:get augmentations here?, using model.run?')        
        if False:
            '''
            augmentations = TODO(TODO,other=visibility)
            TODO.distances
            TODO.patches
            '''
        else:
            augmentations = ref.detach().clone().to(device)
            distances = torch.zeros(1,1,augmentations.shape[-2]-6,augmentations.shape[-1]-6).float().to(device)
            patches = torch.zeros((1,3,augmentations.shape[-2]-6,augmentations.shape[-1]-6,7,7))
            print('TODO: should this be I instead?')
            augmentations = augmentations*visibility
            print('TODO:hardcoding augmentations,distances,patches')

        
        def run_1_saliency_iter(augmentations,cnn,target_class,visibility):
            config['n_saliency_iter'] = 1;print('TODO:hardcoding n_saliency_iter')
            for ii in range(config['n_saliency_iter']):
                print('TODO: jitter the augmentations for the visibility?')
                print('TODO: 1st experiment without jistter?')
                augmentations1 = normalize_tensor(augmentations)
                if 'forward-backward step':
                    
                    scores = cnn(augmentations1)     
                    loss = -scores[:,config['target_class']].sum()
                    optimizer_visibility.zero_grad()
                    loss.backward()
                    optimizer_visibility.step()                
                    print('TODO:1st experiment without smoothness')
                    if False:
                        visibility.data.copy_(visibility.TODO_clamp)
                    else:
                        print('TODO:commented out visibility clamp')
                print('TODO: 1st experiment with a single image')
                trends['loss'].append(loss.item())
                trends['images'] = tensor_to_numpy(augmentations)
                '''
                from nearest_neigbors import get_nearest_neighbors_of_subset
                query_patches = TODO.unfold(augmentations1)
                print('TODO for first experiment modify all patches')
                query_mask = torch.arange(TODO);print('hard coding query mask to pick all patches;)
                I = TODO
                D = TODO; print('TODO: is this distances')
                nn_results = get_nearest_neighbors_of_subset(
                    query_patches,
                    query_mask,
                    keys=None,
                    index=gpnn_model.index,
                    Ddtype='float',
                    Idtype='int',
                    max_batch_size=62496,
                    D=None,I=None)
                # out =  dict(
                #     D = D,
                #     I = I,
                #     index = out_subset['index'],
                # )
                assert torch.allclose(D,nn_results['D']),'should be in place'
                assert torch.allclose(I,nn_results['I']),'should be in place'
                
                # [ combine_patches() for _ ]
                from aggregation import combine_patches
                
                if TODO.e == 0 and TODO.i == 0:
                    values = extract_patches(x, patch_size, stride)
                if True and '1 image at a time':
                    patch_aggregation_results = combine_patches_multiple_images(values, 
                        patch_size, stride, img_shape,
                        as_np=False,
                        patch_aggregation=None,
                        distances=None,I=None)
                elif  'multiple images':
                    patch_aggregation_results = combine_patches(values, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                        patch_aggregation=self.PATCH_AGGREGATION,
                        distances=distances,I=I)

                TODO.update.recombine_patches(TODO.patches,
                                              TODO.visibility,
                                              TODO.distances)
                
                '''
        run_1_saliency_iter(augmentations,cnn,config['target_class'],visibility)
if __name__ == '__main__':
    # original_imname = 'images/372.png'; imagenet_target=372
    # original_imname = 'images/ILSVRC2012_val_00000013.JPEG'; imagenet_target=370
    # original_imname = 'database/balloons.png'
    # original_imname = 'database/volacano.png';imagenet_target = None
    # print('is it class 6 for voc?')
    original_imname = 'images/cars.png'; imagenet_target = 751#829#751#6#
    # original_imname = 'images/vulture.jpeg'; imagenet_target = 23
    # original_imname = 'images/n01443537_16.JPEG'; imagenet_target = 1
    # original_imname = 'images/data/feature_inversion/building.jpg'; imagenet_target = 538
    im = skimage.io.imread(original_imname)
    if im.ndim == 3 and im.shape[-1] > 3:
        im = im[...,:-1]
    target_size = 256
    overshoot = min(im.shape[:2])/target_size
    resize_aspect_ratio = 1/overshoot
    print(im.shape)

    im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))


    config = {
        #---------------------------------------------
        'target_class' : imagenet_target,
        'datasetname':'imagenet',
        'modelname':'resnet50',
        #---------------------------------------------
        'out_dir':'output',
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
        'input_img':im,
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
        }    
    augmentation_attribution2(config=config)
