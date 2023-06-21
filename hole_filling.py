TODO = None
def fill_holes():
    print('TODO:copy this code from PNN_faiss')
    print('TODO:or modify PNN_faiss?')
    holes_as_ON = TODO
    coarse_dim = TODO
    ref = TODO
def faiss_search_with_masks():
    print('TODO: add an example to search faiss with a mask for indexes')
    pass


def PNN_faiss(self, x, x_scaled, y_scaled, patch_size, stride, alpha, mask=None, new_keys=True,
        other_x=None,extra_return={},query_mask=None,
        Dprev=None,Iprev=None):
        if self.resolution == None:
            self.resolution = x.shape
        assert (x.max() <= 1.) and (x.min() >= 0.)
        assert (x_scaled.max() <= 1.) and (x_scaled.min() >= 0.)
        # y_scaled has noise added, so will have >1 and <0 values
        # assert (y_scaled.max() <= 1.) and (y_scaled.min() >= 0.)
        other_x = None;print('setting other_x to None forcefully')
        print('using faiss')
        print('this shouldnt be np.array but also work for tensor')
        assert y_scaled[0].shape[-1] == 3
        queries = torch.stack([extract_patches(ys, patch_size, stride) for ys in y_scaled],dim=0)

            
        from model.hog import gradient_histogram
        
        # import pdb;pdb.set_trace()
        # queries = queries[...,::2,::2]
        print('extracted query',queries.shape)
        keys = extract_patches(x_scaled, patch_size, stride)
        use_lr_flip = False
        if use_lr_flip:
            keys = cat_lr_flip(keys)
        # keys = keys[...,::2,::2]
        print('extracted keys')
        if x.shape not in self.n_keys:
            self.n_keys[x.shape] = keys.shape[0]
        if True:
            values = extract_patches(x, patch_size, stride)
            if use_lr_flip:
                values = cat_lr_flip(values)
            if self.KEYS_TYPE == 'multi-resolution':
                if new_keys:
                    if self.running_values is None:
                        self.running_values = values
                    else:
                        if self.running_values.shape[0] == sum(self.n_keys.values()):
                            self.running_values[-self.n_keys[x.shape]:] = values
                        else:
                            self.running_values = torch.cat([self.running_values,values],dim=0)

        else:
            print('using laplacian pyramid')
            x_high = x - x_scaled
            values = extract_patches(x_high, patch_size, stride)
        if other_x is not None:
            other_values = extract_patches(other_x,patch_size, stride)
        print('extracted values')
        if mask is not None:
            # assert False,'not implemented for 4d query'
            assert mask.ndim == 1,'TODO: should mask look like I?'
            print('TODO: we need a multiscale mask')
            queries = queries[mask]
            keys = keys[~mask]
        #====================================================================
        assert queries.ndim == 5
        queries_flat_batch = queries.flatten(start_dim=0,end_dim=1).contiguous()
        if query_mask is not None:

            queries_flat_batch = queries_flat_batch[query_mask]        
        assert keys.ndim == 4
        if new_keys:
            # keys_proj = keys_flat
            keys_proj = self.get_feats(keys,init=True)
            # import pdb;pdb.set_trace()
            '''
            if self.USE_PCA:
                # keys_proj = self.fit_transform_pca(self.N_PCA_COMPONENTS,keys_proj)
                keys_proj = self.get_pca_feats(keys_proj,n_pca_components = self.N_PCA_COMPONENTS,init=True)
            '''
            #==================================================
            if self.KEYS_TYPE == 'multi-resolution':
                if self.running_keys is not None:
                    # assert False
                    if self.running_keys.shape[0] == sum(self.n_keys.values()):
                        self.running_keys[-self.n_keys[x.shape]:] = keys_proj
                    else:
                        self.running_keys = torch.cat([self.running_keys,keys_proj],dim=0)
                    # self.running_keys =self.running_keys[-keys_proj.shape[0]:]
                else:
                    self.running_keys = keys_proj
                # import pdb;pdb.set_trace()
            #==================================================
            
            n_patches = keys_proj.shape[0]
            print(n_patches)
            # import pdb;pdb.set_trace()

            if self.KEYS_TYPE == 'single-resolution':
                keys_for_index = keys_proj
            elif self.KEYS_TYPE == 'multi-resolution':
                keys_for_index = self.running_keys
            self.index = create_index(keys_for_index,index_type='ivf',index_options={'nlist':200})

        queries_proj = self.get_feats(queries_flat_batch,init=False)
        nearest_neighbor_results = get_nearest_neighbors_of_subset(queries_proj,query_mask,None,index=self.index,Ddtype='float',Idtype='int',max_batch_size=62496,D=Dprev,I=Iprev)
        # these are now the full keys, not just subset
        D,I = nearest_neighbor_results['D'],nearest_neighbor_results['I']
        print('searching')
        print(queries_proj.shape)

        if mask is not None:
            # assert False,'not implemented'
            values[mask] = values[~mask][I.T]
            if self.KEYS_TYPE == 'single-resolution':
                assert I.shape[-1] == 1
                values[mask] = torch.index_select(values[~mask],0,I.squeeze()).unsqueeze(0)
                
            elif self.KEYS_TYPE == 'multi-resolution':
                raise NotImplementedError
        else:

            if self.KEYS_TYPE == 'single-resolution':
                if False:
                    values = values[I.T]
                else:
                    assert I.shape[-1] == 1
                    values = torch.index_select(values,0,I.squeeze()).unsqueeze(0)
                    # assert (values[I.T] == values1).all()
                    # import sys;sys.exit()
            elif self.KEYS_TYPE == 'multi-resolution':
                print('TODO: copy this to mask is not None section')
                print('TODO: delegate maintenance of running keys and values to a utility function')
                print('using running values')
                values = self.running_values[I.T]
                assert False,'index_select not implemented'
        
        assert values.ndim == 5
        assert values.shape[0] == 1

        values = values.squeeze()
        # import pdb;pdb.set_trace()
        values = values.reshape(queries.shape[0],values.shape[0]//queries.shape[0],*values.shape[1:])
        distances = D.reshape(queries.shape[0],D.shape[0]//queries.shape[0],1,1,1)
        debug.distances = distances
        if 'check' and False:
            chosen_keys = keys[I.T]
            chosen_keys = chosen_keys.squeeze()
            d0 = (chosen_keys[keys.shape[0]*0:keys.shape[0]*(0+1)] - keys).abs()
            diffs = [(chosen_keys[keys.shape[0]*ii:keys.shape[0]*(ii+1)] - keys).abs().sum() for ii in range(self.batch_size) ]
            flags = [torch.isclose(d,torch.zeros_like(d)) for d in diffs]
            assert all(flags)
        #====================================================================
        assert len(x_scaled.shape) == 4
        # import pdb;pdb.set_trace()
        # import IPython;IPython.embed()
        '''
        y = torch.stack([combine_patches(v, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                                            patch_aggregation=self.PATCH_AGGREGATION,
                                            distances=d)['combined'] for v,d in zip(values,distances)],dim=0)
        '''
        if True and '1 image at a time':
            y = []
            w = []
            from collections import defaultdict
            patch_aggregation_results = defaultdict(list)
            
            for ii,(v,d) in enumerate(zip(values,distances)):            
                
                # debug.I = I[ii*distances.shape[1]:(ii+1)*distances.shape[1]]
                print('TODO:debug.I should trigger error in distance-weighted aggregation')
                result = combine_patches(v, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=d,I=I)
                for k in result:
                    if k not in ['patch_aggregation']:
                        # import pdb;pdb.set_trace()
                        patch_aggregation_results[k].append(result[k])
                # yi,wi = result['combined'],result['weights']
                # y.append(yi)
                # w.append(wi)
            patch_aggregation_results = {k:v for k,v in patch_aggregation_results.items()}
            patch_aggregation_results['patch_aggregation']=result['patch_aggregation']
            for k in result:
                if k not in ['patch_aggregation']:
                    patch_aggregation_results[k] = torch.stack(patch_aggregation_results[k],dim=0)
                    # .append(torch.stack(patch_aggregation_results[k],dim=0))        
        elif  'multiple images':
            patch_aggregation_results = combine_patches(values, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=distances,I=I)
            
        patch_aggregation_results['D'], patch_aggregation_results['I'] = D,I

        if other_x is not None:
            assert False,'shouldnt be here'
            # assert isinstance(other_x,torch.Tensor)
            other_y = combine_patches(other_values, patch_size, stride, 
                                    x_scaled.shape,patch_aggregation=self.PATCH_AGGREGATION,as_np=False)
            extra_return['other_y']  = other_y
        print('combined')
        if patch_aggregation_results['combined'].shape[-1] !=3:
            import pdb;pdb.set_trace()
        if 1 in patch_aggregation_results['combined'].shape[1:3]:
            import pdb;pdb.set_trace()
        # return y,I,w
        return patch_aggregation_results
    
def augmentation_attribution(config=None):
    device = 'cpu' if config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
    model = gpnn(config)
    #========================================================
    ref = config['input_img']
    patch_size = config['patch_size']
    stride = config['stride']
    ref = torch.tensor(ref,device=device).float().unsqueeze(0).permute(0,3,1,2)
    #========================================================
    # get initial gradcam

    from saliency import get_cams
    cam0,_,_ = get_cams(ref,imagenet_target,method='gradcam')
    cam0 = torch.tensor(cam0).unsqueeze(1).to(device)
    # config['input_img'] = config['input_img']* tensor_to_numpy(cam0[0,0][...,None]);print('setting input_image to masked')
    # debug.cam0 = torch.ones_like(cam0);print('setting cam0 to ones')
    from torch.nn.functional import unfold
    cam0u = unfold(cam0, kernel_size=(patch_size,patch_size), dilation=(1, 1), stride=stride, padding=(0, 0))
    debug.cam0 = cam0u
    
    
    
    #========================================================
    with Timer('model run'):
        augmentations,aggregation_results = model.run(to_save=False)
    print('early return from hole filling')
    return augmentations,aggregation_results,None,model
    # ref = config['input_img']
    # patch_size = config['patch_size']
    if False and 'ignore saliency':
        #========================================================
        # add initial image to the augmentations
        ref = cam0 + torch.zeros_like(ref)
        augmentations = torch.cat([augmentations,ref],dim=0)
        torch.cuda.empty_cache()
        # output_im = skimage.io.imread(output_imname)
        saliency_dict = {}
        # import pdb;pdb.set_trace()
        
        for k in ['weights','I']:
            if k in aggregation_results:
                v = aggregation_results[k]
                if k == 'weights':
                    # new_v = torch.ones_like(v[:1])
                    new_v  = torch.ones_like(v[:1]) * (1/49.)
                elif k == 'I':
                    new_v = torch.arange( 
                                                        np.prod(
                                                            [
                                                            ref.shape[-2] - 2*(patch_size//2),
                                                            ref.shape[-1] - 2*(patch_size//2)
                                                            ]
                                                        ),device=device)
                    new_v = new_v.flatten()[:,None]
                                                        
                aggregation_results[k] = torch.cat([v,new_v],dim=0)
        # import pdb;pdb.set_trace()
        saliency_dict = get_saliency(augmentations,aggregation_results,(config['patch_size'],config['patch_size']),imagenet_target,saliency_method='gradcam')
    return augmentations,aggregation_results,saliency_dict,model
#==============================================
#==============================================
#! /opt/conda/bin/python
#%%
import os
import numpy as np
# os.chdir('/root/evaluate-saliency-5/GPNN')
#%%
# faiss doesnt work without gpu
# !python random_sample.py -in database/balloons.png --faiss
import skimage.io
from matplotlib import pyplot as plt
from model.utils import Timer
from model.my_gpnn  import extract_patches,combine_patches
import torch
from model.utils import *
import skimage.io
import skimage.transform
from torch.nn.functional import fold,unfold
from model.my_gpnn import gpnn
import debug
#==============================================
def main(config=None):

    for d in ['output','camoutput','unpermuted_camsoutput','maskoutput']:
        os.system(f'rm -rf {d}')
    augmentations,aggregation_results,saliency_dict,model = augmentation_attribution(config=config)
    #==============================================
    # save
    # augmentations
    for i,aug in enumerate(augmentations):
        img_save(tensor_to_numpy(aug.permute(1,2,0)), model.out_file[:-len('.png')] + str(i) + '.png' )
    #==============================================

if __name__ == '__main__':    
    if True:
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

        output_imname = os.path.join('output',os.path.basename(original_imname))
        output_imname_root,ext = output_imname.split('.')
        output_imname = output_imname_root + '_random_sample' +'.png'
        original_im = skimage.io.imread(original_imname)
        # !python random_sample.py -in  {original_imname} --faiss
        # assert False
        im = skimage.io.imread(original_imname)
        if im.ndim == 3 and im.shape[-1] > 3:
            im = im[...,:-1]
        target_size = 256
        overshoot = min(im.shape[:2])/target_size
        resize_aspect_ratio = 1/overshoot
        print(im.shape)

            
        im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))
        im = im[-256:,-256:];print('forcibly cropping image')
        print(im.shape)    
        mask = np.zeros(im.shape[:2])
        mask[150:,150:] = 1.
        config = {
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
        # 'task':'random_sample',
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':im,
        'batch_size':2,
        #---------------------------------------------
        'implementation':'gpnn',#'efficient-gpnn','gpnn'
        'init_from':'zeros',#'zeros','target'
        'keys_type':'single-resolution',#'multi-resolution','single-resolution'
        #---------------------------------------------
        'use_pca':True,
        'n_pca_components':30,
        #---------------------------------------------
        'patch_aggregation':'distance-weighted',#'uniform','distance-weighted','median'
        'mask':mask,
        }    
    main(config=config)
