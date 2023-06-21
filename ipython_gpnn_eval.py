imgflt.shape
#[Out]# torch.Size([3, 66752])
maskflt.shape
#[Out]# torch.Size([66752])
img.shape
#[Out]# torch.Size([3, 224, 298])
mask.shape
#[Out]# torch.Size([224, 298])
img.shape,mask.shape
#[Out]# (torch.Size([3, 224, 298]), torch.Size([224, 298]))
if True:
    from model.my_gpnn import gpnn
    
if True:
    from model.my_gpnn import gpnn
            config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
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
        # 'imagenet_target':imagenet_target
        } 
if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
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
        # 'imagenet_target':imagenet_target
        }
        
if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':img,
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
        # 'imagenet_target':imagenet_target
        }
        gpnn_inpainting = gpnn(config)
if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':img,
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
        # 'imagenet_target':imagenet_target
        }
 gpnn_inpainting = gpnn(config)if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':img,
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
        # 'imagenet_target':imagenet_target
        }
  gpnn_inpainting = gpnn(config)if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':img,
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
        # 'imagenet_target':imagenet_target
        }
   gpnn_inpainting = gpnn(config)if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':img,
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    
if True:
    from model.my_gpnn import gpnn
    config = {
        'out_dir':'gpnn-elp/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':img,
        'mask':mask,
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    
mask.shape
#[Out]# torch.Size([224, 298])
img.shape
#[Out]# torch.Size([3, 224, 298])
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(img),
        'mask':tensor_to_numpy(mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(img),
        'mask':tensor_to_numpy(mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(img),
        # NOTE: the mask arrives with 0's at holes. need to flip it
        'mask':tensor_to_numpy(1 - mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(img),
        # NOTE: the mask arrives with 0's at holes. need to flip it
        'mask':tensor_to_numpy(1 - mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
    
img.min()
#[Out]# tensor(-1.8608)
img.max()
#[Out]# tensor(2.5189)
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    out = (t )*torch.tensor(vgg_std).to(device)[None,:,None,None] + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    return out
        
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(denormalize_imagenet(img.unsqueeze(0)).permute(0,2,3,1)[0]),
        # NOTE: the mask arrives with 0's at holes. need to flip it
        'mask':tensor_to_numpy(1 - mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
        device = t.device
        out = (t )*torch.tensor(vgg_std).to(device)[None,:,None,None] + torch.tensor(vgg_mean).to(device)[None,:,None,None]
        return out
        
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(denormalize_imagenet(img.unsqueeze(0)).permute(0,2,3,1)[0]),
        # NOTE: the mask arrives with 0's at holes. need to flip it
        'mask':tensor_to_numpy(1 - mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
    
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    def denormalize_imagenet(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
        device = t.device
        out = (t )*torch.tensor(vgg_std).to(device)[None,:,None,None] + torch.tensor(vgg_mean).to(device)[None,:,None,None]
        return out
        
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(denormalize_imagenet(img.unsqueeze(0)).permute(0,2,3,1)[0]),
        # NOTE: the mask arrives with 0's at holes. need to flip it
        'mask':tensor_to_numpy(1 - mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
    
holefilled.shape
#[Out]# torch.Size([1, 3, 224, 298])
dutils.img_save(tensor_to_numpy(holefilled.permute(0,2,3,1)[0]),'holefilled.png')
mask.shape
#[Out]# torch.Size([224, 298])
img.shape
#[Out]# torch.Size([3, 224, 298])
if True:
    from model.my_gpnn import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    def denormalize_imagenet(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
        device = t.device
        out = (t )*torch.tensor(vgg_std).to(device)[None,:,None,None] + torch.tensor(vgg_mean).to(device)[None,:,None,None]
        return out
        
    config = {
        'out_dir':'gpnn-eval/output',
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
        'task':'inpainting',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':tensor_to_numpy(denormalize_imagenet(img.unsqueeze(0)).permute(0,2,3,1)[0]),
        # NOTE: the mask arrives with 0's at holes. need to flip it
        'mask':tensor_to_numpy(1 - mask),
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
        # 'imagenet_target':imagenet_target
        }
    gpnn_inpainting = gpnn(config)
    import dutils
    dutils.img_save(tensor_to_numpy(1-mask),'imputation_mask.png')
    dutils.img_save(tensor_to_numpy(img.permute(1,2,0)),'imputation_img.png')
    holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
    dutils.img_save(tensor_to_numpy(holefilled.permute(0,2,3,1)[0]),'imputed.png')
    
""" Our linear inputation scheme. """
"""
This is the function to do the linear infilling 
img: original image (C,H,W)-tensor;
mask: mask; (H,W)-tensor

"""
imgflt = img.reshape(img.shape[0], -1)
maskflt = mask.reshape(-1)
        # Indices that need to be imputed.
indices_linear = np.argwhere(maskflt==0).flatten() 
# Set up sparse equation system, solve system.
A, b = NoisyLinearImputer.setup_sparse_system(mask.numpy(), 
            img.numpy(), neighbors_weights)
res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

# Fill the values with the solution of the system.
img_infill = imgflt.clone()
img_infill[:, indices_linear] = res.t() + self.noise*torch.randn_like(res.t())
img_infill.shape
#[Out]# torch.Size([3, 66752])
img_infill.reshape_as(img).shape
#[Out]# torch.Size([3, 224, 298])
holefilled.shape
#[Out]# torch.Size([1, 3, 224, 298])
