if True:
    from benchmark.run_metrics_librecam import run_chattopadhyay
    import skimage.io
    model = modelname
    methodnames = ['scorecam']
    dataset = 'imagenet'
    device = 'cuda'
    #dataset,imroot,modelname,methodnames
    def get_model(dataset,modelname,is_relevancecam,device=None):
        
        assert modelname in ['vgg16','resnet50'],f'{modelname} not recognized'
        rprop = False
        convert_to_fully_convolutional=True
        if is_relevancecam:
            # convert_to_fully_convolutional=False
            rprop = True
        # print('TODO:fix hardcode of voc')
        if dataset in ['imagenet','']:
            dataset = 'imagenet'
            import libre_cam_models.relevance.vgg
            import libre_cam_models.relevance.resnet
            # import ipdb;ipdb.set_trace()
            if modelname == 'vgg16':
                model = libre_cam_models.relevance.vgg.vgg16_bn(pretrained = True).to(device)
            else:
                model = libre_cam_models.relevance.resnet.resnet50(pretrained = True).to(device)                    
        elif dataset in ['voc','pascal']:
            dataset = 'voc'
            from benchmark.architectures import get_model as get_model_
            model = get_model_(
                        arch=modelname,
                        dataset=dataset,
                        rprop = rprop,
                        convert_to_fully_convolutional=convert_to_fully_convolutional,
                        
                    )
        else:
            assert False
        model.to(device)
        model.dataset = dataset
        return model
    
    has_relevancecam = 'relevancecam' if [any('relevancecam' in methodname for methodname in methodnames)] else None
    
    model = get_model(dataset,modelname,has_relevancecam,device=device)    
    model.eval()
    dataset_stub = ''
    methodname = methodnames[0]
    methoddir =  os.path.join(settings.RESULTS_DIR_librecam,f'{dataset_stub}{methodname}-{modelname}')
    #im_save_dirs = sorted(glob.glob(os.path.join(methoddir,'*/')))
    im_save_dir = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset_stub}{methodname}-{modelname}',imroot)
    if True:
                pklname = glob.glob(os.path.join(im_save_dir,'*.pkl'))
                assert len(pklname) == 1
                pklname = pklname[0]
                try:
                    with open(pklname,'rb') as f:
                        loaded = pickle.load(f)
                    saliency = loaded.get('saliency',np.nan)
                    if isinstance(saliency,torch.Tensor):
                        saliency = tensor_to_numpy(saliency)
                    assert not np.isnan(np.sum(saliency))
                except EOFError as e:
                    pass
                target_id = loaded['target_id']
                from benchmark.run_metrics_librecam import get_image_tensor,IMAGENET_ROOT
                #import skimage.io
                import numpy as np
                from PIL import Image
                def get_image_tensor(impath,size=224):
                    from cnn import get_vgg_transform
                    import skimage.io
                    vgg_transform = get_vgg_transform(size)
                    im_ = skimage.io.imread(impath)
                    if im_.ndim == 2:
                        im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
                    im_pil = Image.fromarray(im_)
                    ref = vgg_transform(im_pil).unsqueeze(0)
                    return ref
                imagenet_root = IMAGENET_ROOT
                saliency = loaded.get('saliency',np.nan)
                impath = os.path.join(imagenet_root,'images','val',imroot + '.JPEG')
                input_tensor = get_image_tensor(impath,size=saliency.shape[-2:]).to(device)                
                metric_data =run_chattopadhyay(input_tensor,saliency,model,target_id)
                
