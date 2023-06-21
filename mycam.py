import torch
from regression import *
from model.pca import PCA
from typing import Tuple
import mycam_
import numpy as np
import dutils
import os
tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def get_hook():
    def hook(self,i,o):
        # self.__dict__['feats'] = torch.tensor(tensor_to_numpy(o),device=o.device)   
        if self.__dict__.get('toggle_hook',True):
            self.__dict__['feats'] = o                                             
        # self.__dict__['feats'] = o
    return hook
def get_bhook():
    def bhook(self,ig,og):
        # self.__dict__['grads'] = torch.tensor(tensor_to_numpy(og[0]),device=og[0].device)                                                
        self.__dict__['grads'] = og[0]                         
    return bhook


class Hook():
    def __init__(self,cnn,feat_layers,CAT_FEATS = False,keep_grad=False):
        self.cnn = cnn
        self.feat_layers = feat_layers
        for feat_layer in feat_layers:
            if 'handle' in feat_layer.__dict__:
                del feat_layer.__dict__['handle']
            if 'feats' in feat_layer.__dict__:
                del feat_layer.__dict__['feats']
            h = get_hook()
            feat_layer.handle = feat_layer.register_forward_hook(h)
            if keep_grad:
                # import ipdb;ipdb.set_trace()
                if 'bhandle' in feat_layer.__dict__:
                    del feat_layer.__dict__['bhandle']
                if 'grads' in feat_layer.__dict__:
                    del feat_layer.__dict__['grads']                
                bh = get_bhook()
                feat_layer.bhandle = feat_layer.register_backward_hook(bh)
            # if 'vgg' in str(cnn.__class__).lower():
            #     if 'handle' in cnn.features[feat_layer].__dict__:
            #         cnn.features[feat_layer].handle.remove()    
            #     if 'feats' in cnn.features[feat_layer].__dict__:
            #         del cnn.features[feat_layer].feats
            #     h = get_hook()
            #     cnn.features[feat_layer].handle = cnn.features[feat_layer].register_forward_hook(h)
            # else:
            #     import ipdb;ipdb.set_trace()
    def get_last_feats_(self):
        all_feats = []
        for feat_layer in self.feat_layers:
            feats = feat_layer.feats
            all_feats.append(feats)
        all_feats = tuple(all_feats)
        return all_feats

    def get_last_grads_(self):
        all_grads = []
        for feat_layer in self.feat_layers:
            grads = feat_layer.grads
            all_grads.append(grads)
        all_grads = tuple(all_grads)
        return all_grads

    def __call__(self,normalized_augmentations,inference_mode=True):
        with torch.inference_mode(mode=inference_mode):
            output = self.cnn(normalized_augmentations)
        # all_feats = []
        # for feat_layer in self.feat_layers:
        #     feats = feat_layer.feats
        #     all_feats.append(feats)
        # all_feats = tuple(all_feats)
        all_feats = self.get_last_feats_()
        return (output,) + all_feats
    
class HookToggle():
    def __init__(self,hooked_cnn,state):
        self.hooked_cnn = hooked_cnn
        self.state = state
        pass
    def __enter__(self):
        self.original_toggle_hook = {}
        for feat_layer in self.hooked_cnn.feat_layers:
            self.original_toggle_hook[feat_layer] = feat_layer.__dict__.get('toggle_hook',True)
            feat_layer.toggle_hook = self.state
    def __exit__(self,exc_type, exc_value, traceback):
        for feat_layer in self.hooked_cnn.feat_layers:
            # feat_layer.toggle_hook = False
            feat_layer.toggle_hook = self.original_toggle_hook[feat_layer]

class MyCAM():
    def get_last_feats_(self):
        return self.hooked_cnn.get_last_feats_()
    def get_last_grads_(self):
        return self.hooked_cnn.get_last_grads_()    
    def __init__(self,cnn,dataset,feat_layers=[],batch_size=None,
                 chunk_size = 100*224*224,
                 use_pca=False,keep_grad=False):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        if  len(feat_layers) == 0:
            if 'pascal' in dataset or 'voc' in dataset :
                if 'vgg' in str(cnn.__class__).lower():
                    # feat_layers = [cnn.features[29]]
                    feat_layers = [cnn.features[30]]
                    pass
                elif 'resnet' in str(cnn.__class__).lower():
                    feat_layers = [cnn.layer4]
                pass
            elif dataset == 'imagenet':
                if 'vgg' in str(cnn.__class__).lower():
                    feat_layers = [cnn.features[43]]
                elif 'resnet' in str(cnn.__class__).lower():
                    feat_layers = [cnn.layer4]

        self.hooked_cnn = Hook(cnn,feat_layers,keep_grad=keep_grad)
        self.feat_layers = feat_layers
        self.solved = False
        self.use_pca = use_pca
        pass
    def solve2_using_previous_forward(self,feats_bchw,y_map,sample_weight=None):
        # import ipdb;ipdb.set_trace()
        device = feats_bchw.device
        assert y_map.shape[0] == feats_bchw.shape[0]
        y = y_map.view(-1)
        feats = feats_bchw.flatten(start_dim=-2,end_dim=-1)
        feats = feats.permute(0,2,1)
        feats = feats.reshape(-1,feats.shape[-1])
        #============================================
        feats_mean = feats.mean(dim=(0))
        feats_std = feats.std(dim=(0))
        feats_std = feats_std  + (feats_std==0).float()
        print( feats_mean.shape,feats_std.shape)
        # import ipdb;ipdb.set_trace();#ipdb.post_mortem()
        #============================================        
        X = (feats-feats_mean[None,:])/feats_std[None,:]
        # Y = 
        # w,b = solve_regression(X,y,lambda_ = 0)
        if self.use_pca:
            X = self.pca_obj.fit_transform(X)
        # import ipdb;ipdb.set_trace()
        w,b = solve_standard_regression(X,y,device,
                                        # alpha=0.001
                                        alpha=0.0001,
                                        sample_weight = (None if sample_weight is None else sample_weight)
                                        )
        self.w,self.b = w,b
        self.feats_mean,self.feats_std = feats_mean,feats_std
        self.solved= True

    def solve2(self,normalized_augmentations,target_id,y_map):
        
        device = normalized_augmentations.device
        if self.use_pca:
            self.pca_obj = PCA(200)
        # y_map = torch.ones(50,1,14,14)
        feats_bchw,scores = self.cnn_forward_pass_(normalized_augmentations)
        # import ipdb;ipdb.set_trace()
        assert y_map.shape[0] == feats_bchw.shape[0]
        y = y_map.view(-1)
        feats = feats_bchw.flatten(start_dim=-2,end_dim=-1)
        feats = feats.permute(0,2,1)
        feats = feats.reshape(-1,feats.shape[-1])
        #============================================
        feats_mean = feats.mean(dim=(0))
        feats_std = feats.std(dim=(0))
        feats_std = feats_std  + (feats_std==0).float()
        print( feats_mean.shape,feats_std.shape)
        # import ipdb;ipdb.set_trace();#ipdb.post_mortem()
        #============================================        
        X = (feats-feats_mean[None,:])/feats_std[None,:]
        # Y = 
        # w,b = solve_regression(X,y,lambda_ = 0)
        if self.use_pca:
            X = self.pca_obj.fit_transform(X)
        w,b = solve_standard_regression(X,y,device)
        self.w,self.b = w,b
        self.feats_mean,self.feats_std = feats_mean,feats_std
        self.solved= True
        

    def cnn_forward_pass_(self,normalized_augmentations,inference_mode=True):
        # batch_size = self.batch_size
        n_chunks = (np.prod([normalized_augmentations.shape[0],*normalized_augmentations.shape[-2:]]) + self.chunk_size - 1)//self.chunk_size
        batch_size = (normalized_augmentations.shape[0] + n_chunks - 1)//n_chunks
        
        n_batches = (normalized_augmentations.shape[0] + batch_size - 1) // batch_size
        feats_bchw = []
        scores = []
        for bi in range(n_batches):
            normalized_augmentationsi = normalized_augmentations[bi*batch_size : (bi+1)*batch_size]
            output_and_featuresi  = self.hooked_cnn(normalized_augmentationsi,inference_mode=inference_mode)
            scoresi = output_and_featuresi[0]
            featuresi0 = output_and_featuresi[1:]
            assert len(featuresi0) == 1,'not supported for concatenated features yet'
            #========================================
            featuresi = torch.cat(featuresi0,dim=1)
            feats_bchwi = featuresi
            feats_bchw.append(feats_bchwi)
            scores.append(scoresi)
        feats_bchw = torch.cat(feats_bchw,dim=0)
        scores = torch.cat(scores,dim=0)
        return feats_bchw,scores
    def solve_using_previous_forward(self,feats_bchw,scores,target_id,sample_weight=None):
        device = feats_bchw.device
        if not isinstance(feats_bchw,torch.Tensor):
            feats = torch.tensor(feats_bchw,device=device)
        else:
            feats = feats_bchw
        feats = feats.view(feats_bchw.shape[0],feats_bchw.shape[1],-1).detach()       
        if scores.ndim == 4:
            scores = scores.mean(dim=(-1,-2))     
        #============================================
        feats_mean = feats_bchw.mean(dim=(0,-1,-2))
        feats_std = feats_bchw.std(dim=(0,-1,-2))
        feats_std = feats_std  + (feats_std==0).float()
        print( feats_mean.shape,feats_std.shape)
        #============================================
        prob_full = torch.softmax(scores,dim=-1)
        score = scores[:, target_id]
        prob = prob_full[:, target_id]
        #============================================
        X = (feats-feats_mean[None,:,None])/feats_std[None,:,None]
        Y = score
        #============================================
        if False:
            w,b = solve_shared_regression(X,Y,lambda_ = 0)
        elif True:
            # import ipdb; ipdb.set_trace()
            w,b = solve_standard_regression(X.sum(dim = -1),
                                            torch.atleast_1d(Y.squeeze()),
            # alpha=0.001,
            alpha=0.001*10,
            device=device)
            b = b/X.shape[-1]

        self.w,self.b = w,b
        self.feats_mean,self.feats_std = feats_mean,feats_std
        self.solved= True

    def solve(self,normalized_augmentations,target_id):
        device = normalized_augmentations.device
        feats_bchw, scores= self.cnn_forward_pass_(normalized_augmentations)
        # import ipdb;ipdb.set_trace()
        self.solve_using_previous_forward(feats_bchw,scores,target_id)
        """
        feats = torch.tensor(feats_bchw,device=device).view(feats_bchw.shape[0],feats_bchw.shape[1],-1).detach()            
        #============================================
        feats_mean = feats_bchw.mean(dim=(0,-1,-2))
        feats_std = feats_bchw.std(dim=(0,-1,-2))
        feats_std = feats_std  + (feats_std==0).float()
        print( feats_mean.shape,feats_std.shape)
        #============================================
        prob_full = torch.softmax(scores,dim=-1)
        score = scores[:, target_id]
        prob = prob_full[:, target_id]
        #============================================
        X = (feats-feats_mean[None,:,None])/feats_std[None,:,None]
        Y = score
        #============================================
        w,b = solve_shared_regression(X,Y,lambda_ = 0)
        self.w,self.b = w,b
        self.feats_mean,self.feats_std = feats_mean,feats_std
        self.solved= True
        """
    def get_raw_cam_from_feats(self,feats_bchw,clip=True):
        device= feats_bchw.device
        if not isinstance(feats_bchw,torch.Tensor):
            feats = torch.tensor(feats_bchw,device=device)    
        else:
            feats = feats_bchw
        # feats = feats.view(feats_bchw.shape[0],feats_bchw.shape[1],-1).detach() 
        feats = feats_bchw
        #============================================            
        X = (feats-self.feats_mean[None,:,None,None])/self.feats_std[None,:,None,None]
        # import ipdb;ipdb.set_trace()
        if self.use_pca:
            assert False,'not implemented'
            X1 = X.permute(0,2,1).reshape(-1,X.shape[1])
            X1 = self.pca_obj.transform(X1)
            X = X1.reshape(X.shape[0],X.shape[2],-1).permute(0,2,1)
        # import ipdb;ipdb.set_trace()
        
        # cam = torch.einsum('ijkl,j->ikl',X,self.w).detach().cpu().numpy() + self.b.item()/np.prod(X.shape[-2:])
        if True:
            cam = torch.einsum('ijkl,j->ikl',X,self.w).detach().cpu().numpy() + 0*self.b.item()
        else:
            regr = self.w
            X0 = X
            X = X0.permute(0,2,3,1).reshape(-1,X0.shape[1])
            X_ = X.cpu().numpy()
            cam_ = regr.predict(X_)
            cam_ = cam_.reshape((X0.shape[0],X0.shape[2],X0.shape[3],1))
            # cam = torch.tensor(cam_,device=device)
            cam = np.transpose(cam_,(0,3,1,2))
        cam = cam.reshape(cam.shape[0],feats_bchw.shape[2],feats_bchw.shape[3])
        if clip:
            cam = cam.clip(0)
        return cam        
    def get_raw_cam(self,normalized_augmentations,clip=True) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        device =  normalized_augmentations.device
        device = normalized_augmentations.device
        output_and_features  = self.hooked_cnn(normalized_augmentations)
        scores = output_and_features[0]
        probs = torch.softmax(scores,dim=1)
        features0 = output_and_features[1:]
        assert len(features0) == 1,'not supported for concatenated features yet'
        features = torch.cat(features0,dim=1)
        feats_bchw = features
        cam = self.get_raw_cam_from_feats(feats_bchw,clip=clip)
        """
        feats = torch.tensor(feats_bchw,device=device).view(feats_bchw.shape[0],feats_bchw.shape[1],-1).detach()            
        #============================================            
        X = (feats-self.feats_mean[None,:,None])/self.feats_std[None,:,None]
        # import ipdb;ipdb.set_trace()
        if self.use_pca:
            X1 = X.permute(0,2,1).reshape(-1,X.shape[1])
            X1 = self.pca_obj.transform(X1)
            X = X1.reshape(X.shape[0],X.shape[2],-1).permute(0,2,1)
        # import ipdb;ipdb.set_trace()
        cam = torch.einsum('ijk,j->ik',X,self.w).detach().cpu().numpy() + self.b.item()
        cam = cam.reshape(cam.shape[0],feats_bchw.shape[2],feats_bchw.shape[3])
        if clip:
            cam = cam.clip(0)
        """
        return cam,scores,probs
    def __call__(self,normalized_augmentations):
        cam,scores,probs = self.get_raw_cam(normalized_augmentations)
        if not isinstance(cam,torch.Tensor):
            cam = torch.tensor(cam)    
        cam = cam[:,None]
        cam = torch.nn.functional.interpolate(cam,size=normalized_augmentations.shape[-2:],mode='bilinear',align_corners=False).numpy()
        return cam,scores,probs
        




class MyCAM_multiscale():




    def __init__(self,cnn,dataset):
        # self.CAM_CLASSES = []
        self.NORMALIZE = True
        self.dataset = dataset
        mycam_.create_subcams(self,cnn,dataset)
        # self.CAM_CLASSES.append(self.myCAM_CLASS2)
        self.RESIZE = None
    def solve(self,normalized_augmentations,target_id):
        self.target_id = target_id
        if os.environ.get('SCALE_TEST',False) == '1':
            from scale_test import scale_test
            scale_probs = scale_test(
                self.CAM_CLASSES[0].hooked_cnn,
                normalized_augmentations,
                target_id,
                    NAUG = 100,
                    START=0)
            self.RESIZE = max(scale_probs,key=scale_probs.get)
        # import ipdb;ipdb.set_trace()
        # normalized_augmentations = normalized_augmentations[:1]
        if self.__dict__.get('RESIZE',None) != None:
            normalized_augmentations = torch.nn.functional.interpolate(normalized_augmentations,(self.RESIZE,self.RESIZE),mode='bilinear',align_corners=False)
        
        # import ipdb;ipdb.set_trace()
        if True:
            cam_224,scores,probs = mycam_.solve_stable(self,normalized_augmentations,target_id,self.dataset)
            # import ipdb;ipdb.set_trace()
            if os.environ.get('DBG_DIFFICULT_IMAGES',False) == '1':
                # import IPython;IPython.embed()
                #============================================+
                # use torchvision utils to plot the augmentations in a grid
                import torchvision
                im_grid = torchvision.utils.make_grid(normalized_augmentations,value_range = (-128,128),nrow= int(np.sqrt(normalized_augmentations.shape[0])) )
                dutils.img_save(im_grid,'augmentations.png')
                print(scores[:,target_id].reshape((int(np.sqrt(normalized_augmentations.shape[0])),-1)))
                #============================================+
                pass
            dutils.cipdb('DBG_DIFFICULT_IMAGES')
            dutils.cipdb('DBG_VIZ_AUG')

            return cam_224,scores,probs
        elif False:
            return mycam_.solve_logistic(self,normalized_augmentations,target_id)

    # """
    def __call__(self,normalized_augmentations):
        if os.environ.get('SCALE_AND_CROP',False) == '1':
            from scale_and_crop_forward import scale_and_crop_forward
            return scale_and_crop_forward(self,normalized_augmentations)
        """
        device = normalized_augmentations.device
        last_CAM_CLASS = self.CAM_CLASSES[-1]
        # import ipdb;ipdb.set_trace()
        return last_CAM_CLASS(normalized_augmentations)
        """
        # roll by a small amount
        # normalized_augmentations = torch.roll(normalized_augmentations,(-15,15),dims=(-1,-2))
        # dutils.img_save(normalized_augmentations[0],'ref_rolled.png')
        #==================================================================================
        if self.__dict__.get('RESIZE',None) != None:
            normalized_augmentations = torch.nn.functional.interpolate(normalized_augmentations,(self.RESIZE,self.RESIZE),mode='bilinear',align_corners=False)
        import mycam_
        device = normalized_augmentations.device
        with torch.inference_mode():
            ignr_feats,scores = self.CAM_CLASSES[0].cnn_forward_pass_(normalized_augmentations)
        feat_map_sizes = []
        for CAM_CLASS in self.CAM_CLASSES:
            assert len(CAM_CLASS.feat_layers) == 1
            feat_layer = CAM_CLASS.feat_layers[0]
            feats = feat_layer.feats
            assert feats.ndim == 4
            feat_map_sizes.append(feats.shape[-2:])            
        probs = torch.softmax(scores,dim=1)
        all_cams = []
        for ci,CAM_CLASS in enumerate(self.CAM_CLASSES):
            list_of_feats_bchw = CAM_CLASS.get_last_feats_()
            feats_bchw = torch.cat(list_of_feats_bchw,dim=1)
            cam =  CAM_CLASS.get_raw_cam_from_feats(feats_bchw,clip=True)
            if self.NORMALIZE:
                # cam = set_max_as_1(cam)
                cam = mycam_.set_group_max_as_1(cam)  
            all_cams.append(cam)
        cam_as_gt = mycam_.get_fused_cams(all_cams,feat_map_sizes[-1],self.NORMALIZE,device=device)
        cam_224 = torch.nn.functional.interpolate(cam_as_gt,(224,224),mode = 'bilinear')
        cam_224 = tensor_to_numpy(cam_224)
        #==================================================================================
        # cam_224 = np.roll(cam_224,(15,-15),axis=(-1,-2))
        dutils.cipdb('DBG_VIZ_AUG')
        return cam_224,scores,probs
                    
