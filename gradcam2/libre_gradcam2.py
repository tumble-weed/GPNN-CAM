import numpy as np
import torch
import cv2
import torch.nn.functional as F
import dutils
import skimage.transform
import os
TODO = None
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
class my_GradCAM_multi(object):
    activations = dict()
    gradients = dict()
    def __init__(self, model,  gradients_flag = False, g_flag = True, g_pp_flag = True,arg_target_layer=None):
        self.model = model

        self.g_flag = g_flag
        self.g_pp_flag = g_pp_flag
        self.gradients_flag = gradients_flag
        self.activations =  self.__class__.activations
        self.gradients = self.__class__.gradients
        '''
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        '''
        model.eval()
        # import ipdb;ipdb.set_trace()
        print('check if modelname can be derived from model object')
        print('check what happens when you arrive here the second time')

        if not model.__dict__.get('gradcam_hooked',False):
        # if True:
            # self.gradients = dict()
            # self.activations = dict()
            if 'vgg' in str(model.__class__).lower():
                if 'sanity_model' in model.__dict__:
                        # if modelname == 'vgg19':
                        arg_target_layer = 36
                        

                        # last_spatial_layer = model.features._modules['36']
                        # pass
                        target_layer = model.features[int(arg_target_layer)]
                        layer_path = int(arg_target_layer)
                elif 'dataset' in model.__dict__ and model.dataset == 'voc':
                    # https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/examples/attribution_benchmark.py#L96
                    arg_target_layer = 29  # relu before pool5
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
                else:
                    if arg_target_layer is None:
                        arg_target_layer = 43
                    # if os.environ.get('DBG_REGRESSION',False) == '1':
                    #     arg_target_layer = 33
                        # import ipdb; ipdb.set_trace()
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
            elif 'resnet' in str(model.__class__).lower():
                if 'dataset' in model.__dict__ and model.dataset == 'voc':
                    #https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/examples/attribution_benchmark.py#L100
                    arg_target_layer = 4
                    target_layer = model.layer4  # 'layer3.5'  # res4a
                    layer_path = 'layer4'  # pool before fc 
                
                else:
                    arg_target_layer = 4
                    target_layer = model.layer4
                    layer_path = 'layer4'

            def forward_hook(module, input, output):
                self.activations['value'] = tensor_to_numpy(output)
                return None

            def backward_hook(module,input,output):
                self.gradients['value'] = tensor_to_numpy(output[0])

            self.fwd_handle = target_layer.register_forward_hook(forward_hook)
            self.bwd_handle = target_layer.register_backward_hook(backward_hook)
            model.gradcam_hooked = True

    # def forward(self, input, class_idx=None, retain_graph=False):
    #     """
    #     Args:
    #         input: input image with shape of (1, 3, H, W)
    #         class_idx (int): class index for calculating GradCAM.
    #                 If not specified, the class index that makes the highest model prediction score will be used.
    #     Return:
    #         mask: saliency map of the same spatial dimension with input
    #         logit: model output
    #     """
    #     logit = self.model(input)
    #     if class_idx is None:
    #         score = logit[:, logit.max(1)[-1]].sum()
    #     else:
    #         score = logit[:, class_idx].sum()
    #     self.model.zero_grad()
    #     score.backward(retain_graph=True)
    #     a = self.activations['value'].detach().cpu().numpy()
    #     if self.g_pp_flag == False and self.g_flag == False and self.gradients_flag == False:
    #         return a[0]
    #     g = self.gradients['value'].detach().cpu().numpy()
    #     a = np.transpose(a[0], (1, 2, 0)).copy()
    #     g = np.transpose(g[0], (1, 2, 0)).copy()
    #     g_ = np.mean(g, axis=(0, 1), keepdims=True).copy()

    #     grad_cam, grad_cam_pp = 0,0

    #     if self.g_pp_flag:
    #         g_2 = g**2
    #         g_3 = g**3
    #         alpha_numer = g_2
    #         alpha_denom = 2*g_2 + np.sum(a*g_3,axis=(0,1),keepdims=True) #+ 1e-2

    #         alpha = alpha_numer/ alpha_denom

    #         w = np.sum(alpha*np.maximum(g,0),axis=(0,1),keepdims=True)

    #         grad_cam_pp = np.maximum(w * a,0)
    #         grad_cam_pp = np.sum(grad_cam_pp, axis=-1)
    #         grad_cam_pp = cv2.resize(grad_cam_pp, (224, 224))

    #     if self.g_flag:

    #         grad_cam =a*g_
    #         import ipdb;ipdb.set_trace()
    #         grad_cam = np.sum(np.maximum(grad_cam,0),axis=-1)
    #         grad_cam = cv2.resize(grad_cam,(224,224))

    #     if self.gradients_flag: return g_[0,0], a, grad_cam, grad_cam_pp
    #     else:                   return a, grad_cam, grad_cam_pp

    def forward(self, input, class_idx, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        device= input.device
        input = input.contiguous().requires_grad_(True)
        print('TODO:setting input to contiguous in forward')
        logit = self.model(input)
        # import ipdb;ipdb.set_trace()
        if class_idx is None:
            assert False,'shouldnt be coming to max class, class_idx should have been given'
            score = logit[:, logit.max(1)[-1]].sum()
        else:
            score = logit[:, class_idx].sum()
        scores = logit[:,class_idx]
        probs = torch.softmax(logit,dim=1)[:,class_idx]
        if scores.ndim == 3:
            scores = scores.mean(dim=(-1,-2))
        if probs.ndim == 3:
            probs = probs.mean(dim=(-1,-2))            
        assert scores.ndim == 1,'might not work for pascal VOC'
        # import ipdb;ipdb.set_trace()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        return self.after_backward(input,scores,probs, class_idx, retain_graph)
    def after_backward(self,input,scores,probs, class_idx, retain_graph):
        device = input.device
        if (os.environ.get('EARLY_FUSION',False) == '1') and dutils.aggregation_results is not None:
            # a_bchw
            if False:
                a = self.activations['value'].detach()
                g = self.gradients['value'].detach()
            else:
                a = torch.tensor(self.activations['value'],device=device)
                b = torch.tensor(self.gradients['value'],device=device)
            if 'float16' in str(a.dtype):
                a = a.float()
            if 'float16' in str(b.dtype):
                b = b.float()
                
            from saliency_for_gpnn_gradcam_multi import unpermute_map_old  as unpermute_map
            # import ipdb; ipdb.set_trace()
            unpermuted_a = np.zeros((1,g.shape[1],*input.shape[-2:]))
            unpermuted_g = np.zeros((1,g.shape[1],*input.shape[-2:]))
            batch_size_for_unpermute = 512
            n_batches = (a.shape[0] + batch_size_for_unpermute - 1)//batch_size_for_unpermute
            nsamples = g.shape[0]
            for ubi in range(n_batches):
                g_upi = torch.nn.functional.interpolate(g[:,ubi*batch_size_for_unpermute:(ubi+1)*batch_size_for_unpermute],size=input.shape[-2:],mode='bilinear',align_corners=False)
                a_upi = torch.nn.functional.interpolate(a[:,ubi*batch_size_for_unpermute:(ubi+1)*batch_size_for_unpermute],size=input.shape[-2:],mode='bilinear',align_corners=False)
                unpermuted_gi = unpermute_map(g_upi,dutils.aggregation_results,dutils.patch_size,permuted_shape='fit',indicators_shape=dutils.indicators_shape)
                unpermuted_ai = unpermute_map(a_upi,dutils.aggregation_results,dutils.patch_size,permuted_shape='fit',indicators_shape=dutils.indicators_shape)
                g = g.cpu().numpy()
                unpermuted_gi = unpermuted_gi.sum(dim=0,keepdim=True)/nsamples
                unpermuted_ai = unpermuted_ai.sum(dim=0,keepdim=True)/nsamples
                unpermuted_ai = unpermuted_ai.cpu().numpy()     
                unpermuted_gi = unpermuted_gi.cpu().numpy()     
                unpermuted_a[:,ubi*batch_size_for_unpermute:(ubi+1)*batch_size_for_unpermute] = unpermuted_ai
                unpermuted_g[:,ubi*batch_size_for_unpermute:(ubi+1)*batch_size_for_unpermute] = unpermuted_gi
            #=======================================================
            g = unpermuted_g
            a = unpermuted_a
            # g = unpermuted_g.mean(axis=0,keepdims=True)  
            # a = unpermuted_a.mean(axis=0,keepdims=True)  
            # g = unpermuted_g.mean(dim=0,keepdim=True)
            # a = unpermuted_a.mean(dim=0,keepdim=True)
            # g = g.cpu().numpy()
            # a = a.cpu().numpy()
            #=======================================================
            
            # import ipdb;ipdb.set_trace()
            a = np.transpose(a, (0,2, 3, 1)).copy()
            g = np.transpose(g, (0,2, 3, 1)).copy()
            g_ = np.mean(g, axis=(1, 2), keepdims=True).copy()
        else:
            if False:
                a = self.activations['value'].detach().cpu().numpy()
            else:
                a = self.activations['value']
            if self.g_pp_flag == False and self.g_flag == False and self.gradients_flag == False:
                assert False,'retaining batch dimension not corrected'
                return a[0]
            if False:
                g = self.gradients['value'].detach().cpu().numpy()
            else:
                g = self.gradients['value']
            if 'float16' in str(g.dtype):
                g = g.astype(np.float32)
            # import ipdb;ipdb.set_trace()
            if 'float16' in str(a.dtype):
                a = a.astype(np.float32)

            a = np.transpose(a, (0,2, 3, 1)).copy()
            g = np.transpose(g, (0,2, 3, 1)).copy()
            g_ = np.mean(g, axis=(1, 2), keepdims=True).copy()

        grad_cam, grad_cam_pp = 0,0

        if self.g_pp_flag:
            g_2 = g**2
            g_3 = g**3
            alpha_numer = g_2
            alpha_denom = 2*g_2 + np.sum(a*g_3,axis=(1,2),keepdims=True) #+ 1e-2
            to_check = alpha_numer[alpha_denom==0]
            # assert np.allclose(to_check,np.zeros(to_check.shape))
            if len(to_check) >0:
                assert np.abs(to_check - np.zeros(to_check.shape)).max() < 1e-3
            alpha = alpha_numer/ (alpha_denom + (alpha_denom ==0).astype(np.float32) )
            assert not np.isnan(alpha).any() and not np.isinf(alpha).any()
            w = np.sum(alpha*np.maximum(g,0),axis=(1,2),keepdims=True)

            grad_cam_pp = np.maximum(w * a,0)
            grad_cam_pp = np.sum(grad_cam_pp, axis=-1)
            # import ipdb; ipdb.set_trace()
            # grad_cam_pp = cv2.resize(grad_cam_pp, (224, 224))
            if False:
                grad_cam_pp = np.stack([cv2.resize(grad_cam_ppi,input.shape[-2:]) for grad_cam_ppi in grad_cam_pp],axis=0)
            else:
                grad_cam_pp = np.stack([skimage.transform.resize(grad_cam_ppi,input.shape[-2:]) for grad_cam_ppi in grad_cam_pp],axis=0)
            # import ipdb; ipdb.set_trace()

        if self.g_flag:
            # import ipdb;ipdb.set_trace()
            print('see if gradcams are normalized')
            grad_cam =a*g_
            grad_cam = np.sum(np.maximum(grad_cam,0),axis=-1)
            # grad_cam = np.stack([cv2.resize(grad_cami,input.shape[-2:]) for grad_cami in grad_cam],axis=0)
            # import ipdb; ipdb.set_trace()
            grad_cam = np.stack([skimage.transform.resize(grad_cami,input.shape[-2:]) for grad_cami in grad_cam],axis=0)

        # import ipdb;ipdb.set_trace()
        assert not np.isnan(grad_cam_pp).any()
        print('try visualizing the outputs, as well as the gradcam')
        #===================
        del self.activations['value']
        del self.gradients['value']
        #===================
        if self.gradients_flag: return g_[0,0], a, grad_cam, grad_cam_pp,scores,probs
        else:                   return a, grad_cam, grad_cam_pp,scores,probs

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

#----------------------------------------------------------------------------------
# SCORECAM
#----------------------------------------------------------------------------------
class my_ScoreCAM_multi(object):
    activations = dict()
    gradients = dict()
    def __init__(self, model,  
#gradients_flag = False, g_flag = True, g_pp_flag = True
):
        self.model = model

        #self.g_flag = g_flag
        #self.g_pp_flag = g_pp_flag
        #self.gradients_flag = gradients_flag
        self.activations =  self.__class__.activations
        self.gradients = self.__class__.gradients

        model.eval()
        print('check if modelname can be derived from model object')
        print('check what happens when you arrive here the second time')

        if not model.__dict__.get('hooked',False):
            # self.gradients = dict()
            # self.activations = dict()
            if 'vgg' in str(model.__class__).lower():
                if 'sanity_model' in model.__dict__:
                        # if modelname == 'vgg19':
                        arg_target_layer = 36
                        # last_spatial_layer = model.features._modules['36']
                        # pass
                        target_layer = model.features[int(arg_target_layer)]
                        layer_path = int(arg_target_layer)
                elif 'dataset' in model.__dict__ and model.dataset == 'voc':
                    # https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/examples/attribution_benchmark.py#L96
                    arg_target_layer = 29  # relu before pool5
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
                else:
                    arg_target_layer = 43
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
            elif 'resnet' in str(model.__class__).lower():
                if 'dataset' in model.__dict__ and model.dataset == 'voc':
                    #https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/examples/attribution_benchmark.py#L100
                    arg_target_layer = 4
                    target_layer = model.layer4  # 'layer3.5'  # res4a
                    layer_path = 'layer4'  # pool before fc 
                
                else:
                    arg_target_layer = 4
                    target_layer = model.layer4
                    layer_path = 'layer4'

            def forward_hook(module, input, output):
                self.activations['value'] = output
                return None

            def backward_hook(module,input,output):
                self.gradients['value'] = output[0]

            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
            model.hooked = True
            
    def forward(self,input,class_idx,retain_graph=False):
        device = input.device
        b, c, h, w = input.size()

        score_weight = []
        logit_weight = []
        where_bads = []
        # predication on raw input
        logit = self.model(input).cuda()
        assert class_idx is not None
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            scores = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            scores = logit[:, class_idx].squeeze()
        # import ipdb;ipdb.set_trace()
        assert logit.ndim in [2,4]
        logit = F.softmax(logit,dim=1).squeeze(-1).squeeze(-1)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          scores = scores.cuda()
          logit = logit.cuda()
        score = scores.sum()
        self.model.zero_grad()
        # import ipdb;ipdb.set_trace()
        # score.backward(retain_graph=retain_graph)
        import copy
        activations = copy.deepcopy(self.activations['value'].detach().clone())
        b, k, u, v = activations.size()
        if os.environ.get('DBG_SCORECAM_K',False):
            k = os.environ['DBG_SCORECAM_K']
            k = int(k)
        score_saliency_map = torch.zeros((input.shape[0], 1, h, w),device=device)

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()
            batch_size = 32
            n_batches =  (k + batch_size - 1)//batch_size
            with torch.no_grad():
                for bi in range(n_batches):
                    i = range(bi*batch_size,(bi+1)*batch_size)
                    #assert False
                    # upsampling
                    saliency_map = activations[0, i, :, :].unsqueeze(1)
                    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                    m = saliency_map.flatten(start_dim=1,end_dim=-1).min(dim=-1)[0]
                    M = saliency_map.flatten(start_dim=1,end_dim=-1).max(dim=-1)[0]
                    where_bad = M == m
                    assert saliency_map.min() >= 0
                    #   if :
                    #       score_weight.append(where_bad.float())
                    #       continue

                    # normalize to 0-1
                    denom = (M - m + where_bad.float())[:,None,None,None]
                    norm_saliency_map = (saliency_map - m[:,None,None,None]) / denom

                    # how much increase if keeping the highlighted region
                    # predication on masked input
                    maxs = norm_saliency_map.flatten(start_dim=1,end_dim=-1).max(dim=-1)[0]
                    # check if maxs only contain 0, 1
                    assert len(maxs.unique()) in [1,2]
                    assert maxs.max() in [0,1]
                    output = self.model(input * norm_saliency_map)
                    output = output.squeeze(-1).squeeze(-1)
                    logits = output
                    assert logit.ndim == 2
                    output = F.softmax(output,dim=1)
                    # scores = output[torch.arange(predicted_class.shape[0],device=device),predicted_class]
                    # import ipdb;ipdb.set_trace()
                    scores = output[:,predicted_class.item()]
                    scores = scores * (1 - where_bad.float())
                    #---------------------------------------------
                    logits = logits[:,predicted_class.item()]
                    # logits = logits * (1 - where_bad.float())
                    logit_weight.append(logits.detach().clone())
                    where_bads.append(where_bad)
                    #---------------------------------------------
                    self.scorecam_mode = 'paper';print(colorful.red(f'hardcoding scorecam mode to {self.scorecam_mode}'))
                    if self.scorecam_mode == 'github':
                        score_saliency_map += (scores[:,None,None,None] * saliency_map).sum(dim=0,keepdim=True)
                    elif self.scorecam_mode == 'normalized':
                        print(colorful.red('using norm saliency map in scorecam'))
                        score_saliency_map += (scores[:,None,None,None] * norm_saliency_map).sum(dim=0,keepdim=True)
                    else:
                        pass
                    score_weight.append(scores.detach().cpu().numpy())
                    # import ipdb;ipdb.set_trace()

        #===============================================================
        # from paper
        if self.scorecam_mode == 'paper':
            # import ipdb;ipdb.set_trace()
            logit_weight = torch.concatenate(logit_weight)
            where_bads = torch.concatenate(where_bads)
            
            # score_weight_paper = score_weight  
            alphas = torch.softmax(logit_weight[~where_bads],dim=0)
            # score_saliency_map = (alphas[:,None,None,None] * activations[0,~where_bads,...].unsqueeze(1)).sum(dim=0,keepdim=True)    
            a = activations[0,~where_bads,...].unsqueeze(1)
            score_saliency_map = 0
            for ii in range(len(alphas)):
                ai = F.interpolate(a[ii:ii+1], size=(h, w), mode='bilinear', align_corners=False)
                alphai = alphas[ii]
                score_saliency_map += alphai*ai
        #===============================================================
        score_saliency_map = F.relu(score_saliency_map)
        assert score_saliency_map.shape[0] == 1
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.flatten(start_dim=1,end_dim=-1).min(dim=-1)[0], score_saliency_map.flatten(start_dim=1,end_dim=-1).max(dim=-1)[0]

        if (score_saliency_map_min == score_saliency_map_max).all():
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min[:,None,None,None]).div(score_saliency_map_max[:,None,None,None] - score_saliency_map_min[:,None,None,None]).data

        return score_saliency_map, score_weight
        
    def forward_old(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        score_weight = []
        # predication on raw input
        logit = self.model(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

              if saliency_map.max() == saliency_map.min():
                  score_weight.append(0)
                  continue

              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model(input * norm_saliency_map)
              output = F.softmax(output)
              score = output[0][predicted_class]

              score_saliency_map += score * saliency_map
              score_weight.append(score.detach().cpu().numpy())

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map, score_weight
        

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

# #----------------------------------------------------------------------------------
# # AGF
# #----------------------------------------------------------------------------------
# def my_AGF(test_model,loaded_model,inputs,**kwargs):
#     model = loaded_model
#     model.eval()
    
#     if test_model =='vgg':
#         layer = 'layer43'
        
#     elif test_model == 'resnet':
#         layer = 'layer4'

    
#     temp = inputs[0].detach()
#     in_tensor = inputs.cuda()
#     output = model(in_tensor)
#     #######################################################################################
#     #SigCAM
#     AGF = model.AGF(**kwargs)[0, 0].data.cpu().numpy()
    
#     cam = AGF #AGF
#     cam = cam - np.min(cam)
#     cam = cam / np.max(cam)
#     #######################################################################################
#     return cam

#----------------------------------------------------------------------------------
# Relevance CAM
#----------------------------------------------------------------------------------


from gradcam2.LRP_util import CLRP
class my_RelevanceCAM(object):
    activations = dict()
    layer_path = {}
    def __init__(self, model):
        self.model = model
        self.activations =  self.__class__.activations
        model.eval()
        print('check if modelname can be derived from model object')
        print('check what happens when you arrive here the second time')
        
        if not model.__dict__.get('relevancecam_hooked',False):
            # self.gradients = dict()
            # self.activations = dict()
            
            if 'vgg' in str(model.__class__).lower():
                if 'sanity_model' in model.__dict__:
                        # if modelname == 'vgg19':
                        arg_target_layer = 36
                        # last_spatial_layer = model.features._modules['36']
                        # pass
                        target_layer = model.features[int(arg_target_layer)]
                        layer_path = int(arg_target_layer)
                elif 'dataset' in model.__dict__ and model.dataset == 'voc':
                    # https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/examples/attribution_benchmark.py#L96
                    arg_target_layer = 29  # relu before pool5
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
                else:
                    arg_target_layer = 43
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
            elif 'resnet' in str(model.__class__).lower():
                if 'dataset' in model.__dict__ and model.dataset == 'voc':
                    #https://github.com/facebookresearch/TorchRay/blob/6a198ee61d229360a3def590410378d2ed6f1f06/examples/attribution_benchmark.py#L100
                    arg_target_layer = 4
                    target_layer = model.layer4  # 'layer3.5'  # res4a
                    layer_path = 'layer4'  # pool before fc 
                
                else:
                    arg_target_layer = 4
                    target_layer = model.layer4
                    layer_path = 'layer4'

            def forward_hook(module, input, output):
                self.activations['value'] = tensor_to_numpy(output)
                return None

            # def backward_hook(module,input,output):
            #     self.gradients['value'] = output[0]

            target_layer.register_forward_hook(forward_hook)
            # target_layer.register_backward_hook(backward_hook)
            model.relevancecam_hooked = True            
            self.__class__.layer_path[str(model.__class__).lower()] = layer_path
            # import ipdb;ipdb.set_trace()
    def forward(self,input,class_idx,
                retain_graph=False):
        input = input.contiguous()
        output = self.model(input)
        return self.after_forward(input,output,class_idx=class_idx,
                retain_graph=retain_graph)
    def after_forward(self,input,output,class_idx,
                      retain_graph):
        if class_idx is None:
            assert False,'this is not tested for 4d output (with spatial dimensions)'
            maxindex = np.argmax(output.data.cpu().numpy())
        else:
            maxindex = class_idx

        #    maxindex = label
        # if output.ndim == 4:
        #     #NOTE: because CLRP may expect 4d output
        #     output = output.mean(dim=(-1,-2))
        # assert output.ndim==2,'might not work for pascal VOC'
        Tt, Tn = CLRP(output, maxindex)
        # import ipdb; ipdb.set_trace()
        posi_R = self.model.relprop(Tt,1,flag=self.layer_path[str(self.model.__class__).lower()]).data.cpu().numpy()
        nega_R = self.model.relprop(Tn,1,flag=self.layer_path[str(self.model.__class__).lower()]).data.cpu().numpy()
        posi_R = posi_R.astype(np.float32)
        nega_R = nega_R.astype(np.float32)
        R = posi_R - nega_R

        if True:
            R = posi_R - nega_R
            R = np.transpose(R,(0,2,3,1))
            r_weight = np.sum(R,axis=(1,2),keepdims=True)
            activation = self.activations['value']
            # import ipdb;ipdb.set_trace()
            if isinstance(activation,torch.Tensor):
                activation = activation.permute(0,2,3,1).detach().cpu().numpy()
            else:
                activation = np.transpose(activation,(0,2,3,1))
            R_cam = np.sum(activation * r_weight, axis=-1)
            R_cam = np.sum(activation * r_weight, axis=-1)
            R_cam = R_cam - np.min(R_cam)
            R_cam = R_cam / np.max(R_cam)
            print(R_cam.shape)
            # import ipdb;ipdb.set_trace()
            R_cam = np.array([cv2.resize(R_cami,input.shape[-2:]) for R_cami in R_cam])
            print(R_cam.shape)

        # if input.shape[0] == 1:
        #     def check():
        #         R = posi_R - nega_R
        #         R = np.transpose(R[0],(1,2,0))
        #         r_weight = np.sum(R,axis=(0,1),keepdims=True)
        #         # activation, grad_cam, grad_campp = CAM_CLASS(input, class_idx=maxindex)
        #         activation = self.activations['value']
        #         # import ipdb;ipdb.set_trace()
        #         activation = activation.permute(0,2,3,1)[0].detach().cpu().numpy()
        #         # score_map, _ = Score_CAM_class(input, class_idx=maxindex)
        #         # score_map = score_map.squeeze()
        #         # score_map = score_map.detach().cpu().numpy()

        #         R_cam = np.sum(activation * r_weight, axis=-1)

        #         # 0 ~ 1 normalization
        #         R_cam = R_cam - np.min(R_cam)
        #         R_cam = R_cam / np.max(R_cam)
        #         print('size?')
        #         # import ipdb;ipdb.set_trace()
        #         R_cam = cv2.resize(R_cam,input.shape[-2:])
        #         return R_cam
        #     R_cam2 = check()
        #     print(
        #         np.abs(R_cam2-R_cam[0]).mean()
        #           )
        #     import ipdb;ipdb.set_trace()
            
        # import dutils
        # dutils.cipdb('DBG_AVG_SALIENCY')
        del self.activations['value']
        assert not np.isnan(R_cam).any()
        return R_cam
    
    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
#---------------------------------------------------------------------------------
# Libra CAM
#---------------------------------------------------------------------------------
if False:
    class my_LibraCAM():
        activations = dict()
        gradients = dict()
        def __init__(self,model):
            self.model = model
            self.activations =  self.__class__.activations
            self.ref_cnt = 30
            model.eval()
            print('check if modelname can be derived from model object')
            print('check what happens when you arrive here the second time')

            if not model.__dict__.get('hooked',False):
                import colorful;print(colorful.red('are the correct layers being reffered to?'))
                import time;time.sleep()
                # self.gradients = dict()
                # self.activations = dict()
                if 'vgg' in str(model.__class__).lower():
                    arg_target_layer = 43
                    target_layer = model.features[int(arg_target_layer)]
                    layer_path = int(arg_target_layer)
                elif 'resnet' in str(model.__class__).lower():
                    arg_target_layer = 4
                    target_layer = model.layer4
                    layer_path = 'layer4'

                def forward_hook(module, input, output):
                    self.activations['value'] = output
                    return None

                # def backward_hook(module,input,output):
                #     self.gradients['value'] = output[0]

                target_layer.register_forward_hook(forward_hook)
                # target_layer.register_backward_hook(backward_hook)
                model.hooked = True            
                self.__class__.layer_path = layer_path        
            pass
        def forward(self,input,class_idx=None,
                    retain_graph=False):
            image_tensor = input
            # image_tensor = image_tensor.cuda() 
            logits = model(image_tensor)
            org_target_class_proba = TODO
            dutils.break_only_once(why='check how to set or_target_class_proba') 
            
            score = logits.max()
            model.zero_grad()
            score.backward() 


            relu = torch.nn.ReLU()
            target_activation = self.activations['value']
            target_grad = self.gradients['value']
            
            size_upsample = input.shape[-2:] #setting image size manually -> ImageNet
            output_cam = []
            dutils.break_only_once()
            dutils.break_if('STOP_LIBRE')
            label = class_idx
            ref_libra = self.ref_library[label]
            for i in range(0,self.ref_cnt):

                ref_activation = ref_libra[i]['activation']
                coeff = target_activation - torch.tensor(ref_activation).cuda() #base
                cam0 = torch.sum(target_grad * coeff, dim = 0).detach().cpu().numpy()
                #normalization
                cam0 = cam0 - np.min(cam0)
                cam_img0 = cam0 / np.max(cam0)
                output_cam.append(cv2.resize(cam_img0, size_upsample))

            # forward_result.clear()
            # backward_result.clear()
            # forward_handle.remove()
            # backward_handle.remove()
            

            ####################################
            #rejection criterion
            filter_output_cam = torch.tensor(output_cam).unsqueeze(dim=1).cuda()
            candidates = filter_output_cam * image_tensor
            temp_logits = model(candidates)
            temp_proba = F.softmax(temp_logits)
            ref_target_class_proba = temp_proba[:,label]
            
            candidate_avgincrease =  ref_target_class_proba - org_target_class_proba #[30,1]
            
            l_cam = np.array(output_cam) #27 224, 224    

            ######################################################
            #Add (m, sigma)
            relu_candidate_avgincrease = relu(candidate_avgincrease)
            sampled_idx = (relu_candidate_avgincrease > 0).squeeze().detach().cpu().numpy() #sampled idx
            if sampled_idx.sum() != 0:
                relu_candidate_avgincrease = relu_candidate_avgincrease.squeeze().detach().cpu().numpy() #(30,)
                l_cam = l_cam[sampled_idx,:,:]
                relu_candidate_avgincrease = relu_candidate_avgincrease[sampled_idx]
                m = relu_candidate_avgincrease.mean()
                std = np.std(relu_candidate_avgincrease)
                
                threshold_filter = m - std
                filterd_idx = (relu_candidate_avgincrease >= threshold_filter)
                relu_candidate_avgincrease = relu_candidate_avgincrease[filterd_idx]
                l_cam = l_cam[filterd_idx]
            else:
                avgincrease = candidate_avgincrease.squeeze().detach().cpu().numpy()
                m = avgincrease.mean()
                std = np.std(avgincrease)
                threshold_filter = m + std
                
                filterd_idx = (avgincrease > threshold_filter)
                if filterd_idx.sum() !=0:
                    l_cam = l_cam[filterd_idx]
                #candidate_avgincrease

            l_cam = np.mean(l_cam,axis=0)
                
            del model, forward_result, backward_result
            
            return l_cam