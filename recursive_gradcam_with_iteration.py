import torch
def calculate_cam(feats,grads,spatial_avg=True):
    if spatial_avg:
        w = grads.mean(dim=(-1,-2),keepdim=True)
    else:
        w = grads
    dot = (feats*w).sum(dim=1,keepdim=True)
    cam = torch.nn.functional.relu(dot)
    return cam,w
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

class RecursiveGradCAM():
    def __init__(self,cnn,subnetworks):
        self.subnetworks = subnetworks
        for subnet in self.subnetworks:
            subnet[-1].register_forward_hook(get_hook())
            # subnet[-1].register_backward_hook(get_bhook())
        self.cnn = cnn
        pass
    def __call__(self,normalized_augmentations,target_id):
        # target_id = 14
        device = normalized_augmentations.device
        with torch.inference_mode():
            initial_out = self.cnn(normalized_augmentations)
        if initial_out.ndim == 4:
            initial_out = initial_out.mean(dim=(-1,-2))
        n_classes = initial_out.shape[1]
        last_cam = torch.zeros(normalized_augmentations.shape[0],n_classes,1,1,device=device)
        last_cam[:,target_id,:,:] = 1
        last_cam_bool = last_cam.detach().clone()
        last_w = torch.ones(last_cam.shape[:2],device=device).float()
        subnetworks_reversed = list(reversed(self.subnetworks))
        inputs = [l[-1].feats.detach().clone() for l in subnetworks_reversed]
        dutils.img_save(normalized_augmentations[0,0],'aug.png')
        for fi,feat_layer in enumerate(subnetworks_reversed[:-1]):
            cam0 = torch.ones_like(inputs[fi+1])[:,:1]
            running_inner_cam = cam0
            n_inner_iters = 1000 if fi ==0 else 1
            for inner_i in range(n_inner_iters):
                last_input0 = inputs[fi+1].requires_grad_(True)
                # print(last_input0.sum())
                last_input = last_input0 * (running_inner_cam/running_inner_cam.max()).detach()
                last_input.retain_grad()
                output = feat_layer(last_input)
                if output.ndim == 2:
                    output = output[:,:,None,None]
                # print(output.shape)
                # loss = (last_cam.detach() * output).sum()
                # loss = (last_cam_bool.detach() * (last_w.detach()*output)).sum()
                loss = (last_cam.detach() * (last_w.detach()*output)).sum()
                print(loss)
                loss.backward()
                grads = last_input.grad.detach()
                # print(grads.sum())
                feats = last_input.detach()
                last_cam_up = torch.ones_like(feats)
                last_cam_up_bool = torch.ones_like(feats)
                if fi > 0:
                    last_cam_up = torch.nn.functional.interpolate((last_cam>0).float(),feats.shape[-2:],mode='bilinear')
                    last_cam_up_bool = (last_cam_up >0).float()

                # cam,w = calculate_cam(feats*last_cam_up_bool,grads*last_cam_up_bool)
                cam,w = calculate_cam(feats*last_cam_up,grads*last_cam_up)
                
                # cam,w = calculate_cam(feats*last_cam_up,grads*last_cam_up,spatial_avg=fi==0)
                # cam,w = calculate_cam(feats*last_cam_up,grads*last_cam_up,spatial_avg=False)
                
                # cam,w = calculate_cam(feats*running_cam,grads*running_cam)
                cam = cam/cam.max()
                if inner_i == 0:
                    running_inner_cam = cam.max() * torch.ones_like(cam) * (0.99) + 0.01 * cam
                else:
                    running_inner_cam = running_inner_cam * (0.99) + 0.01 * cam
            last_cam_bool = (cam > 0).float()
            last_cam = cam
            last_w = w
            dutils.img_save(cam[0,0],'cam.png')
            import ipdb;ipdb.set_trace()    
        
        return last_cam
    pass
from torch import nn
def get_layer(l,model):
    return model.features.__dict__[l]
def split_network(vgg16,split_at):
    layers = list(vgg16.features.children())
    subnetworks = []
    for i in range(len(split_at)):
        if i == 0:
            subnetworks.append(nn.Sequential(*layers[:split_at[i]]))
        else:
            subnetworks.append(nn.Sequential(*layers[split_at[i-1]:split_at[i]]))
    # subnetworks.append(nn.Sequential(*layers[split_at[i]:]))
    # subnetworks.append(nn.Sequential(*layers[split_at[i]:]))
    subnetworks.append(
        nn.Sequential(vgg16.avgpool,
                      vgg16.classifier)
    )
    return subnetworks