import torch
def calculate_cam(feats,grads,spatial_avg=True,mask=None):
    if mask is None:
        mask = torch.ones_like(feats)[:,:1]
    if spatial_avg:
        w = (grads*mask).sum(dim=(-1,-2),keepdim=True)/mask.sum(dim=(-1,-2),keepdim=True)
    else:
        w = grads
    dot = (feats*w*mask).sum(dim=1,keepdim=True)
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
        dutils.img_save(normalized_augmentations,'aug.png')
        for fi,feat_layer in enumerate(subnetworks_reversed[:-1]):
            aggregate_cam = torch.zeros_like(inputs[fi+1][:,:1])            
            next_w = torch.zeros_like(inputs[fi+1])
            grad_mask_mass = 0
            average_mass = torch.ones_like(aggregate_cam)
            # mask = last_cam.detach()
            for li,locs in enumerate(last_cam.nonzero()):
                # import ipdb;ipdb.set_trace()
                print(li)
                mask = torch.zeros_like(last_cam)
                if False:
                    mask[locs[0],locs[1],locs[2],locs[3]] = last_cam[locs[0],locs[1],locs[2],locs[3]]
                else:
                    mask[locs[0],locs[1],locs[2],locs[3]] = last_cam_bool[locs[0],locs[1],locs[2],locs[3]]
                last_input = inputs[fi+1].requires_grad_(True)
                last_input.retain_grad()
                last_input.grad = None
                output = feat_layer(last_input)
                if output.ndim == 2:
                    output = output[:,:,None,None]
                print(output.shape)
                
                # loss = (mask*last_cam.detach() * output).sum()
                # loss = (last_cam_bool.detach() * (last_w.detach()*output)).sum()
                loss = (mask * (last_w.detach()*output)).sum()
                loss.backward()


                feats = last_input.detach()
                grads = last_input.grad.detach()
                
                #================================================================
                last_cam_up = torch.ones_like(feats)
                last_cam_up_bool = torch.ones_like(feats)
                if fi > 0:
                    last_cam_up = torch.nn.functional.interpolate((last_cam).float(),feats.shape[-2:],mode='bilinear')
                    last_cam_up_bool = (last_cam_up >0).float()
                #================================================================

                # cam,w = calculate_cam(feats*last_cam_up_bool,grads*last_cam_up_bool)
                # cam,w = calculate_cam(feats*last_cam_up,grads*last_cam_up)
                # cam,w = calculate_cam(feats*last_cam_up,grads*last_cam_up,spatial_avg=fi==0)
                # cam,w = calculate_cam(feats*last_cam_up,grads*last_cam_up,spatial_avg=False)
                
                # cam,w = calculate_cam(feats*running_cam,grads*running_cam)
                grad_mask = (grads!=0).any(dim=1,keepdim=True).float()
                cam,w = calculate_cam(feats,grads,mask=grad_mask)
                # next_w[locs[0],:,locs[2],locs[3]] = w[0,:,0,0]
                """
                next_w += (grad_mask * w)
                grad_mask_mass += grad_mask
                aggregate_cam += cam
                """
                """
                weighted_grad_mask = grad_mask * mask.sum()
                choose_new = (weighted_grad_mask > grad_mask_mass).float()
                aggregate_cam = choose_new * cam + (1-choose_new) * aggregate_cam
                next_w = choose_new * w + (1-choose_new)*next_w
                grad_mask_mass = choose_new * weighted_grad_mask + (1-choose_new) * grad_mask_mass
                """
                
                # choose_new = (cam > aggregate_cam)
                # choose_both_old_and_new = (cam == aggregate_cam)
                choose_both_old_and_new = (cam > 0) & (aggregate_cam > 0)
                choose_new = (cam > 0) & (aggregate_cam == 0)
                # aggregate_cam = torch.maximum(aggregate_cam,cam)
                aggregate_cam[choose_new] = cam[choose_new]
                aggregate_cam[choose_both_old_and_new] += cam[choose_both_old_and_new]
                """
                next_w[choose_new.repeat(1,next_w.shape[1],1,1)] = w[0,:,0,0]
                next_w[choose_both_old_and_new] += w[choose_both_old_and_new]
                """
                for ci in range(next_w.shape[1]):
                    next_w[:,ci:ci+1,:,:][choose_new] = w[:,ci,:,:].squeeze()
                    next_w[:,ci:ci+1,:,:][choose_both_old_and_new] += w[:,ci,:,:].squeeze()
                average_mass[choose_both_old_and_new] += 1
            if False:
                aggregate_cam = aggregate_cam/(grad_mask_mass + 1e-6)
                w = next_w/(grad_mask_mass + 1e-6)
            else:
                w = next_w/average_mass
                aggregate_cam = aggregate_cam/average_mass
                cam = aggregate_cam
                
            # cam  = (cam > 0).float()
            # cam = cam/cam.max()
            # cam = (cam == cam.max()).float()
            last_cam_bool = (cam > 0).float()
            last_cam = cam
            last_w = w
            dutils.img_save(cam,f'cam{fi}.png')
            # import ipdb;ipdb.set_trace()            
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