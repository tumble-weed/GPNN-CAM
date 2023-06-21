import imagenet_classes
import torch
import torchvision
from typing import Dict, Callable
import copy

tensor_to_numpy = lambda t:t.detach().cpu().numpy()
#=====================================================
def get_target_id(class_name):
    target_id = [i for i,desc in imagenet_classes.classes.items() if class_name.lower() in desc.lower()]
    assert len(target_id) == 1
    target_id = target_id[0]
    return target_id
#=====================================================
features = {}
grads = {}
def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)
def get_forward_hook(i):
    #32
    def forward_hook(self,fin,fout):
        # 1,128,32,32
        features[i] = fout
        # features[32] = (1,128,32,32)
    return forward_hook

def get_backward_hook(i):
    #21
    def backward_hook(self,gin,gout):
        assert len(gout) == 1
        # gout = [(1,128,32,32)]
        grads[i] = gout[0]
        # grads[21] = 1,128,32,32
    return backward_hook
#=====================================================
def remove_inplace(model):
    info = enumerate(model.features.children())
    for i,l in info:
        if isinstance(l,torch.nn.ReLU):
            l.inplace = False
#=====================================================
def make_untrainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
#=====================================================
def init_cnn(cnn,layers):
    make_untrainable(cnn)
    cnn.eval()
    remove_inplace(cnn)
    # alexnet, [21]
    for i in layers:
        # 21
        h = get_forward_hook(i)
        # 21
        l = cnn.features[i]
        # features[21]
        print(f'registering forward hook for {l}')
        l.register_forward_hook(h)
    return features
#=====================================================
# def cnn_forward(cnn,t):
#     scores = cnn(t)
#     return scores,features
def get_class_similarity(model):
    w_class = model.classifier[-1].weight
    n_classes = w_class.shape[0]
    print(w_class.shape)
    w_class = w_class/w_class.norm(dim=-1,keepdim=True)
    # print(w_class.norm(dim=-1))
    dots = (w_class[:,None,:] * w_class[None,:,:]).sum(dim=-1)
    print(dots.max())
    # torch.arange(n_classes)[dots[370] > 0].shape
    return dots
class_similarity = []
def get_similar_class_ixs(class_similarity,class_ix,t_sim):
    sim = class_similarity[class_ix]
    sort_order = torch.argsort(sim)
    sorted_sim = sim[sort_order]
#     print(sorted_dots.shape)
    sim_class_ix = sort_order[sorted_sim >= t_sim]
    return sim_class_ix
#=====================================================

def cnn_forward(model,input,target_id,out={}):
    if True:
        if len(class_similarity) == 0:
            if False:
                class_similarity.append(get_class_similarity(model))
            else:
                print('skipping class similarity')
                class_similarity.append(None)

    full_scores = model(input)
    # 32,1000,2,4
    score_map = full_scores[:,target_id,...]
    # 32,2,4
    score = score_map
    if len(score_map.shape) == 3:
        # assert False
        score = score_map.mean(dim=(-1,-2))
    # 32,
    # assert False
    out['full_scores'] = full_scores
    out['score_map'] = score_map
    out['score'] = score
    out['features'] = features
    return out

###############################
# VGG transform
###############################
vgg_mean = (0.485, 0.456, 0.406)
vgg_std = (0.229, 0.224, 0.225)
def get_vgg_transform(size=224):
    vgg_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size),
            torchvision.transforms.Normalize(mean=vgg_mean,std=vgg_std),
            ]
        )
    return vgg_transform
def denormalize_vgg(t,vgg_mean=vgg_mean,vgg_std=vgg_std):
    device = t.device
    vgg_mean = torch.tensor(vgg_mean)[None,:,None,None].to(device)
    vgg_std = torch.tensor(vgg_std)[None,:,None,None].to(device)
    out = (t * vgg_std) +vgg_mean
    return out
    pass


def approximate_denormalize_voc(t):
    device = t.device
    t = (t - t.min())/(t.max() - t.min())
    t = 0.8 * t
    # vgg_mean = torch.tensor(vgg_mean)[None,:,None,None].to(device)
    # vgg_std = torch.tensor(vgg_std)[None,:,None,None].to(device)
    # out = (t * vgg_std) +vgg_mean
    return t
denormalize_vgg = approximate_denormalize_voc
    