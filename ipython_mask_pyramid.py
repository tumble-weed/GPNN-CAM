import model.mask_pyramid
if True:
    import torch
    R = 4/3
    PATCH_SIZE = 3
    STRIDE = 1
    mask = torch.zeros(32,32)
    mask[::2,::2] = 1
    mp = model.mask_pyramid.mask_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
if True:
    import torch
    R = 4/3
    PATCH_SIZE = 3
    STRIDE = 1
    mask = torch.zeros(32,32)
    mask[::2,::2] = 1
    mp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
if True:
    import torch
    R = 4/3
    PATCH_SIZE = 3
    STRIDE = 1
    mask = torch.zeros(1,1,32,32)
    mask[::2,::2] = 1
    mp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
if True:
    import torch
    import model.mask_pyramid
    import dutils
    R = 4/3
    PATCH_SIZE = 3
    STRIDE = 1
    mask = torch.zeros(1,1,32,32)
    mask[::2,::2] = 1
    mp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32)
    mask[::2,::2] = 1
    mp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32)
    mask[::2,::2] = 1
    mp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
mp[0].shape
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32)
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    assert mp[0].shape == mask.shape[-2:]
    dutils.img_save(tensor_to_numpy(mp[0][0,0]),'explore_mask.png')
    
mp[0].shape
#[Out]# torch.Size([900, 1, 3, 3])
mk[0].shape
kp.shape
kp[0].shape
#[Out]# torch.Size([900, 1])
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = (mi - (pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    assert mp[0].shape == mask.shape[-2:]
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(30,30)),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = (mi - (pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(30,30)),'explore_mask.png')
    
mshape
#[Out]# <generator object <genexpr> at 0x7fa1e7689a80>
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - (pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(30,30)),'explore_mask.png')
    
mshape
#[Out]# [31, 31]
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(30,30)),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(30,30)),'explore_mask.png')
    
kp[0].shape
#[Out]# torch.Size([900, 1])
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[-1][:,0].reshape(mshape)),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(mshape)),'explore_mask.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(22,22)),'explore_mask1.png')
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(22,22)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (3,3)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(22,22)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (2,2)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(mshape)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(22,22)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
import math
math.sqrt(961)
#[Out]# 31.0
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (2,2)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(31,31)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(22,22)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
math.sqrt(529)
#[Out]# 23.0
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (2,2)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(31,31)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(23,23)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (1,1)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(31,31)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(23,23)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (1,1)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(32,32)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(23,23)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
math.sqrt(576)
#[Out]# 24.0
if True:
    import torch
    import model.mask_pyramid
    import dutils
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    R = 4/3
    PATCH_SIZE = (1,1)
    STRIDE = 1
    mask = torch.zeros(1,1,32,32).float()
    mshape = list(mi - 2*(pi//2) for mi,pi in zip(mask.shape[-2:],PATCH_SIZE))
    mask[...,::2,::2] = 1
    mp,qp,kp = model.mask_pyramid.create_pyramid(mask,R,PATCH_SIZE,STRIDE,3)
    #assert mp[0].shape == mshape
    dutils.img_save(tensor_to_numpy(kp[0][:,0].reshape(32,32)),'explore_mask0.png')
    dutils.img_save(tensor_to_numpy(kp[1][:,0].reshape(24,24)),'explore_mask1.png')
    print(kp[0].sum())
    print(kp[1].sum())
    print(kp[2].sum())
    
24*24
#[Out]# 576
import torch.nn.functional as F
mask2 = F.interpolate(mask,size=(64,64), mode = 'bilinear')
mask2
#[Out]# tensor([[[[1.0000, 0.7500, 0.2500,  ..., 0.7500, 0.2500, 0.0000],
#[Out]#           [0.7500, 0.5625, 0.1875,  ..., 0.5625, 0.1875, 0.0000],
#[Out]#           [0.2500, 0.1875, 0.0625,  ..., 0.1875, 0.0625, 0.0000],
#[Out]#           ...,
#[Out]#           [0.7500, 0.5625, 0.1875,  ..., 0.5625, 0.1875, 0.0000],
#[Out]#           [0.2500, 0.1875, 0.0625,  ..., 0.1875, 0.0625, 0.0000],
#[Out]#           [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]]])
mask2[:10,:10]
#[Out]# tensor([[[[1.0000, 0.7500, 0.2500,  ..., 0.7500, 0.2500, 0.0000],
#[Out]#           [0.7500, 0.5625, 0.1875,  ..., 0.5625, 0.1875, 0.0000],
#[Out]#           [0.2500, 0.1875, 0.0625,  ..., 0.1875, 0.0625, 0.0000],
#[Out]#           ...,
#[Out]#           [0.7500, 0.5625, 0.1875,  ..., 0.5625, 0.1875, 0.0000],
#[Out]#           [0.2500, 0.1875, 0.0625,  ..., 0.1875, 0.0625, 0.0000],
#[Out]#           [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]]])
mask2nn = F.interpolate(mask,size=(64,64), mode = 'nearest')
mask2nn
#[Out]# tensor([[[[1., 1., 0.,  ..., 1., 0., 0.],
#[Out]#           [1., 1., 0.,  ..., 1., 0., 0.],
#[Out]#           [0., 0., 0.,  ..., 0., 0., 0.],
#[Out]#           ...,
#[Out]#           [1., 1., 0.,  ..., 1., 0., 0.],
#[Out]#           [0., 0., 0.,  ..., 0., 0., 0.],
#[Out]#           [0., 0., 0.,  ..., 0., 0., 0.]]]])
