import torch
import skimage.io
import numpy as np
import os
from model.my_gpnn import extract_patches
print('TODO: recheck the test flow')
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
#==============================================
from matplotlib import pyplot as plt
'''
print('TODO:unfinished, should be repurposed to save_plot')
def save_fig():
    plt.figure()
    plt.imshow()
    plt.title(TODO.title)        
    plt.savefig(TODO.figname)
    plt.close()
'''
#==============================================
def save_imtensor(t,filename):
    # t = t[0].permute(1,2,0)
    t = t[0]
    im = tensor_to_numpy(t)
    skimage.io.imsave(filename,im)
    pass


def convert_I_to_YX(I,img_shape,patch_size):
    device = I.device
    NROWS = img_shape[0] - patch_size[0]
    NCOLS = img_shape[1] - patch_size[1]
    YX = torch.zeros(I.shape[0],2).long().to(device)
    YX[:,0] = I//NCOLS
    YX[:,1] = I%NCOLS
    return YX
def convert_YX_to_I(YX,img_shape,patch_size):
    device = YX.device
    NROWS = img_shape[0] - patch_size[0]
    NCOLS = img_shape[1] - patch_size[1]
    I = torch.zeros(YX.shape[0]).long().to(device)
    X,Y = YX[:,-1],YX[:,0]
    I = Y*NROWS + X
    return I

def jitter_patches(I,img_shape,patch_size,radius = 50):
    #TODO.check_with_radius_0
    device = I.device
    YX = convert_I_to_YX(I,img_shape,patch_size)
    low = -radius
    high = radius + 1
    changeX = torch.randint(low=low,high=high,size=I.shape[:1],device=device)
    changeY = torch.randint(low=low,high=high,size=I.shape[:1],device=device)
    newY = YX[:,0] + changeY
    newX = YX[:,1] + changeX
    newY,newX = newY.abs(),newX.abs()
    assert len(img_shape ) == 2
    newY,newX = newY.clamp(None,img_shape[0]-1-2*(patch_size[0]//2)),newX.clamp(None,img_shape[1]-1 - 2*(patch_size[1]//2))
    
    newYX = torch.stack([newY,newX],dim=-1)
    newI = convert_YX_to_I(newYX,img_shape,patch_size)
    return newI

def get_image(original_imname,target_size = 256,device=None):
    im = skimage.io.imread(original_imname)
    if im.ndim == 3 and im.shape[-1] > 3:
        im = im[...,:-1]
    overshoot = min(im.shape[:2])/target_size
    resize_aspect_ratio = 1/overshoot
    print(im.shape)
    im = skimage.transform.rescale(im,(resize_aspect_ratio,resize_aspect_ratio,1) if im.ndim == 3 else (resize_aspect_ratio,resize_aspect_ratio))
    im = im[-256:,-256:];print('forcibly cropping image')    
    ref = torch.tensor(im[None,...]).to(device)
    return ref
def test():
    device = 'cpu'
    out_dir = 'test-jitter'
    original_imname = 'images/cars.png'
    
    target_size = 256
    patch_size = (7,7)
    stride = 1
    jitter_radius = 5; print('checking with jitter_radius 0')
    # jitter_radius = 50
    ref = get_image(original_imname,target_size = target_size,device=device)
    assert ref.shape[-1] in [1,3]
    img_shape = ref.shape[1:3]
    patches = extract_patches(ref, patch_size, stride)
    
    I = torch.arange(np.prod([ref.shape[1] - 2*(patch_size[0]//2),ref.shape[2] - 2*(patch_size[1]//2)]),device=device).long()[:,None]
    D = torch.zeros(I.shape[0])
    '''
    combine_patches(v, patch_size, stride, x_scaled.shape[1:3]+(3,),as_np=False,
                patch_aggregation=self.PATCH_AGGREGATION,
                distances=d,I=I)
    '''
    from aggregation import combine_patches
    aggregation_results =  combine_patches(patches, patch_size, stride, 
                                    ref.shape[1:3]+(3,),as_np=False,patch_aggregation='uniform',
                                    distances=D,I=I)        
    ref_combined = aggregation_results['combined'].unsqueeze(0)
    assert ref_combined.shape == ref.shape
    assert torch.allclose(ref_combined,ref)
    print('STAGE:ref_combined')
    jitteredI = jitter_patches(I.squeeze(1),img_shape,patch_size,radius = jitter_radius)
    jitteredI = jitteredI.unsqueeze(1)
    jittered_patches = patches[jitteredI.T][0]
    aggregation_results =  combine_patches(jittered_patches, patch_size, stride, 
                                    ref.shape[1:3]+(3,),as_np=False,patch_aggregation='uniform',
                                    distances=D,I=jitteredI)        
    jittered = aggregation_results['combined'].unsqueeze(0)
    if jitter_radius == 0:
        assert torch.allclose(jittered,ref)
    # os.system(f'rm -rf {out_dir}')
    os.makedirs(out_dir,exist_ok=True)
    save_imtensor(ref,os.path.join(out_dir,'original.png'))
    save_imtensor(ref_combined,os.path.join(out_dir,'original_recombined.png'))
    save_imtensor(jittered,os.path.join(out_dir,'jittered.png'))

if __name__ == '__main__':
    test()
#*****************************************************************