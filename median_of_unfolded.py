import torch
import numpy as np
h,w = 12,13
ph,pw = 3,3
t1 = torch.zeros(1,1,h,w)
print(t1[0,0,h//2 - ph//2:h//2 + ph//2 + 1,w//2 - pw//2 : w//2 + pw//2 +1])
Y,X= torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
t1[0,0][Y.flatten(),X.flatten()]  = torch.arange(h*w).float()
print(t1.shape)
t1u = t1.unfold(2,ph,1).unfold(3,pw,1)
print(t1u.shape)
print(t1u[0,0,h//2,w//2])
# contains the final result from the median aggregation
out = np.inf *torch.ones((1,1,h+2*(ph//2),w + 2*(pw//2),ph,pw))
print('t1u will have 2*(ph//2), 2*(pw//2) lesser values')

if False and 'trying with single patch?':
        # out[:,:,ph//2:-(ph//2),pw//2:-(pw//2)] = t1u
        out[:,:,2*(ph//2):-2*(ph//2),2*(pw//2):-2*(pw//2)] = t1u
        YY,XX = torch.meshgrid(torch.arange(ph),torch.arange(pw),indexing='ij')
        at = 5,5
        v = out[0,0,YY + (at[0] - (ph//2)),XX  + (at[1] - (pw//2)),ph -1 - YY,pw -1 - XX]
        # I  = torch.stack([YY + (at[0] - (ph//2)),XX  + (at[1] - (pw//2)),ph - YY,pw - XX],dim=0)
if  False and 'trying with multiple patches':
    # place unfolded t1u inside out
    #out[:,:,ph//2:-(ph//2),pw//2:-(pw//2)] = t1u
    out[:,:,2*(ph//2):-2*(ph//2),2*(pw//2):-2*(pw//2)] = t1u
    YY,XX = torch.meshgrid(torch.arange(ph),torch.arange(pw),indexing='ij')
    At = torch.meshgrid(torch.arange(ph//2,h+(ph//2)),torch.arange(pw//2, w+(pw//2)),indexing='ij')
    I2 = YY[None,None,:,:] + (At[0][:,:,None,None] - (ph//2))
    I3 = XX[None,None,:,:]  + (At[1][:,:,None,None] - (pw//2))
    I4 = ph -1 - (torch.zeros_like(At[0])[:,:,None,None] +  YY[None,None,:,:])
    I5 = pw -1 - (torch.zeros_like(At[1])[:,:,None,None] +  XX[None,None,:,:])
    V = out[0,0,
            I2,I3,I4,I5]
    # sort the values in ascending order
    Vs = V.flatten(start_dim=-2,end_dim=-1)
    Vs =  Vs.sort(dim=-1)[0]
    #locate the median of valid values ( the ends will be padded with infinity)
    Vs2 = V.flatten(start_dim=-2,end_dim=-1)
    Vs2 = 1 - Vs2.isinf().float()
    Vs2 = Vs2.sum(dim=-1).long()
    median_at = Vs2//2
    '''
    # median_at = ((ph*pw)//2)*torch.ones(h-2*(ph//2),w-2*(pw//2))
    median_at = ((ph*pw)//2)*torch.ones(h-2,w-2)
    median_at = torch.conv2d(median_at[None,None,...],
                            torch.ones(1,1,ph+3,pw+3)/((ph+3)*(pw+3)),
                            stride=1,padding='same').ceil().long()
    # median_at =median_at.long()
    '''
    I1,I2 = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    # I1,I2 = torch.meshgrid(torch.arange(h-2),torch.arange(w-2),indexing='ij')
    # out2 = torch.zeros(h,w,ph*pw)
    # out2[1:-1,1:-1] = Vs
    # # out2[I1,I2,median_at[0,0]]
    # out2[I1,I2,median_at[0,0]]
    combined = Vs[I1,I2,median_at[0,0]]
    print(combined)
    assert combined.shape == t1.shape[-2:]

def embed_in_larger(t1u,output_shape,patch_size):
    device = t1.device
    h,w = output_shape
    ph,pw = patch_size
    out = np.inf *torch.ones((1,1,h+2*(ph//2),w + 2*(pw//2),ph,pw))
    out[:,:,2*(ph//2):-2*(ph//2),2*(pw//2):-2*(pw//2)] = t1u
    return out
def overlap_form(unfolded,output_shape,patch_size,sort_order=None):
    ph,pw = patch_size
    h,w = output_shape
    assert unfolded.shape[-4:] == (h+2*(ph//2),w + 2*(pw//2),ph,pw)
    YY,XX = torch.meshgrid(torch.arange(ph),torch.arange(pw),indexing='ij')
    # a patch-sized indexing window
    At = torch.meshgrid(torch.arange(ph//2,h+(ph//2)),torch.arange(pw//2, w+(pw//2)),indexing='ij')
    I2 = YY[None,None,:,:] + (At[0][:,:,None,None] - (ph//2))
    I3 = XX[None,None,:,:]  + (At[1][:,:,None,None] - (pw//2))
    I4 = ph -1 - (torch.zeros_like(At[0])[:,:,None,None] +  YY[None,None,:,:])
    I5 = pw -1 - (torch.zeros_like(At[1])[:,:,None,None] +  XX[None,None,:,:])
    V = unfolded[0,0,
            I2,I3,I4,I5]
    flat_overlap  = V.flatten(start_dim=-2,end_dim=-1)
    if sort_order is None:
        # sort the values in ascending order
        sorted_vals,sort_order =  flat_overlap.sort(dim=-1)
    else:
        print('check this')
        import pdb;pdb.set_trace()
        sorted_vals = torch.gather(flat_overlap,-1,sort_order)
    return sorted_vals,sort_order
def get_median(embedded,output_shape,patch_size):
    ph,pw = patch_size
    h,w = output_shape
    assert embedded.shape[-4:] == (h+2*(ph//2),w + 2*(pw//2),ph,pw)
    YY,XX = torch.meshgrid(torch.arange(ph),torch.arange(pw),indexing='ij')
    # a patch-sized indexing window
    At = torch.meshgrid(torch.arange(ph//2,h+(ph//2)),torch.arange(pw//2, w+(pw//2)),indexing='ij')
    I2 = YY[None,None,:,:] + (At[0][:,:,None,None] - (ph//2))
    I3 = XX[None,None,:,:]  + (At[1][:,:,None,None] - (pw//2))
    I4 = ph -1 - (torch.zeros_like(At[0])[:,:,None,None] +  YY[None,None,:,:])
    I5 = pw -1 - (torch.zeros_like(At[1])[:,:,None,None] +  XX[None,None,:,:])
    V =embedded[0,0,
            I2,I3,I4,I5]
    
    sorted_vals,sort_order = overlap_form(embedded,output_shape,patch_size,sort_order=None)
    #locate the median of valid values ( the ends will be padded with infinity)
    Vs2 = V.flatten(start_dim=-2,end_dim=-1)
    Vs2 = 1 - Vs2.isinf().float()
    Vs2 = Vs2.sum(dim=-1).long()
    median_at = Vs2//2
    '''
    # median_at = ((ph*pw)//2)*torch.ones(h-2*(ph//2),w-2*(pw//2))
    median_at = ((ph*pw)//2)*torch.ones(h-2,w-2)
    median_at = torch.conv2d(median_at[None,None,...],
                            torch.ones(1,1,ph+3,pw+3)/((ph+3)*(pw+3)),
                            stride=1,padding='same').ceil().long()
    # median_at =median_at.long()
    '''
    I1,I2 = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    # I1,I2 = torch.meshgrid(torch.arange(h-2),torch.arange(w-2),indexing='ij')
    # out2 = torch.zeros(h,w,ph*pw)
    # out2[1:-1,1:-1] = Vs
    # # out2[I1,I2,median_at[0,0]]
    # out2[I1,I2,median_at[0,0]]
    combined = sorted_vals[I1,I2,median_at[0,0]]
    print(combined)
    assert combined.shape == output_shape[-2:]
    return combined,median_at,sort_order
if __name__ == '__main__':
    if  True and 'refactor':
        out = embed_in_larger(t1u,(h,w),(ph,pw))
        combined,median_at,sort_order = get_median(out,(h,w),(ph,pw))
        # place unfolded t1u inside out
        #out[:,:,ph//2:-(ph//2),pw//2:-(pw//2)] = t1u
        #out[:,:,2*(ph//2):-2*(ph//2),2*(pw//2):-2*(pw//2)] = t1u
        print(combined)
