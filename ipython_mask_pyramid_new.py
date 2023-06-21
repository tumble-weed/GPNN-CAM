ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R)
device = self.mask.device
device = self.input_img_tensor.device
ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R)
ds1.shape
#[Out]# torch.Size([1, 3, 168, 223])
ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R)
ds1.shape
#[Out]# torch.Size([1, 3, 168, 223])
import dutils
dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),'blurred.png')
import os;print(os.getcwd())
ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**2)
dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),'blurred.png')
dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),'blurred2.png')
ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**1)
dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),'blurred2.png')
ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**1)
dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),'blurred.png')
ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor(self.mask,device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**2)
dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),'blurred2.png')
self.R
#[Out]# 1.3333333333333333
if True:
    i = 2
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
if True:
    i = 1
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
pyramid_depth
#[Out]# 10
if True:
    i = 9
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
if True:
    i = 9
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
if True:
    i = 9
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
if True:
    i = 9
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...],f'masked_original.png')
if True:
    i = 9
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]),f'masked_original.png')
if True:
    i = 9
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
if True:
    i = 18
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
if True:
    i = 15
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
if True:
    i = 13
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
from kornia.geometry.transform.pyramid import pyrdown as pyrdowno
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdown(self.input_img_tensor,
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 10
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdown(self.input_img_tensor,
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 10
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'maskup{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 10
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'maskup{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'maskup{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'maskup{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'maskup{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 14
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_query  = mask1.any().float()
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_query  = torch.all(torch.all(mask1_query,dim=-1),dim=-1).float()
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1, (5,5), 1)
    mask1_query  = torch.all(torch.all(mask1_patches,dim=-1),dim=-1).float()
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
mask1_patches.shape
mask1.shape
#[Out]# torch.Size([1, 1, 7, 9])
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), (5,5), 1)
    mask1_query  = torch.all(torch.all(mask1_patches,dim=-1),dim=-1).float()
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
mask1_query.shape
#[Out]# torch.Size([15, 1])
mask1.shape
#[Out]# torch.Size([1, 1, 7, 9])
patch_size
PATCH_SIZE
patch_size
self.PATCH_SIZE
#[Out]# (7, 7)
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), self.PATCH_SIZE,self.STRIDE)
    mask1_query  = torch.all(torch.all(mask1_patches,dim=-1),dim=-1).float()
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
mask1_query.shape
#[Out]# torch.Size([3, 1])
mask1.shape
#[Out]# torch.Size([1, 1, 7, 9])
mask1_query
#[Out]# tensor([[0.],
#[Out]#         [0.],
#[Out]#         [0.]], device='cuda:0')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), self.PATCH_SIZE,self.STRIDE)
    mask1_query  = torch.all(torch.all(mask1_patches,dim=-1),dim=-1).float()
    mask1_query = mask1_query.reshape((1,1,mask1.shape[2] - 2*(self.PATCH_SIZE[0]//2),mask1.shape[3] - 2*(self.PATCH_SIZE[1]//2) ))
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), self.PATCH_SIZE,self.STRIDE)
    mask1_query  = torch.all(torch.all(mask1_patches,dim=-1),dim=-1).float()
    mask1_query = mask1_query.reshape((1,1,mask1.shape[2] - 2*(self.PATCH_SIZE[0]//2),mask1.shape[3] - 2*(self.PATCH_SIZE[1]//2) ))
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1_query = torch.nn.functional.interpolate(mask1_query,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1_query[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 12
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), self.PATCH_SIZE,self.STRIDE)
    mask1_query  = torch.any(torch.any(mask1_patches,dim=-1),dim=-1).float()
    mask1_query = mask1_query.reshape((1,1,mask1.shape[2] - 2*(self.PATCH_SIZE[0]//2),mask1.shape[3] - 2*(self.PATCH_SIZE[1]//2) ))
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1_query = torch.nn.functional.interpolate(mask1_query,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1_query[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 10
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), self.PATCH_SIZE,self.STRIDE)
    mask1_query  = torch.any(torch.any(mask1_patches,dim=-1),dim=-1).float()
    mask1_query = mask1_query.reshape((1,1,mask1.shape[2] - 2*(self.PATCH_SIZE[0]//2),mask1.shape[3] - 2*(self.PATCH_SIZE[1]//2) ))
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1_query = torch.nn.functional.interpolate(mask1_query,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1_query[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
if True:
    i = 10
    ds1 = pyrdown(self.input_img_tensor, 
               mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                border_type = 'reflect', 
                                align_corners = True, 
                                factor = self.R**i)
    ds1o = pyrdowno(self.input_img_tensor,
                                     border_type = 'reflect',
                                     align_corners = True,
                                     factor = self.R**i)
    mask1 = pyrdown(torch.tensor( (1 - self.mask),device=device)[None,None,...],
                    mask = torch.tensor( (1 - self.mask),device=device)[None,None,...],
                                     border_type = 'reflect',
                                    align_corners = True,
                                     factor = self.R**i)
    mask1 = 1 - mask1
    mask1_patches = extract_patches(mask1.permute(0,2,3,1), self.PATCH_SIZE,self.STRIDE)
    mask1_query  = torch.any(torch.any(mask1_patches,dim=-1),dim=-1).float()
    mask1_query = mask1_query.reshape((1,1,mask1.shape[2] - 2*(self.PATCH_SIZE[0]//2),mask1.shape[3] - 2*(self.PATCH_SIZE[1]//2) ))
    dutils.img_save(tensor_to_numpy(ds1o[0].permute(1,2,0)),f'blurredo{i}.png')
    dutils.img_save(tensor_to_numpy(ds1[0].permute(1,2,0)),f'blurred{i}.png')
    dutils.img_save(tensor_to_numpy(mask1[0].permute(1,2,0)),f'mask{i}.png')
    dutils.img_save(tensor_to_numpy(mask1_query[0].permute(1,2,0)),f'mask_query{i}.png')
    up_ds1 = torch.nn.functional.interpolate(ds1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_ds1o = torch.nn.functional.interpolate(ds1o,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1 = torch.nn.functional.interpolate(mask1,self.input_img_tensor.shape[-2:],mode='bilinear')
    up_mask1_query = torch.nn.functional.interpolate(mask1_query,self.input_img_tensor.shape[-2:],mode='bilinear')
    dutils.img_save(tensor_to_numpy(up_ds1[0].permute(1,2,0)),f'up{i}.png')
    dutils.img_save(tensor_to_numpy(up_ds1o[0].permute(1,2,0)),f'upo{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1[0].permute(1,2,0)),f'upmask{i}.png')
    dutils.img_save(tensor_to_numpy(up_mask1_query[0].permute(1,2,0)),f'upmask_query{i}.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor* torch.tensor( (1 - self.mask),device=device)[None,None,...]).transpose(0,2,3,1)[0],f'masked_original.png')
    dutils.img_save(tensor_to_numpy(self.input_img_tensor).transpose(0,2,3,1)[0],f'original.png')
    
