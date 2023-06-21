import torch
import blosc2
import importlib
import register_ipdb
importlib.reload(register_ipdb)
import os
import numpy as np
import dutils

DTYPE = 'uint8'
class GPNN_loadgen_augmentations():
    def __init__(self,config):
        self.previous_load = None
        self.config = config
        # self.DTYPE = 'uint8'
        pass
    def load(self,device):    
        augmentations_reloaded = blosc2.load_array(os.path.join(self.config['blosc_dir'],'saved_gpnn_augmentations.bl2'))      
        if DTYPE == 'uint8':
            augmentations_reloaded = augmentations_reloaded/255.
        # print('set correct codec, also set correct dtype')
        
        # import ipdb; ipdb.set_trace()
        
        augmentations_reloaded = torch.tensor(augmentations_reloaded,device=device).float()
        self.augmentations_reloaded2 = augmentations_reloaded.view(self.config['n_super_iters'],self.config['batch_size'],*augmentations_reloaded.shape[1:])        
        # self.previous_load['augmentations_reloaded2'] = augmentations_reloaded2
        self.i = 0
        # return augmentations_reloaded2
    #===========================================        
    def get_augmentations_batch(self):
        aggregation_results = {}
        augmentations = self.augmentations_reloaded2[self.i].permute(0,3,1,2)
        aggregation_results['combined'] = augmentations        
        self.i += 1
        return augmentations,aggregation_results
    #===========================================
class GPNN_onlygen_save_augmentations():
    def __init__(self,config):
        self.config = config
        # self.DTYPE = 'uint8'
        pass
    def add(self,augmentations_np):
        # import ipdb; ipdb.set_trace()
        if 'augmentations_for_save' not in self.__dict__:
            n_super_iters = self.config['n_super_iters']
            self.augmentations_for_save = np.zeros((augmentations_np.shape[0]* n_super_iters,*augmentations_np.shape[1:]),dtype=np.float32)
            self.i = 0
        self.augmentations_for_save[self.i*augmentations_np.shape[0]:(self.i+1)*augmentations_np.shape[0]] = augmentations_np
        self.i += 1
    def dump(self,):
        from blosc2 import Codec
        # import ipdb; ipdb.set_trace()
        if DTYPE == 'uint8':
            blosc2.save_array(
            
            # self.augmentations_for_save.astype(np.float32),
            (255*self.augmentations_for_save).astype(np.uint8),
            os.path.join(self.config['blosc_dir'],'saved_gpnn_augmentations.bl2'),mode='w',
                          cparams= dict(
                clevel=9,
            # codec =Codec.BLOSCLZ
            # codec =Codec.ZFP_RATE
            code = Codec.ZLIB,
            # typesize = 255
            )
                          )
        else:
            assert False
        if 'check un boscing' and False:
            if 'check load':
                # blosc2.save_array(D_for_save,'saved_gpnn_D.bl2',mode='w')            
                # blosc2.save_array(I_for_save,'saved_gpnn_I.bl2',mode='w')            
                augmentations_reloaded = blosc2.load_array(os.path.join(self.config['blosc_dir'],'saved_gpnn_augmentations.bl2'))            
                print(np.abs(self.augmentations_for_save - augmentations_reloaded).sum())  
            import ipdb; ipdb.set_trace()               

