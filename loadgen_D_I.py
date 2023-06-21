import torch
import blosc2
import importlib
import register_ipdb
import os
import numpy as np
importlib.reload(register_ipdb)
class GPNN_loadgen_D_I():
    def __init__(self,config):
        self.previous_loaded = None
        self.config = config
        pass
    def load(self,device):
        from gpnn_gradcam_multi2 import extract_patches
        Dreloaded = blosc2.load_array(os.path.join(self.config['blosc_dir'],'saved_gpnn_D.bl2'))            
        Ireloaded = blosc2.load_array(os.path.join(self.config['blosc_dir'],'saved_gpnn_I.bl2'))                                    
        Dreloaded = torch.tensor(Dreloaded,device=device).float()
        Ireloaded = torch.tensor(Ireloaded,device=device).long()

        # dutils.cipdb('TRACE_STEP_BY_STEP')
        Dreloaded2 = Dreloaded.view(self.config['n_super_iters'],self.config['batch_size'],-1,1,1,1)
        
        Ireloaded2 = Ireloaded.reshape((self.config['n_super_iters'],self.config['batch_size'],-1)+Ireloaded.shape[1:])
        input_img_bhwc = self.config['input_img'].permute(0,2,3,1)
        ref_patches =  extract_patches(input_img_bhwc,( self.config['patch_size'],self.config['patch_size']), self.config['stride'])     
        """           
        for i in range(n_super_iters):
            augmentations,aggregation_results,saliency_dicts,model =   augmentation_attribution(config=config,cnns=cnns,dataset=dataset,running_saliency_dicts=running_saliency_dicts,ref=ref,Dreloaded2=Dreloaded2[i],Ireloaded=Ireloaded2[i],ref_patches=ref_patches)
            import ipdb; ipdb.set_trace()
        """
        # import ipdb; ipdb.set_trace()    
        self.previous_loaded['Dreloaded2'] = Dreloaded2
        self.previous_loaded['Ireloaded'] = Ireloaded
        self.previous_loaded['ref_patches'] = ref_patches
        self.i = 0
        # return Dreloaded2,Ireloaded,ref_patches    
        
    def get_augmentations_batch(self,
    # ref_patches,Ireloaded,Dreloaded2
                          ):    
        # from gpnn_gradcam_multi2 import gpnn
        device = 'cpu' if self.config['no_cuda'] or not torch.cuda.is_available() else 'cuda'
        # model = gpnn(config)
        #========================================================        
        from aggregation import combine_patches_multiple_images
        # Dreloaded = blosc2.load_array(os.path.join(config['blosc_dir'],'saved_gpnn_D.bl2'))            
        # Ireloaded = blosc2.load_array(os.path.join(config['blosc_dir'],'saved_gpnn_I.bl2'))                                    
        # Dreloaded = torch.tensor(Dreloaded,device=device).float()
        # Ireloaded = torch.tensor(Ireloaded,device=device).long()

        # Dreloaded2 = Dreloaded.reshape(config['batch_size'],Dreloaded.shape[0]//config['batch_size'],1,1,1)
        # import ipdb; ipdb.set_trace()
        permuted_patches = [torch.index_select(self.previous_loaded['ref_patches'],0,Ireloaded[self.i][j].squeeze()).unsqueeze(0) for j in range(self.config['batch_size'])]
        permuted_patches = torch.cat(permuted_patches)
        permuted_patches1 = permuted_patches
        # assert permuted_patches.shape[0] ==1
        # permuted_patches1 = permuted_patches[0].reshape(config['batch_size'],permuted_patches.shape[1]//config['batch_size'],permuted_patches.shape[2],permuted_patches.shape[3],permuted_patches.shape[4])
        """
        aggregation_results = combine_patches(permuted_patches[0][Dre2.shape[1]*aix :(Dre2.shape[1])*(aix+1) ], 
                            model.PATCH_SIZE, 
                            model.STRIDE, 
                            model.input_img.shape,
                            as_np=False,
                            patch_aggregation=model.PATCH_AGGREGATION,
                            distances=Dre2[aix],
                            I=Ire2[Dre2.shape[1]*aix :(Dre2.shape[1])*(aix+1) ])
        """
        # import ipdb; ipdb.set_trace()
        Ireloaded = self.previous_loaded['Ireloaded'][self.i].flatten(start_dim=0,end_dim=1)
        # import ipdb; ipdb.set_trace()
        aggregation_results = combine_patches_multiple_images(
            permuted_patches1, 
            self.config['patch_size'], self.config['stride'], self.config['input_img'].shape[-2:] + (self.config['input_img'].shape[1],),
            as_np=False,
            patch_aggregation=self.config['patch_aggregation'],
            distances_bp111=self.previous_loaded['Dreloaded2'][self.i],I_bp1=Ireloaded) 
        # import ipdb; ipdb.set_trace()
        # augmentations,aggregation_results = 
        augmentations = aggregation_results['combined']
        augmentations = augmentations.permute(0,3,1,2)
        aggregation_results['D'] = self.previous_loaded['Dreloaded2'][self.i]
        aggregation_results['I'] = Ireloaded[self.i]      
        
        # saliency_dicts = [{} for _ in cnns]  
        # return augmentations,aggregation_results,saliency_dicts
        self.i += 1
        return augmentations,aggregation_results

class GPNN_onlygen_save_D_I():
    def __init__(self,config):
        self.config = config
    def add(self,Dnp,Inp):
        n_super_iters = self.config['n_super_iters']
        if "D_for_save" not in self.__dict__:

            self.D_for_save = np.zeros((Dnp.shape[0]* n_super_iters,*Dnp.shape[1:]))
            self.I_for_save = np.zeros((Inp.shape[0]* n_super_iters,*Inp.shape[1:]))
            self.i = 0
        self.D_for_save[self.i*Dnp.shape[0]:(self.i+1)*Dnp.shape[0]] = Dnp
        self.I_for_save[self.i*Inp.shape[0]:(self.i+1)*Inp.shape[0]] = Inp
        # import ipdb; ipdb.set_trace()
        self.i += 1
        assert False,'untested'
    def dump(self):
        blosc2.save_array(self.D_for_save,os.path.join(self.config['blosc_dir'],'saved_gpnn_D.bl2'),mode='w')            
        blosc2.save_array(self.I_for_save,os.path.join(self.config['blosc_dir'],'saved_gpnn_I.bl2'),mode='w')                    
        if 'check un boscing' and True:
            # Dre = np.empty(self.D_for_save.shape,self.D_for_save.dtype)
            # Ire = np.empty(self.I_for_save.shape,self.I_for_save.dtype)
            # blosc2.decompress(compressed_D,dst=Dre)
            # blosc2.decompress(compressed_I,dst=Ire)
            # print(np.abs(D_for_save - Dre).sum())  
            # print(np.abs(I_for_save - Ire).sum())
            # if 'check re augment':
            #     aix = 0
            #     Dre2 = Dre.reshape(augmentations.shape[0],Dre.shape[0]//augmentations.shape[0],1,1,1)
            #     Dre2 =torch.tensor(Dre2,device=device)
            #     Ire2 = torch.tensor(Ire,device=device,dtype=torch.long)
            #     ref_patches =  extract_patches(model.input_img, model.PATCH_SIZE, model.STRIDE)
            #     ref_patches = torch.index_select(ref_patches,0,Ire2.squeeze()).unsqueeze(0)
            #     patch_aggregation_results = combine_patches(ref_patches[0][Dre2.shape[1]*aix :(Dre2.shape[1])*(aix+1) ], 
            #                                                 model.PATCH_SIZE, 
            #                                                 model.STRIDE, 
            #                                                 model.input_img.shape,
            #                                                 as_np=False,
            #                                                 patch_aggregation=model.PATCH_AGGREGATION,
            #                                                 distances=Dre2[aix],
            #                                                 I=Ire2[Dre2.shape[1]*aix :(Dre2.shape[1])*(aix+1) ])
            #     augmentations_re = patch_aggregation_results['combined']
            #     print((augmentations_re-augmentations[aix].permute(1,2,0)).abs().sum())
            #     dutils.img_save(augmentations_re,'acomb.png')
            #     dutils.img_save(augmentations[0],'a.png')    
            if 'check load':
                blosc2.save_array(self.D_for_save,'saved_gpnn_D.bl2',mode='w')            
                blosc2.save_array(self.I_for_save,'saved_gpnn_I.bl2',mode='w')            
                Dreloaded = blosc2.load_array('saved_gpnn_D.bl2')            
                Ireloaded = blosc2.load_array('saved_gpnn_I.bl2')                  
                print(np.abs(self.D_for_save - Dreloaded).sum())  
                print(np.abs(self.I_for_save - Ireloaded).sum())
            import ipdb; ipdb.set_trace()        