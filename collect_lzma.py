import lzma
import os
import glob
import pickle
import numpy as np
from cam_benchmark import settings
from collections import defaultdict
import blosc2
import ipdb
# SAVE_OR_LOAD = 'load'
# BREAK_1_IMAGE = lambda :exec('break')
def ipdbn(*args):
    ipdb.set_trace()
class Data():
    def __init__(self,SAVE_ALL_OR_APPEND,
                 dataset,methodname,modelname,
                 outdir):
        self.dataset,self.methodname,self.modelname = dataset,methodname,modelname
        # self.saliency = []
        # self.ref = []
        # self.out_all = []
        self.SAVE_ALL_OR_APPEND = SAVE_ALL_OR_APPEND
        self.outdir = outdir   
        self.arrayix = 0     
        pass

class SaliencyData(Data):
    def __init__(self,SAVE_ALL_OR_APPEND,
                 dataset,methodname,modelname,
                 ):
        # ipdbn('CHECK SALIENCY')
        outdir = settings.RESULTS_DIR_librecam
        super().__init__(SAVE_ALL_OR_APPEND,
                    dataset,methodname,modelname,
                    outdir)
        # assert False,'untested'
        self.saliency = []
        self.ref = []
        self.out_all = defaultdict(list)
    def add(self,data,
            # arrayix,
            imroot,classname):
        # ipdbn('CHECK SALIENCY')
        if self.SAVE_ALL_OR_APPEND == 'all':
            self.saliency.append(data.pop('saliency'))
            self.ref.append(data.pop('ref',None))
            data['arrayix'] = self.arrayix
            self.arrayix += 1
            self.out_all[imroot].append( {classname:data})
            # import ipdb; ipdb.set_trace()       
        elif self.SAVE_ALL_OR_APPEND == 'append':
            assert False,'not tested'
            out = {imroot:{classname:data}}
            with lzma.open(os.path.join(outdir,'out.xz'),'ab') as f:
                pickle.dump(out,f)            
        # assert False
    def dump(self):
        if self.SAVE_ALL_OR_APPEND == 'all':
            # """
            
            # outdir = methoddir
            with lzma.open(os.path.join(self.outdir,f'{self.dataset}-{self.methodname}-{self.modelname}.xz'),'wb') as f:
                pickle.dump(self.out_all,f)
            # import ipdb; ipdb.set_trace()
            saliency = np.stack(self.saliency)
            # ref = np.stack(ref)
            blosc2.save_array(saliency,os.path.join(self.outdir,f'{self.dataset}-{self.methodname}-{self.modelname}-saliency.bl2'),mode='w')
            # blosc2.save_array(ref,os.path.join(outdir,f'{dataset}-{methodname}-{modelname}-ref.bl2'),mode='w')
            """
            # try out saving xz file instead of bl2 file
            with lzma.open(os.path.join(self.outdir,f'{self.dataset}-{self.methodname}-{self.modelname}-saliency2.bl2'),'wb') as f:
                pickle.dump(saliency,f)
            """
            os.system(f'du -sh {self.outdir}/{self.dataset}-{self.methodname}-{self.modelname}.xz')
            os.system(f'du -sh {self.outdir}/{self.dataset}-{self.methodname}-{self.modelname}-saliency.bl2')        
            # ipdbn('CHECK SALIENCY')
        pass

class MetricsData(Data):
    def __init__(self,SAVE_ALL_OR_APPEND,
                 metricname,
                 dataset,methodname,modelname,
                 ):
        outdir = settings.METRICS_DIR_librecam
        super().__init__(SAVE_ALL_OR_APPEND,
                    dataset,methodname,modelname,
                    outdir)
        self.metricname = metricname
    def add(self,data,
            # arrayix,
            imroot,classname):
        import ipdb;ipdb.set_trace()
    def dump(self):
        import ipdb;ipdb.set_trace()
        pass
    def get_im_save_dirs(self):
        self.methoddir =  os.path.join(settings.METRICS_DIR_librecam,f'{self.dataset}-{self.metricname}-{self.methodname}-{self.modelname}')
        self.im_save_dirs = glob.glob(os.path.join(methoddir,'*/'))
        
def load(outdir,
            dataset,methodname,modelname,
            METRICS_OR_SALIENCY = 'saliency',
            SAVE_ALL_OR_APPEND = 'all',
            expected_size = None):
    if METRICS_OR_SALIENCY == 'saliency':
        outdir = settings.RESULTS_DIR_librecam
    
    if SAVE_ALL_OR_APPEND == 'all':
        with lzma.open(os.path.join(outdir,f'{dataset}-{methodname}-{modelname}.xz'),'rb') as f:
            loaded = pickle.load(f)
        loaded_saliency = blosc2.load_array(os.path.join(outdir,f'{dataset}-{methodname}-{modelname}-saliency.bl2'))            
        # loaded_ref = blosc2.load_array(os.path.join(outdir,f'{dataset}-{methodname}-{modelname}-ref.bl2')                    
        assert len(loaded_saliency.shape[0]) == expected_size
        return loaded,loaded_saliency
def save(
    # outdir,
            metricname,
            dataset,methodname,modelname,
            METRICS_OR_SALIENCY = 'saliency',
            SAVE_ALL_OR_APPEND = 'all',
            REMOVE_ORIGINAL_DIR = False,
            expected_size = None,
            ):
    
    if METRICS_OR_SALIENCY == 'metrics':
        metrics_data_obj = MetricsData(SAVE_ALL_OR_APPEND,
                                        metricname,
                                        dataset,methodname,modelname,
                                        )
        
        methoddir = os.path.join(settings.METRICS_DIR_librecam,f'{dataset}-{metricname}-{methodname}-{modelname}')
        outdir = settings.METRICS_DIR_librecam
        collect_obj = metrics_data_obj
    elif METRICS_OR_SALIENCY == 'saliency':
        # import ipdb; ipdb.set_trace()
        # ipdbn('CHECK SALIENCY')
        methoddir = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodname}-{modelname}')
        outdir = settings.RESULTS_DIR_librecam
        saliency_data_obj = SaliencyData(SAVE_ALL_OR_APPEND,
                 dataset,methodname,modelname,)
                 
        collect_obj = saliency_data_obj
    print(methoddir)
    arrayix = 0
    im_save_dirs = glob.glob(os.path.join(methoddir,'*/'))
    # print(im_save_dirs)
    # pklname = glob.glob(os.path.join(im_save_dir,'*.xz

    # outdir = methoddir
    
    # out_all = defaultdict(dict)
    out_all = defaultdict(list)
    if False:
        saliency = []
        ref = []
    # import ipdb; ipdb.set_trace()
    if len(im_save_dirs) == 0 :
        
        assert False,f'no im_save_dirs found, {methoddir}'
    for i,im_save_dir in enumerate(im_save_dirs):
        print(im_save_dir)
        # if arrayix > 500:
        #     DBG_BREAK_500 = 1
        #     break
        imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))
        # print(i,imroot)
        pklnames = glob.glob(os.path.join(im_save_dir,'*.xz'))
        # assert len(pklnames) == 1,'len(pklnames) = %d'%len(pklnames)
        for pklname in pklnames:
            print(arrayix,i,pklname)
            with lzma.open(pklname,'rb') as f:
                data = pickle.load(f)
            classname = data['classname']
            if SAVE_ALL_OR_APPEND == 'all':
                collect_obj.add(data,imroot,classname)
                
                if False:
                    if METRICS_OR_SALIENCY == 'saliency':
                        saliency.append(data.pop('saliency'))
                        ref.append(data.pop('ref'))
                        
                        data['arrayix'] = arrayix
                        arrayix += 1
                        out_all[imroot].append( {classname:data})
                        # import ipdb; ipdb.set_trace()
                    elif METRICS_OR_SALIENCY == 'metrics':
                        assert False,'not implemented'
            if SAVE_ALL_OR_APPEND == 'append':
                if False:
                    # """
                    if METRICS_OR_SALIENCY == 'saliency':
                        out = {imroot:{classname:data}}
                        with lzma.open(os.path.join(outdir,'out.xz'),'ab') as f:
                            pickle.dump(out,f)
                    elif METRICS_OR_SALIENCY == 'metrics':
                        assert False,'not implemented'
                    # """
            # break
            # if i > 5:
            #     break
        # if "BREAK_1_IMAGE":
        #     break
        # assert False
    # import ipdb; ipdb.set_trace()
    assert len(collect_obj.saliency) == expected_size
    if False:
        if SAVE_ALL_OR_APPEND == 'all':
            # """
            import blosc2
            # outdir = methoddir
            with lzma.open(os.path.join(outdir,f'{dataset}-{methodname}-{modelname}.xz'),'wb') as f:
                pickle.dump(out_all,f)
            # import ipdb; ipdb.set_trace()
            saliency = np.stack(saliency)
            # ref = np.stack(ref)
            blosc2.save_array(saliency,os.path.join(outdir,f'{dataset}-{methodname}-{modelname}-saliency.bl2'),mode='w')
            # blosc2.save_array(ref,os.path.join(outdir,f'{dataset}-{methodname}-{modelname}-ref.bl2'),mode='w')
            with lzma.open(os.path.join(outdir,f'{dataset}-{methodname}-{modelname}-saliency2.bl2'),'wb') as f:
                pickle.dump(saliency,f)
            
            # """
        os.system(f'du -sh {outdir}/{dataset}-{methodname}-{modelname}.xz')
        os.system(f'du -sh {outdir}/{dataset}-{methodname}-{modelname}-saliency.bl2')
        # os.system(f'du -sh {outdir}/{dataset}-{methodname}-{modelname}-saliency2.bl2')
    # import ipdb;ipdb.set_trace()
    
    collect_obj.dump()

    if 'reload':
        loaded,loaded_saliency = load(collect_obj.outdir,
            dataset,methodname,modelname,
            METRICS_OR_SALIENCY = 'saliency',
            SAVE_ALL_OR_APPEND = 'all')    
        print(
            'n_images in loaded:',len(loaded.keys())
            )
        print(
            'n_saliency:',loaded_saliency.shape)
        print('n image folders')
        os.system(f'find {methoddir} -type f -name "*.xz" | grep -i "\.xz$" | wc -l')    
    if REMOVE_ORIGINAL_DIR:
        import ipdb;ipdb.set_trace()
        os.system(f'rm -rf {methoddir}')    
def main(
        
        SAVE_OR_LOAD = 'save',
        METRICS_OR_SALIENCY = 'saliency',
        SAVE_ALL_OR_APPEND = 'all',
        REMOVE_ORIGINAL_DIR = False,
        methodname= 'gradcam',
        metricname = 'chattopadhyay',
        modelname = 'vgg16',
        dataset = 'pascal',
        expected_size = None,
):
    # # import ipdb;ipdb.set_trace()
    # ipdbn('CHECK SALIENCY')
    # if not os.environ.get('CUSTOM',None) == '1':

    if SAVE_OR_LOAD== 'save':
        # ipdbn('CHECK SAVE')
        save(
            # outdir,
            metricname,
            dataset,methodname,modelname,
            METRICS_OR_SALIENCY = METRICS_OR_SALIENCY,
            SAVE_ALL_OR_APPEND = SAVE_ALL_OR_APPEND,
            REMOVE_ORIGINAL_DIR = REMOVE_ORIGINAL_DIR,
            expected_size = expected_size
            )
    elif SAVE_OR_LOAD == 'load':
        load(
            # outdir,
            dataset,methodname,modelname,
            METRICS_OR_SALIENCY = METRICS_OR_SALIENCY,
            SAVE_ALL_OR_APPEND = SAVE_ALL_OR_APPEND,
            expected_size = expected_size,
            )

    # import ipdb; ipdb.set_trace()
if __name__ == '__main__':
    main()
