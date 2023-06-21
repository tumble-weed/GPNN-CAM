import torch
# global faiss, res
# import faiss
# res = faiss.StandardGpuResources()
# print('faiss initialized!')
from model.nearest_neighbors import create_index,initialize_faiss
initialize_faiss()
# keys = torch.arange(1000,device='cuda').float()[:,None]
keys = torch.randn(10000,10).cuda().float()
create_index(keys,index_type='ivf',index_options={'nlist':20},keys_to_keep=range(10))