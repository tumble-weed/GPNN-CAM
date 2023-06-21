# https://gist.github.com/samson-wang/a6073c18f2adf16e0ab5fb95b53db3e6
def get_grad(model,feats=feats):
     #device = model.device
     device = 'cuda'
     model.eval()
     ref = torch.ones((1,3,224,224)).to(device).float().requires_grad_(
 True)
     if ref.grad is not None:
         ref.grad.zero_()
     dummy_out = model(ref)
     loss = feats[0][...,3,3].sum()
     loss.backward()
     g = ref.grad
     g1 = (g!=0).float()
     print(g1.mean())
get_grad(model)
feats = []
def get_feats(self,inp,out,feats=feats):
    feats.insert(0,out)
model.layer4.register_forward_hook(get_feats)
