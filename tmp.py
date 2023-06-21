def get_hook(featname):
    def hook(self,i,o):
        self.__dict__[featname] = torch.tensor(tensor_to_numpy(o),device=o.device)                                                
    return hook