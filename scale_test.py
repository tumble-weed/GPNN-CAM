import torch
import numpy as np

def scale_test(hooked_cnn,
               normalized_augmentations,
               target_id,
                  NAUG = 20,
                  START=0):
    NAUG = 13
    from mycam import HookToggle
    scale_probs = {}
    memory_chunk = 100*224*224
    # for SZ in [224,256,288,320,352,384,416,448,480,512,544,576]:
    with HookToggle(hooked_cnn,False):
        for SZ in [224,240,256,272,288,304,320,352,420,484,568,632,
                #    696,764
                   ]:

            with torch.inference_mode():
                aug = torch.nn.functional.interpolate(normalized_augmentations[START:START+NAUG],size=(SZ,SZ),mode='bilinear')
                n_chunks = (np.prod([aug.shape[0],*aug.shape[-2:]]) + memory_chunk - 1)//memory_chunk
                batch_size = (aug.shape[0] + n_chunks - 1)//n_chunks
                scale_prob = 0
                for ci in range(n_chunks):
                    scores = hooked_cnn.cnn(aug[ci*batch_size:(ci+1)*batch_size])
                    print(scores.shape)
                    probs = torch.softmax(scores,dim=1)

                    # import ipdb;ipdb.set_trace()
                    if probs.ndim == 4:
                        
                        # probs = probs.mean(dim=(-1,-2))
                        # probs = probs.amax(dim=-1).amax(dim=-1)
                        
                        # scores = (scores - scores.amax(dim=1,keepdim=True)).amax(dim=-1).amax(dim=-1)
                        # scores = (scores - scores.amax(dim=1,keepdim=True)).mean(dim=(-1,-2))
                        # scores = scores.mean(dim=(-1,-2))
                        # entropy = -(probs*torch.log(probs)).sum(dim=1).mean(dim=(-1,-2))
                        # probs = (probs - probs.amax(dim=1,keepdim=True)).amax(dim=-1).amax(dim=-1)
                        # probs = (probs - probs.amax(dim=1,keepdim=True)).mean(dim=(-1,-2))
                        if True:
                            scores = scores.mean(dim=(-1,-2))    
                        elif False:
                            scores = scores.mean(dim=(-1,-2))    
                            # scores = scores - scores.amax(dim=1,keepdim=True)
                        elif False:
                            scores = scores.amax(dim=-1).amax(dim=-1)                            
                        elif False:
                            probs = (probs - probs.amax(dim=1,keepdim=True).mean(dim=(-1,-2),keepdim=True)).mean(dim=(-1,-2)
                                                                                                                )
                            
                        if False:
                            avg_scores = scores.mean(dim=(-1,-2))
                            probs = torch.softmax(avg_scores,dim=-1)
                        elif False:
                            max_scores = scores.amax(dim=(-1,-2))
                            probs = torch.softmax(max_scores,dim=-1)
                    #=======================================================
                    if False:
                        scale_prob = scale_prob + probs[:,target_id].sum()
                    elif True:
                        scale_prob = scale_prob + scores[:,target_id].sum()
                    elif False:
                        scale_prob = scale_prob + entropy.sum()
                    #=======================================================
                    # scale_prob = scale_prob + (probs[:,target_id]/probs.amax(dim=-1)).sum()
                    # print(probs[:,target_id].mean())
                scale_probs[SZ] = scale_prob/aug.shape[0]
    print(scale_probs)
    best_scale = max(scale_probs,key=scale_probs.get)
    print('best_scale',best_scale,'best_scale_prob',scale_probs[best_scale])        
    return scale_probs