    savename = os.path.join(im_save_dir,f'{i}_{classname}{target_ids[0]}.pkl')
    sanity_data = {
      'saliency':(heat_map['saliency']),
      'modelname' : modelname,
      # 'refs' : [tensor_to_numpy(r) for r in refs],
      'target_ids' : target_ids,
      'classname':classname,
      }
    with open(savename,'wb') as f:
      pickle.dump(sanity_data,f)                    
      pass
    
    heatmap_savename = os.path.join(im_save_dir,f'{i}_{classname}{target_ids[0]}.png')
    saliency=  (sanity_data['saliency'])[0,0]
    M = saliency.max()
    M=M+int(M==0)
    saliency=saliency/M
    saliency = cm.jet((sanity_data['saliency'])[0,0])
    
    skimage.io.imsave(heatmap_savename,saliency)
    
