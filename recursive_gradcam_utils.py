                
                if False:
                    def find_local_maximum(cam):
                        cam0 = cam
                        cam = tensor_to_numpy(cam)
                        """
                        keep only the maximum in a local neighborhood
                        """
                        import scipy.signal
                        nrows,ncols = cam.shape[-2:]
                        peak_indexes1 = []
                        for ri in range(nrows):
                            camrow = cam[:,:,ri]
                            peaks,_ = scipy.signal.find_peaks(camrow[0,0])
                            # peaks = peaks.squeeze()
                            npeaks = len(peaks)
                            for pair in zip([ri]*npeaks,peaks):
                                print(pair)
                                peak_indexes1.append(pair)
                        peak_indexes2 = []
                        for ci in range(ncols):
                            camcol = cam[:,:,:,ci]
                            peaks,_ = scipy.signal.find_peaks(camcol[0,0])
                            # peaks = peaks.squeeze()
                            npeaks = len(peaks)
                            for pair in zip(peaks,[ci]*npeaks):
                                peak_indexes2.append(pair)                            
                        peaks = set(peak_indexes1).intersection(set(peak_indexes2))
                        print(peaks)
                        out = torch.zeros_like(cam0)
                        assert cam0.shape[0]== 1
                        out[0,0,list(zip(*peaks))] = cam0[0,0,list(zip(*peaks))]
                        
                        return out
                    last_cam_up0 = last_cam_up
                    last_cam_up = find_local_maximum(last_cam_up)
                    last_cam_up = find_local_maximum(last_cam_up)
                    import ipdb;ipdb.set_trace()
                
