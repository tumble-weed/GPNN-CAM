
def get_difficult_images(dataset,modelnames):
    if dataset in ["","imagenet"]:
        image_paths = [
        #USING AI
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00002445.JPEG', # solar panel
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000685.JPEG', #bunny
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00001144.JPEG', # dump truck?
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00002178.JPEG', # watch
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00004818.JPEG', # dog
        #USING APD
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00003628.JPEG',# black dog blur

        '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000013.JPEG', # monkey
        '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000009.JPEG', # mousetrap?
        '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000020.JPEG    ', # crane?
        
        
        '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00000072.JPEG', # walrus like?
        
        
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00002691.JPEG',
        # '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00004429.JPEG',
        '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00001279.JPEG',
        
        '/root/bigfiles/dataset/imagenet/images/val/ILSVRC2012_val_00004446.JPEG',

        ]
    elif dataset == 'pascal':
        # vgg16
        image_paths = [
            
    #     '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/006165.jpg', # cycle side angle
    # '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/006877.jpg', # dog
    # '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/008613.jpg', # dog on couch
    # '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000539.jpg', # dog and black kid

'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/008119.jpg',# sofa
'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000004.jpg',# cars
'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000003.jpg',# chair

'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000008.jpg',# chair


'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000006.jpg',#pottedplant 

'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000014.jpg',# bus,car,person






'/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000248.jpg',# horse













   










         








    





   

    
    


    



    
    
    
    

    
    
    
    
    
    '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/005974.jpg',# car
    '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/002138.jpg', #dog
    '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000765.jpg',# person?
    '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/007505.jpg',# bird
    
    '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/002217.jpg',# plane
    
    '/root/bigfiles/dataset/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
    




    ]            
    return image_paths