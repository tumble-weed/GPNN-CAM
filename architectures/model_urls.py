import os
base_model_url = 'https://dl.fbaipublicfiles.com/torchray/'

model_urls = {
    'vgg16': {
        'coco': os.path.join(base_model_url, 'vgg16_coco-d82c8150.pth.tar'),
        'voc': os.path.join(base_model_url, 'vgg16_voc-b050e8c3.pth.tar'),
    },
    'resnet50': {
        'coco': os.path.join(base_model_url, 'resnet50_coco-e99468c5.pth.tar'),
        'voc': os.path.join(base_model_url, 'resnet50_voc-9ca920d6.pth.tar'),
    },
}
