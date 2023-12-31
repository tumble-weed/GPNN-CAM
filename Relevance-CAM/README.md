# PyTorch Implementation of [Relevance-CAM: Your Model Already Knows Where to Look] [CVPR 2021]

This is implementation of **[Relevance-CAM: Your Model Already Knows Where to Look]** which is accepted by CVPR 2021

<img src="https://github.com/mongeoroo/Relevance-CAM/blob/main/images/Comparison.jpg?raw=true" width="400px" height="350px" title="px(픽셀) 크기 설정" alt="Comparison"></img><br/>



## Introduction
Official implementation of [Relevance-CAM: Your Model Already Knows Where to Look].

We introduce a novel method which allows to analyze the intermediate layers of a model as well as the last convolutional layer.

Method consists of 3 phases:

1. Propagation to get the activation map of a target layer of a model.

2. Backpropagation of to the target layer with Layer-wise Relevance Propagation Rule to calculate the weighting component

3. Weighted summation with activation map and weighting component.

<img src="https://github.com/mongeoroo/Relevance-CAM/blob/main/images/R_CAM_pipeline.jpg?raw=true" width="400px" height="350px" title="px(픽셀) 크기 설정" alt="Relevance-CAM pipeline"></img><br/>


### Code explanation
Example: By running Multi_CAM.py, multiple CAM results of images in the picture folder can be saved in the results folder. 
```
python Multi_CAM.py --models resnet50 --target_layer layer2   for the last layer of ResNet50
python Multi_CAM.py --models resnet50 --target_layer layer4   for the intermediate layer of ResNet50

python Multi_CAM.py --models vgg16 --target_layer 43          for the last layer of VGG16
python Multi_CAM.py --models vgg16 --target_layer 23          for the intermediate layer of VGG16

python Multi_CAM.py --models vgg19 --target_layer 52          for the last layer of VGG16
python Multi_CAM.py --models vgg19 --target_layer 39          for the intermediate layer of VGG16
```
1. Model: resnet50 or vgg16, by --models. 
2. Target layer for R-CAM: you can choose layer2 of resnet50 by --target_layer.
3. Target class: If you want to make R-CAM for african elephant class, you can use --target_class 386, 386 is index of ImageNet for elephant. default value is maximum index of model output.

### Citation
```
@InProceedings{Lee_2021_CVPR,
    author    = {Lee, Jeong Ryong and Kim, Sewon and Park, Inyong and Eo, Taejoon and Hwang, Dosik},
    title     = {Relevance-CAM: Your Model Already Knows Where To Look},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14944-14953}
}
```
