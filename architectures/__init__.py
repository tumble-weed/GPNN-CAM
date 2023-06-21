# from .resnet50 import resnet50
from .vgg16 import vgg16
from torch import nn
import torch
# from torchray.benchmark.models import model_urls
from .utils import replace_module
import re
import copy
from architectures.model_urls import model_urls

def get_model(arch='vgg16',
              dataset='voc',
              convert_to_fully_convolutional=False):
    r"""
    Return a reference model for the specified architecture and dataset.
    The model is returned in evaluation mode.
    Args:
        arch (str, optional): name of architecture. If :attr:`dataset`
            contains ``"imagenet"``, all :mod:`torchvision.models`
            architectures are supported; otherwise, only "vgg16" and
            "resnet50" are currently supported). Default: ``'vgg16'``.
        dataset (str, optional): name of dataset, should contain
            ``'imagenet'``, ``'voc'``, or ``'coco'``. Default: ``'voc'``.
        convert_to_fully_convolutional (bool, optional): If True, convert the
            model to be fully convolutional. Default: False.
    Returns:
        :class:`torch.nn.Module`: model.
    """

    # Set number of classes in dataset.
    if 'voc' in dataset:
        num_classes = 20
    elif 'coco' in dataset:
        num_classes = 80
    elif 'imagenet' in dataset:
        num_classes = 1000
    else:
        assert False, 'Unknown dataset {}'.format(dataset)

    # Get/load the model from torchvision.
    if arch == 'resnet50': #'resnet50'
        model = resnet50(pretrained=True)
#         import pdb;pdb.set_trace()
    elif arch == 'vgg16': #'resnet50'
        model = vgg16(pretrained=True)
    else:
        model = models.__dict__[arch](pretrained='imagenet' in dataset)

    if arch == 'inception_v3':
        model.aux_logits = False

    if 'imagenet' not in dataset:
        # The torchvision models terminate in a classifier for ImageNet.
        # Replace that classifier if we target a different dataset.
        last_name, last_module = list(model.named_modules())[-1]

        # Construct new last layer.
        assert isinstance(last_module, nn.Linear)
        in_features = last_module.in_features
        bias = last_module.bias is not None
        new_layer_module = nn.Linear(in_features, num_classes, bias=bias)

        # Replace the last layer.
        model = replace_module(model, last_name, new_layer_module)

        # Load the model state dict from url.
        if 'voc' in dataset:
            k = 'voc'
        elif 'coco' in dataset:
            k = 'coco'
        else:
            assert False

        checkpoint = torch.hub.load_state_dict_from_url(model_urls[arch][k])

        # Apply the state dict and patch the torchvision models. the
        if arch == 'vgg16':
            _load_caffe_vgg16(model, checkpoint)
#             assert False
            if True:
                if convert_to_fully_convolutional:
                    model = _caffe_vgg16_to_fc(model)

        elif arch == 'resnet50':
            _load_caffe_resnet50(model, checkpoint)
            if True:
                if convert_to_fully_convolutional:
                    model = _caffe_resnet50_to_fc(model)

        else:
            assert False

    else:
        # We don't know how to convert generic models.
        assert not convert_to_fully_convolutional

    # Set model to evaluation mode.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

#=================================================
# fix max pooling
#=================================================

def _fix_caffe_maxpool(model):
    for module in model.modules():
        if isinstance(module, torch.nn.MaxPool2d):
            module.ceil_mode = True
#=================================================
# load caffe resnet50 weights
#=================================================

def _caffe_resnet50_to_fc(model):
    # Shallow copy.
    model_ = copy.copy(model)
#     import pdb;pdb.set_trace()
    # Patch the last layer: fc -> conv.
    out_ch, in_ch = model.fc.weight.shape
    conv = nn.Conv2d(in_ch, out_ch, (1, 1))
    conv.weight.data.copy_(model.fc.weight.view(conv.weight.shape))
    conv.bias.data.copy_(model.fc.bias)
    model_.fc = conv

    # Patch average pooling.
    # model_.avgpool = nn.AvgPool2d((7, 7), stride=1, ceil_mode=True)
    '''
    def forward(self, x):
        # Same as original, but skip flatten layer.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x
    
    model_.forward = types.MethodType(forward, model_)
    '''
    return model_

def _load_caffe_resnet50(model, checkpoint, make_bn_positive=False):
    # Patch the torchvision model to match the Caffe definition.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                                 ceil_mode=True)
    for i in range(2, 5):
        getattr(model, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(model, 'layer%d' % i)[0].conv2.stride = (1, 1)

    # Patch the checkpoint dict and load it.
    def rename(name):
        name = re.sub(r'bn(\d).(0|1).(.*)', r'bn\1.\3', name)
        name = re.sub(r'downsample.(\d).(0|1).(.*)', r'downsample.\1.\3', name)
        return name

    checkpoint = {rename(k): v for k, v in checkpoint.items()}

    # Convert from BGR to RGB.
    checkpoint['conv1.weight'] = checkpoint['conv1.weight'][:, [2, 1, 0], :, :]

    model.load_state_dict(checkpoint)

    # For EBP: the signs of the linear BN weights should be positive.
    # In practice there is only a tiny fraction of neg weights
    # and this does not seem to affect the results much.
    if make_bn_positive:
        conv = None
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                sign = module.weight.sign()
                module.weight.data *= sign
                module.running_mean.data *= sign
                conv.weight.data *= sign.view(-1, 1, 1, 1)
                if conv.bias is not None:
                    conv.bias.data *= sign
            conv = module

    _fix_caffe_maxpool(model)


#=================================================
# load caffe vgg16 weights
#=================================================


# from torchray.benchmark.models import model_urls

def _load_caffe_vgg16(model, checkpoint):
    def filt(key, value):
        # Rename some parameters to allow for the dropout layers,
        # which are not in the original checkpointed data.
        remap = {
            'classifier.0.weight': 'classifier.0.weight',
            'classifier.0.bias': 'classifier.0.bias',
            'classifier.2.weight': 'classifier.3.weight',
            'classifier.2.bias': 'classifier.3.bias',
            'classifier.4.weight': 'classifier.6.weight',
            'classifier.4.bias': 'classifier.6.bias',
        }
        key = remap.get(key, key)

        # Reshape the classifier weights.
        if key == 'features.0.weight':
            # BGR -> RGB
            value = value[:, [2, 1, 0], :, :]
        elif 'classifier' in key and 'weight' in key:
            value = value.reshape(value.shape[0], -1)
        return key, value

    checkpoint = {k: v for k, v in [
        filt(k, v) for k, v in checkpoint.items()]}

    model.load_state_dict(checkpoint)
    _fix_caffe_maxpool(model)


def _caffe_vgg16_to_fc(model):
    # Make shallow copy.
    model_ = copy.copy(model)

    # Transform the fully-connected layers into convolutional ones.
    for i, layer in enumerate(model.classifier.children()):
        if isinstance(layer, nn.Linear):
            out_ch, in_ch = layer.weight.shape
            k_size = 1
            if i == 0:
                in_ch = 512
                k_size = 7
            conv = nn.Conv2d(in_ch, out_ch, (k_size, k_size))
            conv.weight.data.copy_(layer.weight.view(conv.weight.shape))
            conv.bias.data.copy_(layer.bias.view(conv.bias.shape))
            model_.classifier[i] = conv.to(layer.weight.device)
    '''
    def forward(self, x):
        # PyTorch uses a 7x7 adaptive pooling layer to feed the first
        # FC layer; here we skip it for fully-conv operation.
        x = self.features(x)
#         assert False
        x = self.classifier(x)
        return x
    if True:
        model_.forward = types.MethodType(forward, model_)
    '''
    return model_






