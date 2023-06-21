import torch.nn as nn
import torch
import copy
class DenseAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class DenseVGG19(nn.Module):
    def __init__(self, stop_at_layer=None):
        super(DenseVGG19, self).__init__()
        self.stop_at_layer = stop_at_layer

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            # nn.ZeroPad2d((0,1,0,1)),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # nn.ZeroPad2d((0,2,0,2)),
            nn.MaxPool2d(kernel_size=2, stride=1,dilation=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            # nn.ZeroPad2d((0,4,0,4)),
            nn.MaxPool2d(kernel_size=2, stride=1,dilation=4),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            # nn.ZeroPad2d((0,8,0,8)),
            nn.MaxPool2d(kernel_size=2, stride=1,dilation=8),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            # nn.ZeroPad2d((0,16,0,16)),
            nn.MaxPool2d(kernel_size=2, stride=1,dilation=16),
        )
    def forward(self, x):
        features = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i == self.stop_at_layer:
                return x
            if isinstance(module, nn.Conv2d):
                features.append(x)
        return features

from torchvision.models import vgg19
model0 = vgg19(pretrained=True)
model = DenseVGG19()
model.features.load_state_dict(copy.deepcopy(model0.features.state_dict()))
features = list(model.features.modules())[1:]
new_features = []


# next_dilation =1 
next_padding = 1 
for li,l in enumerate(features):
    # l = features[li]
    if isinstance(l,torch.nn.MaxPool2d):
        z = torch.nn.ZeroPad2d((0,next_padding,0,next_padding))
        next_padding = next_padding * 2
        new_features.append(z)
    new_features.append(l)
model.features = torch.nn.Sequential(*new_features)
fout = model(torch.ones(1,3,224,224))