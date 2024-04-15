""" Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py """

import torch
import torch.nn as nn
import torchvision

class VGGModel(torchvision.models.VGG):
    def __init__(self, d_in=3, d_out=20, classifier_dim1=512, classifier_dim2=512, 
                conv_kernel_size=3, input_size=(64, 64), init_weights=True):

        cfg = [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M']
        pooling_layers = cfg.count('M')
        classifier_inputdim = cfg[-2] * int(input_size[0] / 2 ** pooling_layers) * int(input_size[1] / 2 ** pooling_layers)

        # Make layers
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(d_in, v, kernel_size=conv_kernel_size, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                d_in = v
        features = nn.Sequential(*layers)

        super(VGGModel, self).__init__(features)

        if hasattr(self, 'avgpool'):  # Compat Pytorch>1.0.0
            self.avgpool = torch.nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(classifier_inputdim, classifier_dim1),
            nn.ReLU(True),
            nn.Linear(classifier_dim1, classifier_dim2),
            nn.ReLU(True),
        )
        self.upper = nn.Linear(classifier_dim2, d_out)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = self.features(x)
        o = self.avgpool(o)
        o = torch.flatten(o, 1)
        o = self.classifier(o)
        o = self.upper(o)
        return o
