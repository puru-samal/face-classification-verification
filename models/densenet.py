'''DenseNet in PyTorch.'''
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=1000, embedding_size=512, dropout_rate=0.0):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        
        # Add embedding layer
        self.embedding = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(p=dropout_rate)),
            ('linear', nn.Linear(num_planes, embedding_size, bias=False)),
            ('bn', nn.BatchNorm1d(embedding_size))
        ])) if self.embedding_size != num_planes else nn.Identity()

        # Classification head
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        x = self.forward_features(x)
        x = self.embedding(x)
        cls_output = self.cls_head(x)
        return {'embedding': x, 'cls_output': cls_output}

    def get_embedding(self, x):
        """Extract embedding from input image."""
        return self.forward(x)['embedding']

    def get_classification(self, x):
        """Extract classification output from input image."""
        return self.forward(x)['cls_output']

def get_densenet(model_type='densenet121', num_classes=1000, embedding_size=512, dropout_rate=0.0):
    if model_type == 'densenet121':
        return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, 
                       num_classes=num_classes, embedding_size=embedding_size, 
                       dropout_rate=dropout_rate)
    elif model_type == 'densenet169':
        return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32,
                       num_classes=num_classes, embedding_size=embedding_size,
                       dropout_rate=dropout_rate)
    elif model_type == 'densenet201':
        return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32,
                       num_classes=num_classes, embedding_size=embedding_size,
                       dropout_rate=dropout_rate)
    elif model_type == 'densenet161':
        return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48,
                       num_classes=num_classes, embedding_size=embedding_size,
                       dropout_rate=dropout_rate)
    elif model_type == 'densenet_cifar':
        return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,
                       num_classes=num_classes, embedding_size=embedding_size,
                       dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Invalid DenseNet type: {model_type}")

if __name__ == "__main__":
    # Example usage:
    model_type = 'densenet_cifar'
    model = get_densenet(model_type=model_type, num_classes=8631, embedding_size=384, dropout_rate=0.0)
    
    # Print model summary with the same input size as other models
    input_size = (4, 3, 112, 112)
    summary(model, input_size=input_size, device='cpu')