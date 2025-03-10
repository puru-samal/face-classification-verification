'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchinfo import summary


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, embedding_size=512, num_classes=1000, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3,
                              32,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Feature layers
        self.layers = self._make_layers(in_channels=32)
        
        # Add embedding layer similar to other models
        self.embedding = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(p=dropout_rate)),
            ('linear', nn.Linear(cfg['out_channels'][-1], embedding_size, bias=False)),
            ('bn', nn.BatchNorm1d(embedding_size))
        ])) if self.embedding_size != cfg['out_channels'][-1] else nn.Identity()

        # Classification head
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                    'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)

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


def get_efficientnet(efficientnet_type='efficientnet_b0', embedding_size=512, num_classes=1000, dropout_rate=0.2):
    if efficientnet_type == 'efficientnet_b0':
        cfg = {
            'num_blocks': [1, 2, 2, 3, 3, 4, 1],
            'expansion': [1, 6, 6, 6, 6, 6, 6],
            'out_channels': [16, 24, 40, 80, 112, 192, 320],
            'kernel_size': [3, 3, 5, 3, 5, 5, 3],
            'stride': [1, 2, 2, 2, 1, 2, 1],
            'dropout_rate': dropout_rate,
            'drop_connect_rate': 0.2,
        }
        return EfficientNet(cfg, embedding_size=embedding_size, num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Invalid EfficientNet type: {efficientnet_type}")


if __name__ == "__main__":
    # Example usage:
    model = get_efficientnet('efficientnet_b0', embedding_size=320, num_classes=8631, dropout_rate=0.2)
    
    # Print model summary
    input_size = (4, 3, 112, 112)
    summary(model, input_size=input_size, device='cpu')