'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchinfo import summary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.adaptive_avg_pool2d(out, (1, 1))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*planes, (self.expansion*planes)//16, kernel_size=1)
        self.fc2 = nn.Conv2d((self.expansion*planes)//16, self.expansion*planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Squeeze
        w = F.adaptive_avg_pool2d(out, (1, 1))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, embedding_size=512, num_classes=1000, dropout_rate=0.0):
        super(SENet, self).__init__()
        self.in_planes = 64
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Add embedding layer similar to other models
        self.embedding = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(p=dropout_rate)),
            ('linear', nn.Linear(512 * block.expansion, embedding_size, bias=False)),
            ('bn', nn.BatchNorm1d(embedding_size))
        ])) if self.embedding_size != 512 * block.expansion else nn.Identity()  

        # Classification head
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Update in_planes with expansion factor
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
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


def get_seresnet(model_type='seresnet18', embedding_size=512, num_classes=1000, dropout_rate=0.0):
    if model_type == 'seresnet18':
        return SENet(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            embedding_size=embedding_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    elif model_type == 'seresnet34':
        return SENet(
            block=BasicBlock,
            num_blocks=[3, 4, 6, 3],
            embedding_size=embedding_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    elif model_type == 'seresnet50':
        return SENet(
            block=Bottleneck,
            num_blocks=[3, 4, 6, 3],
            embedding_size=embedding_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Invalid SEResNet type: {model_type}")


if __name__ == "__main__":
    # Example usage:
    model = get_seresnet(model_type='seresnet34', embedding_size=512, num_classes=8631, dropout_rate=0.0)
    
    # Print model summary
    input_size = (4, 3, 112, 112)
    summary(model, input_size=input_size, device='cpu')


