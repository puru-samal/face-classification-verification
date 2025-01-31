'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from collections import OrderedDict
from torchinfo import summary


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, embedding_size=512, num_classes=1000, dropout_rate=0.0):
        super(VGG, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.features = self._make_layers(cfg[vgg_name])
        
        # Add embedding layer similar to other models
        self.embedding = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(p=dropout_rate)),
            ('linear', nn.Linear(512, embedding_size, bias=False)),
            ('bn', nn.BatchNorm1d(embedding_size))
        ])) if self.embedding_size != 512 else nn.Identity()    

        # Classification head
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

    def forward_features(self, x):
        out = self.features(x)
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

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]  # Global average pooling
        return nn.Sequential(*layers)


def get_vgg(model_type='VGG11', embedding_size=512, num_classes=1000, dropout_rate=0.0):
    if model_type.upper() in cfg:
        return VGG(
            vgg_name=model_type.upper(),
            embedding_size=embedding_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Invalid VGG type: {model_type}")


if __name__ == "__main__":
    # Example usage:
    model = get_vgg(model_type='VGG19', embedding_size=512, num_classes=8631, dropout_rate=0.0)
    
    # Print model summary
    input_size = (4, 3, 112, 112)
    summary(model, input_size=input_size, device='cpu')