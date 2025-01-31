import torch
from torch import nn
from typing import Tuple
from torch import Tensor
from collections import OrderedDict
from torchinfo import summary

class ConvBlock(nn.Module):
    """Convolution block with batch normalization and PReLU activation."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 groups: int = 1) -> None:
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            groups: Number of groups for grouped convolution
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel,
                             stride=stride, padding=padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: conv -> bn -> prelu"""
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    """Convolution block with batch normalization (no activation)."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel,
                             stride=stride, padding=padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: conv -> bn"""
        return self.bn(self.conv(x))


class DepthWise(nn.Module):
    """Depthwise separable convolution with optional residual connection."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (2, 2),
                 padding: Tuple[int, int] = (1, 1),
                 groups: int = 1,
                 residual: bool = False) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, groups, kernel=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, kernel=kernel,
                                stride=stride, padding=padding, groups=groups)
        self.project = LinearBlock(groups, out_channels, kernel=(1, 1))
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        shortcut = x if self.residual else None
        x = self.project(self.conv_dw(self.conv(x)))
        return x + shortcut if shortcut is not None else x


class Residual(nn.Module):
    """Stack of residual blocks."""
    def __init__(self,
                 channels: int,
                 num_blocks: int,
                 groups: int,
                 kernel: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        blocks = [
            DepthWise(channels, channels, kernel=kernel,
                     stride=stride, padding=padding,
                     groups=groups, residual=True)
            for _ in range(num_blocks)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MobileFaceNet(nn.Module):
    """MobileFaceNet architecture for face recognition."""

    def __init__(self, embedding_size: int = 512, num_classes: int = 1000, dropout_rate: float = 0.0) -> None:
        """
        Args:
            embedding_size: Size of the face embedding vector
            num_classes: Number of classes for classification
            dropout_rate: Dropout rate for the embedding layer
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Network architecture with shape annotations
        self.features = nn.Sequential(OrderedDict([
            # Input shape: (B, 3, H, W)
            ('conv1', ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))),
            # Shape: (B, 64, H/2, W/2)
            ('conv2_dw', ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)),
            ('conv_23', DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)),
            # Shape: (B, 64, H/4, W/4)
            ('conv_3', Residual(64, num_blocks=4, groups=128)),
            ('conv_34', DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)),
            # Shape: (B, 128, H/8, W/8)
            ('conv_4', Residual(128, num_blocks=6, groups=256)),
            ('conv_45', DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)),
            # Shape: (B, 128, H/16, W/16)
            ('conv_5', Residual(128, num_blocks=2, groups=256)),
            ('conv_6_sep', ConvBlock(128, 512, kernel=(1, 1))),
            # Shape: (B, 512, H/16, W/16)
            ('conv_6_dw', LinearBlock(512, 512, groups=512, kernel=(7, 7))),
            # Shape: (B, 512, 1, 1)
            ('flatten', nn.Flatten()),
            # Shape: (B, 512)
        ]))

        self.embedding = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(p=dropout_rate)),
            ('linear', nn.Linear(512, embedding_size, bias=False)),
            ('bn', nn.BatchNorm1d(embedding_size))
        ])) if self.embedding_size != 512 else nn.Identity()

        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(self.embedding_size),
            nn.Linear(self.embedding_size, self.num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dict: Dictionary containing the following keys:
                - 'embedding':  embedding tensor of shape (B, embedding_size)
                - 'cls_output': Classification output tensor of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.embedding(x)
        cls_output = self.cls_head(x)
        return {'embedding': x, 'cls_output': cls_output}

    def get_embedding(self, x: Tensor) -> Tensor:
        """Extract face embedding from input image."""  
        return self.forward(x)['embedding']

    def get_classification(self, x: Tensor) -> Tensor:
        """Extract classification output from input image."""
        return self.forward(x)['cls_output']


def get_mobilefacenet(model_type='mobilefacenet', embedding_size=512, num_classes=1000, dropout_rate=0.0):
    return MobileFaceNet(embedding_size=embedding_size, num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":

    # Example usage:    
    model = get_mobilefacenet(model_type='mobilefacenet', embedding_size=512, num_classes=8631, dropout_rate=0.0)

    # Print model summary
    input_size = (4, 3, 112, 112)
    summary(model, input_size=input_size, device='cpu')