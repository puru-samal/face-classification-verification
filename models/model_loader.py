import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# Usage example:
"""
# Initialize the loader
loader = ModelLoader()

# List available models
loader.list_available_models()

# Load a specific model
model = loader.load_model(
    model_name='resnet50',
    embedding_size=512,
    num_classes=1000,
    dropout_rate=0.1,
    device=torch.device('cuda')
)

# Print model summary
loader.print_model_summary(model)

# Get model configuration
config = loader.get_model_info('resnet50')
print(f"Model config: {config}")
"""

class ModelLoader:
    """
    A class to manage model configurations and loading.
    
    Attributes:
        MODEL_CONFIGS (dict): Dictionary containing model configurations
    """
    # Available models
    MODEL_CONFIGS = {
        # ResNet variants
        'resnet18': {
            'model_fn': 'get_resnet',
            'model_type': 'resnet18',
            'module': 'models.resnet'
        },
        'resnet34': {
            'model_fn': 'get_resnet',
            'model_type': 'resnet34',
            'module': 'models.resnet'
        },
        'resnet50': {
            'model_fn': 'get_resnet',
            'model_type': 'resnet50',
            'module': 'models.resnet'
        },

        # SEResNet variants
        'seresnet18': {
            'model_fn': 'get_seresnet',
            'model_type': 'seresnet18',
            'module': 'models.seresnet'
        },
        'seresnet34': {
            'model_fn': 'get_seresnet',
            'model_type': 'seresnet34',
            'module': 'models.seresnet'
        },
        'seresnet50': {
            'model_fn': 'get_seresnet',
            'model_type': 'seresnet50',
            'module': 'models.seresnet'
        },

        # VGG variants
        'vgg11': {
            'model_fn': 'get_vgg',
            'model_type': 'VGG11',
            'module': 'models.vgg'
        },
        'vgg13': {
            'model_fn': 'get_vgg',
            'model_type': 'VGG13',
            'module': 'models.vgg'
        },
        'vgg16': {
            'model_fn': 'get_vgg',
            'model_type': 'VGG16',
            'module': 'models.vgg'
        },
        'vgg19': {
            'model_fn': 'get_vgg',
            'model_type': 'VGG19',
            'module': 'models.vgg'
        },

        # MobileFaceNet
        'mobilefacenet': {
            'model_fn': 'get_mobilefacenet',
            'model_type': 'mobilefacenet',
            'module': 'models.mobilefacenet'
        },

        # ShuffleNetV2 variants
        'shufflenetv2_0.5': {
            'model_fn': 'get_shufflenet',
            'model_type': 'shufflenetv2_0.5',
            'module': 'models.shufflenetv2'
        },
        'shufflenetv2_1.0': {
            'model_fn': 'get_shufflenet',
            'model_type': 'shufflenetv2_1.0',
            'module': 'models.shufflenetv2'
        },
        'shufflenetv2_1.5': {
            'model_fn': 'get_shufflenet',
            'model_type': 'shufflenetv2_1.5',
            'module': 'models.shufflenetv2'
        },
        'shufflenetv2_2.0': {
            'model_fn': 'get_shufflenet',
            'model_type': 'shufflenetv2_2.0',
            'module': 'models.shufflenetv2'
        },

        # EfficientNet variants
        'efficientnet_b0': {
            'model_fn': 'get_efficientnet',
            'model_type': 'efficientnet_b0',
            'module': 'models.efficientnet'
        },

        # ConvNeXt variants
        'convnext_tiny': {
            'model_fn': 'get_convnext',
            'model_type': 'convnext_tiny',
            'module': 'models.convnext'
        },
        'convnext_tinier': {
            'model_fn': 'get_convnext',
            'model_type': 'convnext_tinier',
            'module': 'models.convnext'
        },
        'convnext_tiniest': {
            'model_fn': 'get_convnext',
            'model_type': 'convnext_tiniest',
            'module': 'models.convnext'
        },

        # DenseNet variants
        'densenet121': {
            'model_fn': 'get_densenet',
            'model_type': 'densenet121',
            'module': 'models.densenet'
        },
        'densenet169': {
            'model_fn': 'get_densenet',
            'model_type': 'densenet169',
            'module': 'models.densenet'
        },
        'densenet201': {
            'model_fn': 'get_densenet',
            'model_type': 'densenet201',
            'module': 'models.densenet'
        },
        'densenet161': {
            'model_fn': 'get_densenet',
            'model_type': 'densenet161',
            'module': 'models.densenet'
        },
        'densenet_cifar': {
            'model_fn': 'get_densenet',
            'model_type': 'densenet_cifar',
            'module': 'models.densenet'
        },

        # Inception-ResNet variants
        'inception_resnet_v1': {
            'model_fn': 'get_inception_resnet',
            'model_type': 'inception_resnet_v1',
            'module': 'models.inception_resnet_v1'
        },
    }
    
    def __init__(self):
        """Initialize the ModelLoader."""
        self._validate_configs()
    
    def _validate_configs(self):
        """Validate all model configurations on initialization."""
        for model_name, config in self.MODEL_CONFIGS.items():
            required_keys = {'model_fn', 'model_type', 'module'}
            if not all(key in config for key in required_keys):
                raise ValueError(f"Invalid config for {model_name}. Required keys: {required_keys}")
    
    @classmethod
    def list_available_models(cls, return_list: bool = False) -> Optional[List[str]]:
        """
        Return a sorted list of available model names, grouped by model family.
        
        Returns:
            Prints a formatted view grouped by architecture.
            Returns a list of model names if return_list is True.   
        """
        # Group models by family
        model_families = {}
        for model_name in cls.MODEL_CONFIGS.keys():
            # Extract family name (everything before the number/underscore)
            family = ''.join(c for c in model_name.split('_')[0] if not c.isdigit()).lower()
            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model_name)
        
        # Print formatted groups
        print("\nAvailable Models:")
        print("================")
        for family, models in sorted(model_families.items()):
            print(f"\n{family.upper()}:")
            for model in sorted(models):
                print(f"  • {model}")
        print()  # Extra newline for cleaner output
        
        if return_list:
            return sorted(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, str]:
        """Get configuration information for a specific model."""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls.MODEL_CONFIGS[model_name].copy()
    
    @classmethod
    def load_model(
        cls,
        model_name: str,
        embedding_size: int = 512,
        num_classes: int = 1000,
        dropout_rate: float = 0.0,
        pretrained_path: Optional[str] = None,
        **model_kwargs
    ) -> nn.Module:
        """
        Load a model by name from the available configurations.
        
        Args:
            model_name: Name of the model to load
            embedding_size: Size of the embedding layer
            num_classes: Number of output classes
            dropout_rate: Dropout rate to use
            pretrained_path: Path to pretrained weights
            **model_kwargs: Additional keyword arguments for the model
            
        Returns:
            The instantiated PyTorch model
        """
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {cls.list_available_models()}"
            )
        
        config = cls.MODEL_CONFIGS[model_name]
        
        try:
            module = __import__(config['module'], fromlist=[config['model_fn']])
            model_fn = getattr(module, config['model_fn'])
        except ImportError as e:
            raise ImportError(f"Failed to import {config['module']}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Failed to find {config['model_fn']} in {config['module']}: {e}")
        
        # Build kwargs
        kwargs = {
            'embedding_size': embedding_size,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate,
        }

        # Update with additional model kwargs
        kwargs.update(model_kwargs)
        
        # Create model
        try:
            model = model_fn(config['model_type'], **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create model {model_name}: {e}")
        
        if pretrained_path:
            try:
                model.load_state_dict(torch.load(pretrained_path, weights_only=True))
            except Exception as e:
                raise RuntimeError(f"Failed to load pretrained weights: {e}")
        
        return model
    
    @staticmethod
    def print_model_summary(
        model: nn.Module,
        input_size: Tuple[int, ...] = (4, 3, 112, 112)
    ) -> None:
        """
        Print a summary of the model architecture.
        
        Args:
            model: The PyTorch model
            input_size: Input tensor size (batch_size, channels, height, width)
        """
        try:
            from torchinfo import summary
            summary(model, input_size=input_size)
        except ImportError:
            print("torchinfo not installed. Run: pip install torchinfo")
            print(model)


