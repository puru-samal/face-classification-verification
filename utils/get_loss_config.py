from pytorch_metric_learning import losses, miners, distances
from typing import Optional, Any
from dataclasses import dataclass
from torch import nn

@dataclass
class LossConfig:
    """Configuration for both classification and verification losses."""
    cls_loss_fn: Optional[nn.Module]  = None
    cls_weight: Optional[float] = None
    ver_loss_fn: Optional[nn.Module]  = None
    ver_weight: Optional[float] = None
    miner: Optional[Any]        = None
    distance: Optional[Any]     = None

def get_loss_config(num_classes: int, config: dict) -> LossConfig:
    """
    Factory function to create both classification and verification losses.

    Args:
        num_classes: Number of classes in the dataset
        config: Dictionary containing model configuration

    Returns:
        LossConfig object containing both losses and verification components
    """
    # Get configs
    cls_config = config['classification_loss']
    ver_config = config['verification_loss']
    ver_loss_name = ver_config['name']
    embedding_size = config['model']['embedding_size']

    # Setup classification loss
    cls_loss = nn.CrossEntropyLoss(
        label_smoothing=cls_config.get('label_smoothing', 0.0)
    )

    # Default distance for verification
    default_distance = distances.CosineSimilarity()

    # Mining configuration
    mining_config = ver_config.get('mining', {})
    if mining_config.get('type') == 'multi_similarity':
        default_miner = miners.MultiSimilarityMiner(
            epsilon=mining_config.get('epsilon', 0.1)
        )
    else:
        default_miner = None

    ver_loss_configs = {
        'arcface': LossConfig(
            ver_loss_fn=losses.ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=ver_config['arcface'].get('margin', 0.5),
                scale=ver_config['arcface'].get('scale', 64.0)
            ),
            distance=default_distance
        ),

        'cosface': LossConfig(
            ver_loss_fn=losses.CosFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=ver_config['cosface'].get('margin', 0.35),
                scale=ver_config['cosface'].get('scale', 64.0)
            ),
            distance=default_distance
        ),

        'sphereface': LossConfig(
            ver_loss_fn=losses.SphereFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=ver_config['sphereface'].get('margin', 0.35),
                scale=ver_config['sphereface'].get('scale', 64.0)
            ),
            distance=default_distance
        ),

        'triplet': LossConfig(
            ver_loss_fn=losses.TripletMarginLoss(
                margin=ver_config['triplet'].get('margin', 0.05),
                distance=default_distance,
                reducer=None
            ),
            miner=default_miner,
            distance=default_distance
        ),

        'margin': LossConfig(
            ver_loss_fn=losses.MarginLoss(
                margin=ver_config['margin'].get('margin', 0.2),
                distance=default_distance,
                reducer=None
            ),
            miner=default_miner,
            distance=default_distance
        ),

        'npair': LossConfig(
            ver_loss_fn=losses.NPairsLoss(
                distance=default_distance
            ),
            distance=default_distance
        ),

        'contrastive': LossConfig(
            ver_loss_fn=losses.ContrastiveLoss(
                pos_margin=ver_config['contrastive'].get('pos_margin', 1.0),
                neg_margin=ver_config['contrastive'].get('neg_margin', 0.0),
                distance=default_distance
            ),
            miner=miners.PairMarginMiner(
                pos_margin=ver_config['contrastive'].get('pos_margin', 1.0),
                neg_margin=ver_config['contrastive'].get('neg_margin', 0.0),
                distance=default_distance
            ),
            distance=default_distance
        )
    }

    if ver_loss_name not in ver_loss_configs:
        raise ValueError(
            f"Unsupported loss function: {ver_loss_name}. "
            f"Supported losses: {list(ver_loss_configs.keys())}"
        )

    # Set cls_loss as a member
    loss_config = ver_loss_configs[ver_loss_name]
    loss_config.cls_loss_fn = cls_loss

    # Set the weights if specified
    loss_config.cls_weight = cls_config.get('weight', 0.0)
    loss_config.ver_weight = ver_config.get('weight', 0.0)

    # Print configuration
    print("\n🔧 Loss Components Loaded:")
    print(f"├── Classification: {loss_config.cls_loss_fn.__class__.__name__}")
    print(f"├── Classification Weight: {loss_config.cls_weight}")
    print(f"├── Verification: {loss_config.ver_loss_fn.__class__.__name__}")
    print(f"├── Verification Weight: {loss_config.ver_weight}")
    if loss_config.miner:
        print(f"├── Miner: {loss_config.miner.__class__.__name__}")
        if hasattr(loss_config.miner, 'epsilon'):
            print(f"│   └── Epsilon: {loss_config.miner.epsilon}")
    print(f"└── Distance: {loss_config.distance.__class__.__name__}")
    print()

    return loss_config