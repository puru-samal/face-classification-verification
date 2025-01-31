import torch
from typing import Dict, Any, Optional
from torch.optim import lr_scheduler

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    train_loader: Optional[torch.utils.data.DataLoader] = None
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on config settings.
    """
    scheduler_config = config['scheduler']
    scheduler_name = scheduler_config['name'].lower()

    print("\n📈 Configuring Learning Rate Scheduler:")
    print(f"├── Type: {scheduler_name.upper()}")

    # Create base scheduler
    if scheduler_name == 'reduce_lr':
        reduce_config = scheduler_config['reduce_lr']
        print("├── ReduceLROnPlateau Settings:")
        print(f"│   ├── Mode: {reduce_config.get('mode', 'min')}")
        print(f"│   ├── Factor: {reduce_config.get('factor', 0.1)}")
        print(f"│   ├── Patience: {reduce_config.get('patience', 10)}")
        print(f"│   ├── Threshold: {reduce_config.get('threshold', 0.0001)}")
        print(f"│   ├── Threshold Mode: {reduce_config.get('threshold_mode', 'rel')}")
        print(f"│   ├── Cooldown: {reduce_config.get('cooldown', 0)}")
        print(f"│   └── Min LR: {reduce_config.get('min_lr', 0.00001)}")

        base_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=reduce_config.get('mode', 'min'),
            factor=reduce_config.get('factor', 0.1),
            patience=reduce_config.get('patience', 10),
            threshold=reduce_config.get('threshold', 0.0001),
            threshold_mode=reduce_config.get('threshold_mode', 'rel'),
            cooldown=reduce_config.get('cooldown', 0),
            min_lr=reduce_config.get('min_lr', 0.00001),
            eps=reduce_config.get('eps', 1e-8)
        )

    elif scheduler_name == 'cosine':
        cosine_config = scheduler_config['cosine']
        print("├── Cosine Annealing Settings:")
        print(f"│   ├── T_max: {cosine_config.get('T_max', 60)}")
        print(f"│   └── Min LR: {cosine_config.get('eta_min', 0.0001)}")

        base_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_config.get('T_max', 60),
            eta_min=cosine_config.get('eta_min', 0.0001),
            last_epoch=cosine_config.get('last_epoch', -1)
        )

    elif scheduler_name == 'cosine_warm':
        warm_config = scheduler_config['cosine_warm']
        print("├── Cosine Annealing Warm Restarts Settings:")
        print(f"│   ├── T_0: {warm_config.get('T_0', 10)}")
        print(f"│   ├── T_mult: {warm_config.get('T_mult', 2)}")
        print(f"│   └── Min LR: {warm_config.get('eta_min', 0.0001)}")

        base_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=warm_config.get('T_0', 10),
            T_mult=warm_config.get('T_mult', 2),
            eta_min=warm_config.get('eta_min', 0.0001),
            last_epoch=warm_config.get('last_epoch', -1)
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            f"Supported: ['reduce_lr', 'cosine', 'cosine_warm']"
        )

    # Add warmup if enabled
    if scheduler_config.get('warmup', {}).get('enabled', False):
        warmup_config = scheduler_config['warmup']
        print("├── Warmup Settings:")
        print(f"│   ├── Epochs: {warmup_config.get('epochs', 5)}")
        print(f"│   ├── Start Factor: {warmup_config.get('start_factor', 0.1)}")
        print(f"│   └── End Factor: {warmup_config.get('end_factor', 1.0)}")

        scheduler = create_warmup_scheduler(
            optimizer,
            base_scheduler,
            warmup_config,
            train_loader
        )
    else:
        print("└── Warmup: Disabled")
        scheduler = base_scheduler

    return scheduler


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    base_scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Create a warmup scheduler wrapped around the base scheduler.
    """
    warmup_epochs = warmup_config.get('epochs', 5)
    start_factor = warmup_config.get('start_factor', 0.1)
    end_factor = warmup_config.get('end_factor', 1.0)

    # Calculate the number of warmup steps
    warmup_steps = len(train_loader) * warmup_epochs

    # Create warmup scheduler
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_steps
    )

    # Combine warmup with main scheduler
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps]
    )

    return scheduler