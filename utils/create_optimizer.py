import torch

def create_optimizer(model, loss_fn, config):
    """
    Create optimizer with different learning rates for model and loss function.
    """
    opt_config = config['optimizer']
    opt_name = opt_config['name'].lower()

    print(f"\n🔧 Configuring Optimizer:")
    print(f"├── Type: {opt_name.upper()}")
    print(f"├── Model LR: {opt_config['model_lr']}")
    print(f"├── Weight Decay: {opt_config['weight_decay']}")

    # Prepare parameter groups
    parameter_groups = []

    # Model parameters with layer-wise learning rates
    if opt_config.get('layer_decay', {}).get('enabled', False):
        base_lr = opt_config['model_lr']
        decay_rate = opt_config['layer_decay']['decay_rate']
        print(f"├── Layer-wise Decay: Enabled")
        print(f"│   └── Decay Rate: {decay_rate}")

        # Track layers for printing
        layer_lrs = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Calculate layer depth and corresponding lr
            depth = name.count('.')
            lr = base_lr * (decay_rate ** depth)

            # Store for printing
            layer_lrs[name] = lr

            parameter_groups.append({
                'params': param,
                'lr': lr,
                'name': f"model.{name}"
            })

        # Print first few layer LRs as example
        print("├── Layer Learning Rates (sample):")
        for i, (name, lr) in enumerate(layer_lrs.items()):
            if i < 3:  # Show first 3 layers
                print(f"│   ├── {name}: {lr:.6f}")
            elif i == 3:
                print(f"│   └── ... ({len(layer_lrs)-3} more layers)")
            else:
                break
    else:
        # Without layer-wise decay
        print(f"├── Layer-wise Decay: Disabled")
        parameter_groups.append({
            'params': model.parameters(),
            'lr': opt_config['model_lr'],
            'name': "model"
        })

    # Loss function parameters (if trainable)
    if hasattr(loss_fn, 'parameters'):
        print(f"├── Loss Function: {loss_fn.__class__.__name__}")
        loss_lr = opt_config['loss_lr']
        print(f"├── Loss LR: {loss_lr}")
        parameter_groups.append({
            'params': loss_fn.parameters(),
            'lr': loss_lr,
            'name': "loss"
        })

    # Create optimizer with specific parameters
    if opt_name == 'sgd':
        print("└── SGD Specific:")
        print(f"    ├── Momentum: {opt_config['sgd']['momentum']}")
        print(f"    ├── Nesterov: {opt_config['sgd']['nesterov']}")
        print(f"    └── Dampening: {opt_config['sgd']['dampening']}")

        optimizer = torch.optim.SGD(
            parameter_groups,
            momentum=opt_config['sgd']['momentum'],
            weight_decay=opt_config['weight_decay'],
            nesterov=opt_config['sgd']['nesterov'],
            dampening=opt_config['sgd']['dampening']
        )
    elif opt_name == 'adam':
        print("└── Adam Specific:")
        print(f"    ├── Betas: {opt_config['adam']['betas']}")
        print(f"    ├── Epsilon: {opt_config['adam']['eps']}")
        print(f"    └── AMSGrad: {opt_config['adam']['amsgrad']}")

        optimizer = torch.optim.Adam(
            parameter_groups,
            betas=opt_config['adam']['betas'],
            eps=opt_config['adam']['eps'],
            weight_decay=opt_config['weight_decay'],
            amsgrad=opt_config['adam']['amsgrad']
        )
    elif opt_name == 'adamw':
        print("└── AdamW Specific:")
        print(f"    ├── Betas: {opt_config['adamw']['betas']}")
        print(f"    ├── Epsilon: {opt_config['adamw']['eps']}")
        print(f"    └── AMSGrad: {opt_config['adamw']['amsgrad']}")

        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=opt_config['adamw']['betas'],
            eps=opt_config['adamw']['eps'],
            weight_decay=opt_config['weight_decay'],
            amsgrad=opt_config['adamw']['amsgrad']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return optimizer