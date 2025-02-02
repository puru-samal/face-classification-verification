import yaml
import os
import shutil
import torch
from torchinfo import summary
import argparse
from models.model_loader import ModelLoader
from torch.utils.data import DataLoader
import gc
import wandb
from data import (
    ImageDataset,
    ImagePairDataset,
    print_cls_dataset_statistics,
    print_cls_dataloader_info,
    print_ver_dataset_statistics,
    print_ver_dataloader_info
)
from utils import (
    create_transforms,
    get_loss_config,
    create_optimizer,
    create_lr_scheduler,
    create_weight_scheduler,
    FaceTrainer
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", DEVICE)

# Model Loader
# --------------------------------------------------------------------------------
model_loader = ModelLoader()
model_loader.list_available_models()
# --------------------------------------------------------------------------------

# Parse Arguments
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Face Recognition Training")   
parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
args = parser.parse_args()
# --------------------------------------------------------------------------------

# Load Config 
# --------------------------------------------------------------------------------
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
# --------------------------------------------------------------------------------

# Classification Data
# --------------------------------------------------------------------------------
## Datasets
cls_train_dataset = ImageDataset(
    root          = config['cls_data_dir'] + '/train', 
    transform     = create_transforms(augment=config['augument']), 
    num_classes   = config['num_classes'], 
    preload       = config['preload']
)
cls_val_dataset   = ImageDataset(
    root          = config['cls_data_dir'] + '/dev',   
    transform     = create_transforms(augment=False), 
    num_classes   = config['num_classes'], 
    preload       = config['preload']
)
cls_test_dataset  = ImageDataset(
    root          = config['cls_data_dir'] + '/test', 
    transform     = create_transforms(augment=False), 
    num_classes   = config['num_classes'], 
    preload       = config['preload']
)

## Dataloaders
cls_train_loader = DataLoader(
    cls_train_dataset, 
    batch_size   = config['batch_size'], 
    shuffle      = True,  
    num_workers  = 5, 
    pin_memory   = True, 
    prefetch_factor = 4
)
cls_val_loader   = DataLoader(
    cls_val_dataset,   
    batch_size   = config['batch_size'], 
    shuffle      = False, 
    num_workers  = 5, 
    pin_memory   = True, 
    prefetch_factor = 4
)
cls_test_loader  = DataLoader(
    cls_test_dataset, 
    batch_size   = config['batch_size'], 
    shuffle      = False, 
    num_workers  = 5, 
    pin_memory   = True, 
    prefetch_factor = 4
)

# Print Dataset Statistics
# --------------------------------------------------------------------------------
print_cls_dataset_statistics(cls_train_dataset, cls_val_dataset, cls_test_dataset)
# --------------------------------------------------------------------------------

# Print Dataloader Info
# --------------------------------------------------------------------------------
print_cls_dataloader_info(cls_train_loader, cls_val_loader, cls_test_loader)
# --------------------------------------------------------------------------------
gc.collect()
# --------------------------------------------------------------------------------


# Verification Data
# --------------------------------------------------------------------------------
## Datasets
ver_val_dataset  = ImagePairDataset(
    root         = config['ver_data_dir'], 
    pairs_file   = config['val_pairs_file'],  
    transform    = create_transforms(augment=False), 
    preload      = config['preload'])

ver_test_dataset = ImagePairDataset(
    root         = config['ver_data_dir'], 
    pairs_file   = config['test_pairs_file'], 
    transform    = create_transforms(augment=False), 
    preload      = config['preload'])

## Dataloaders
ver_val_loader   = DataLoader(
    ver_val_dataset,   
    batch_size   = config['batch_size'], 
    shuffle      = False, 
    num_workers  = 5, 
    pin_memory   = True, 
    prefetch_factor = 4
)
ver_test_loader  = DataLoader(
    ver_test_dataset, 
    batch_size   = config['batch_size'], 
    shuffle      = False, 
    num_workers  = 5, 
    pin_memory   = True, 
    prefetch_factor = 4
)

# Print Dataset Statistics
# --------------------------------------------------------------------------------
print_ver_dataset_statistics(ver_val_dataset, ver_test_dataset)
# --------------------------------------------------------------------------------

# Print Dataloader Info
# --------------------------------------------------------------------------------
print_ver_dataloader_info(ver_val_loader, ver_test_loader)
# --------------------------------------------------------------------------------
gc.collect()    
# --------------------------------------------------------------------------------  

# Model
# --------------------------------------------------------------------------------
model = model_loader.load_model(
    config['model']['name'], 
    embedding_size = config['model']['embedding_size'], 
    num_classes    = len(cls_train_dataset.classes), 
    dropout_rate   = config['model']['dropout_rate']
)
# --------------------------------------------------------------------------------

# Print Model Info
# --------------------------------------------------------------------------------
print(f"Model successfully loaded!")
print(f"Model Type: {type(model).__name__}")
print(f"Model Name: {config['model']['name']}")
print(f"Embedding size: {config['model']['embedding_size']}")
print(f"Number of classes: {len(cls_train_dataset.classes)}")

model = model.to(DEVICE)
# Print model summary
input_size = (config['batch_size'], 3, 112, 112)
model_stats = summary(model, input_size=input_size)
# --------------------------------------------------------------------------------


# Experiment Setup
# --------------------------------------------------------------------------------  
experiment_name = f"{config['TA_name']}_{config['model']['name']}_{config['train_type']}"
if config["train_type"] == "finetune" or config["train_type"] == "joint":
    experiment_name = f"{experiment_name}_{config['verification_loss']['name']}"

# make experiment directory
expt_root = f"expts/{experiment_name}"
os.makedirs(expt_root, exist_ok=True)
# Copy config file
shutil.copy(args.config, f"{expt_root}/config.yaml")
# Write stats to file
with open(f"{expt_root}/model_summary.txt", 'w') as f:
    f.write(str(model_stats))
# --------------------------------------------------------------------------------


# WandB
# --------------------------------------------------------------------------------
wandb_run = None
if config['use_wandb']:
    wandb_run = wandb.init(
        name = experiment_name, 
        project = "HW2p2_TA", 
        config = config 
    )
# --------------------------------------------------------------------------------


# Loss | Optimizer | Scheduler | Weight Scheduler
# --------------------------------------------------------------------------------
loss_config = get_loss_config(len(cls_train_dataset.classes),config)
optimizer   = create_optimizer(model, loss_config, config)
scheduler   = create_lr_scheduler(optimizer, config, cls_train_loader)
weight_scheduler = create_weight_scheduler(config)
# --------------------------------------------------------------------------------  


# Train | Eval  
# --------------------------------------------------------------------------------
if config['train_type'] == 'cls':

    # Update weights to use classification loss only
    loss_config.ver_weight = 0.0
    loss_config.cls_weight = 1.0

    print("\n🔧 Updated Loss Weights:")
    print(f"├── Verification Weight: {loss_config.ver_weight}")
    print(f"├── Classification Weight: {loss_config.cls_weight}")

    trainer = FaceTrainer(
        model = model,
        optimizer = optimizer,  
        scheduler = scheduler,
        loss_config = loss_config,
        device = DEVICE,
        wandb_run = wandb_run,
        weight_scheduler = weight_scheduler,

    )

    # Load checkpoint if specified
    if config['checkpoint']['path']:
        trainer.load_checkpoint(
            path = config['checkpoint']['path'],
            load_scheduler = config['checkpoint']['load_scheduler'],
            load_optimizer = config['checkpoint']['load_optimizer']
        )   

    # Train 
    trainer.train(
        train_loader   = cls_train_loader,
        val_cls_loader = cls_val_loader,
        val_ver_loader = ver_val_loader,
        num_epochs     = config['train']['epochs'],
        save_dir       = f"{expt_root}/checkpoints" 
    )

    # Evaluate
    trainer.evaluate(
        cls_test_loader = cls_test_loader,
        ver_test_loader = ver_test_loader
    )

elif config['train_type'] == 'ver_finetune':
    
    # Update weights to use verification loss only
    loss_config.ver_weight = 1.0
    loss_config.cls_weight = 0.0

    print("\n🔧 Updated Loss Weights:")
    print(f"├── Verification Weight: {loss_config.ver_weight}")
    print(f"├── Classification Weight: {loss_config.cls_weight}")

    trainer = FaceTrainer(
        model     = model,
        optimizer = optimizer,  
        scheduler = scheduler,
        loss_config = loss_config,
        device = DEVICE,
        wandb_run = wandb_run,
        weight_scheduler = weight_scheduler
    )

    # Load checkpoint if specified
    if config['checkpoint']['path']:
        trainer.load_checkpoint(
            path = config['checkpoint']['path'],
            load_scheduler = config['checkpoint']['load_scheduler'],
            load_optimizer = config['checkpoint']['load_optimizer']
        )  

    # Finetune
    trainer.train(
        train_loader   = cls_train_loader,
        val_cls_loader = cls_val_loader,
        val_ver_loader = ver_val_loader,
        num_epochs     = config['train']['epochs'],
        save_dir       = f"{expt_root}/checkpoints"
    )

    # Evaluate
    trainer.evaluate(
        cls_test_loader = cls_test_loader,
        ver_test_loader = ver_test_loader
    )

elif config['train_type'] == 'joint':

    trainer = FaceTrainer(
        model = model,
        optimizer = optimizer,  
        scheduler = scheduler,
        loss_config = loss_config,
        device = DEVICE,
        wandb_run = wandb_run,
        weight_scheduler = weight_scheduler
    )

    # Load checkpoint if specified
    if config['checkpoint']['path']:
        trainer.load_checkpoint(
            path = config['checkpoint']['path'],
            load_scheduler = config['checkpoint']['load_scheduler'],
            load_optimizer = config['checkpoint']['load_optimizer']
        )  

    # Train
    trainer.train(
        train_loader   = cls_train_loader,  
        val_cls_loader = cls_val_loader,
        val_ver_loader = ver_val_loader,
        num_epochs     = config['train']['epochs'], 
        save_dir       = f"{expt_root}/checkpoints"
    )

    # Evaluate
    trainer.evaluate(
        cls_test_loader = cls_test_loader,
        ver_test_loader = ver_test_loader
    )
# --------------------------------------------------------------------------------
