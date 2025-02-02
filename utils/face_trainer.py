import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import gc
import os   
from typing import Dict, Optional, Any, Tuple, Union
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
from .metrics_calculator import MetricsCalculator
from .get_loss_config import LossConfig


class FaceTrainer:
    """
    Unified trainer for face classification and verification using ArcFaceModel.
    Handles both training and evaluation for classification and verification tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler._LRScheduler] = None,
        loss_config: Optional[LossConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        wandb_run: Optional[Any] = None,
        weight_scheduler: Optional[Any] = None,
    ):
        """
        Initialize face system.

        Args:
            model: model instance
            optimizer: Optional optimizer for training
            scheduler: Optional learning rate scheduler
            loss_config: Loss configuration
            device: Computation device
            wandb_run: Optional WandB run object
            weight_scheduler: Optional scheduler for loss weights
        """
        self.model       = model.to(device)
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.loss_config = loss_config
        self.device      = device
        self.metrics     = MetricsCalculator()
        self.scaler      = GradScaler(device = str(self.device))
        self.wandb_run   = wandb_run
        self.weight_scheduler = weight_scheduler

    # --------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch with progress bar and optional mixed precision.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary containing epoch metrics
        """

        if self.optimizer is None:
            raise ValueError("Optimizer must be provided for training.")

        if self.loss_config is None:
            raise ValueError("Loss configuration must be provided for training.")


        self.model.train()
        self.metrics.reset()
        epoch_loss = 0.0

        # Initial memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        pbar = tqdm(train_loader, desc='Training', leave=True)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Update weights if weight scheduler is available
            if self.weight_scheduler is not None:
                ver_weight = self.weight_scheduler.get_weight()
                cls_weight = 1.0 - ver_weight
            else:
                cls_weight = self.loss_config.cls_weight
                ver_weight = self.loss_config.ver_weight

            # Forward pass with autocast
            with autocast(device_type=str(self.device)):
                outputs  = self.model(images)
                total_loss = 0.0

                # Classification loss (if enabled)
                if cls_weight > 0:
                    cls_loss = self.loss_config.cls_loss_fn(outputs['cls_output'], labels)
                    total_loss += cls_weight * cls_loss

                # Verification loss (if enabled)
                if ver_weight > 0:
                    embeddings = outputs['embedding']

                    # Use miner if configured for that loss
                    if self.loss_config.miner is not None:
                        miner_out = self.loss_config.miner(embeddings, labels)
                        ver_loss  = self.loss_config.ver_loss_fn(embeddings, labels, miner_out)
                    else:
                        ver_loss = self.loss_config.ver_loss_fn(embeddings, labels)

                    total_loss += ver_weight * ver_loss

            # Backward pass with scaler
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics (use float32 for metrics)
            with autocast(device_type=str(self.device), enabled=False):
                self.metrics.update_classification(outputs['cls_output'].detach().float(), labels)

            epoch_loss += total_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'cls_loss': f"{cls_loss.item():.4f}" if cls_weight > 0 else "N/A",
                'ver_loss': f"{ver_loss.item():.4f}" if ver_weight > 0 else "N/A",
                'cls_w': f"{cls_weight:.2f}",
                'ver_w': f"{ver_weight:.2f}",
                'acc': f'{self.metrics.get_averages()["classification"]["top1"]:.2f}%',
                'top5': f'{self.metrics.get_averages()["classification"]["top5"]:.2f}%',
                'lr': f"{float(self.optimizer.param_groups[0]['lr']):.4e}"
            })

            # Clear memory after each batch
            del images, labels, outputs, total_loss

        if self.scheduler is not None:
            self.scheduler.step()

        # Final memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Log epoch metrics
        metrics = self.metrics.get_averages()['classification']
        metrics['loss'] = epoch_loss / len(train_loader)
        self._log_metrics(metrics, prefix='train')
        return metrics


    def train(
        self,
        train_loader: DataLoader,
        val_cls_loader: Optional[DataLoader] = None,
        val_ver_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_dir: str = 'checkpoints'
    ) -> None:
      """
      Train with progress tracking and periodic threshold updates.

      Args:
          train_loader: Training data loader
          val_cls_loader: Classification validation loader
          val_ver_loader: Verification validation loader
          num_epochs: Number of epochs to train
          save_dir: Directory to save checkpoints
      """
      os.makedirs(save_dir, exist_ok=True)

      # Initialize best metrics if not already done
      if not hasattr(self, 'best_cls_acc'):
          self.best_cls_acc = 0.0
      if not hasattr(self, 'best_ver_acc'):
          self.best_ver_acc = 0.0

      print(f"\nStarting training for {num_epochs} epochs")
      print("=" * 50)

      for epoch in range(num_epochs):

          print(f"\nEpoch {epoch+1}/{num_epochs}")
          print("-" * 30)

          # Train
          train_metrics = self.train_epoch(train_loader)

          # Validate classification
          cls_metrics = {}
          if val_cls_loader is not None:
              cls_metrics = self.evaluate_classification(val_cls_loader)

              if cls_metrics['top1'] > self.best_cls_acc:
                  self.best_cls_acc = cls_metrics['top1']
                  self.save_checkpoint(os.path.join(save_dir, 'best_cls_model.pt'))

          # Validate verification and update threshold
          ver_metrics = {}
          if val_ver_loader is not None:
              # Update threshold periodically
              ver_metrics = self.evaluate_verification(val_ver_loader)

              if ver_metrics['ACC'] > self.best_ver_acc:
                  self.best_ver_acc = ver_metrics['ACC']
                  self.save_checkpoint(os.path.join(save_dir, 'best_ver_model.pt'))

          # Save latest checkpoint with current threshold
          self.save_checkpoint(os.path.join(save_dir, 'latest.pt'))

          # Log epoch results
          self._log_epoch_results(
              epoch + 1,
              train_metrics,
              cls_metrics if val_cls_loader else None,
              ver_metrics if val_ver_loader else None
          )

          # Memory cleanup after each epoch
          gc.collect()
          torch.cuda.empty_cache()

      print("\nTraining completed!")
      print("=" * 50)
      print(f"Best classification accuracy: {self.best_cls_acc:.2f}%")
      print(f"Best verification accuracy: {self.best_ver_acc:.2f}%")

      # Log final best metrics
      if self.wandb_run is not None:
          self.wandb_run.summary.update({
              'best_cls_acc': self.best_cls_acc,
              'best_ver_acc': self.best_ver_acc
          })

    # --------------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------------- 


    @torch.no_grad()
    def evaluate_classification(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate classification performance with progress bar.

        Args:
            val_loader: Validation/test data loader

        Returns:
            Dictionary containing classification metrics
        """

        if self.loss_config is None:
            raise ValueError("Loss configuration must be provided for evaluation.")

        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        # Clear memory before evaluation
        gc.collect()
        torch.cuda.empty_cache()

        # Create progress bar
        pbar = tqdm(val_loader, desc='Evaluating classification', leave=True)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.loss_config.cls_loss_fn(outputs['cls_output'], labels)

            # Update metrics
            self.metrics.update_classification(outputs['cls_output'], labels)
            total_loss += loss.item()

            # Update progress bar with current metrics
            current_metrics = self.metrics.get_averages()['classification']
            pbar.set_postfix({
                'cls_loss': f'{loss.item():.4f}',
                'acc': f'{current_metrics["top1"]:.2f}%',
                'top5': f'{current_metrics["top5"]:.2f}%',
                'lr': f"{float(self.optimizer.param_groups[0]['lr']):.4e}"
            })

            # Clear batch memory
            del images, labels, outputs, loss

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Calculate and log final metrics
        metrics = self.metrics.get_averages()['classification']
        metrics['loss'] = total_loss / len(val_loader)
        self._log_metrics(metrics, prefix='val_cls')

        return metrics


    @torch.no_grad()
    def evaluate_verification(
        self,
        val_loader: DataLoader,
        return_details: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
        """
        Evaluate verification performance on paired dataset.

        Args:
            val_loader: Validation/test data loader
            return_details: Whether to return detailed results DataFrame

        Returns:
            Dictionary of metrics, and optionally a DataFrame with detailed results
        """

        self.model.eval()
        self.metrics.reset()

        results = []
        all_scores = []
        all_labels = []

        # Initial memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Create progress bar
        pbar = tqdm(val_loader, desc='Evaluating pairs', leave=True)

        for (img1_batch, img2_batch, labels) in pbar:
            # Move data to device
            images = torch.cat([img1_batch, img2_batch], dim=0).to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(images)
            embeddings = outputs['embedding']

            # Split embeddings and normalize
            embeddings = F.normalize(embeddings, dim=1)
            emb1, emb2 = embeddings.chunk(2)

            # Compute similarity similarities
            similarities = F.cosine_similarity(emb1, emb2, dim=1)

            all_scores.extend(similarities.cpu().tolist())
            all_labels.extend(labels.tolist())

            # Clear batch memory
            del images, labels, emb1, emb2, similarities

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Calculate and log metrics
        metrics = self.metrics.update_verification(all_scores, all_labels)
        self._log_metrics(metrics, prefix='val_ver')

        # Create results
        threshold = metrics['threshold']
        results = [{
            'similarity': score,
            'prediction': int(score >= threshold),
            'ground_truth': label
        } for score, label in zip(all_scores, all_labels)]

        if return_details:
            df = pd.DataFrame(results)
            df['correct'] = df['prediction'] == df['ground_truth']
            return metrics, df

        return metrics


    def evaluate(
        self, 
        cls_test_loader: DataLoader, 
        ver_test_loader: DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate model on test set.

        Args:
            cls_test_loader: Classification test loader
            ver_test_loader: Verification test loader

        Returns:
            Tuple of classification and verification metrics
        """
        test_cls_metrics = self.evaluate_classification(cls_test_loader)
        test_ver_metrics = self.evaluate_verification(ver_test_loader)

        self._log_test_results(test_cls_metrics, test_ver_metrics)

        return test_cls_metrics, test_ver_metrics
    
    # --------------------------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------------------------  
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint with AMP state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_cls_acc': self.best_cls_acc if hasattr(self, 'best_cls_acc') else None,
            'best_ver_acc': self.best_ver_acc if hasattr(self, 'best_ver_acc') else None,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_scheduler: bool = False, load_optimizer: bool = False) -> None:
        """Load model checkpoint with AMP state."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict'] and load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict'] and load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_cls_acc = checkpoint['best_cls_acc']
        self.best_ver_acc = checkpoint['best_ver_acc']

        print(f"Loaded checkpoint from {path}")
        print(f"├── Optimizer: {'Loaded' if load_optimizer else 'Not Loaded using config'}")
        print(f"├── Scheduler: {'Loaded' if load_scheduler else 'Not Loaded using config'}")
        print(f"└── Best Classification Accuracy: {self.best_cls_acc:.2f}%")
        print(f"└── Best Verification Accuracy: {self.best_ver_acc:.2f}%")  

    # --------------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------------  
    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        cls_metrics: Optional[Dict[str, float]],
        ver_metrics: Optional[Dict[str, float]]) -> None:
        """Log detailed epoch results."""
        print(f"\nEpoch {epoch} Results:")
        print("-" * 50)

        print("\nTraining:")
        print(f"├── Loss: {train_metrics['loss']:.4f}")
        print(f"├── Top-1 Acc: {train_metrics['top1']:.2f}%")
        print(f"└── Top-5 Acc: {train_metrics['top5']:.2f}%")

        if cls_metrics:
            print("\nClassification Validation:")
            print(f"├── Loss: {cls_metrics['loss']:.4f}")
            print(f"├── Top-1 Acc: {cls_metrics['top1']:.2f}%")
            print(f"└── Top-5 Acc: {cls_metrics['top5']:.2f}%")

        if ver_metrics:
            print("\nVerification Validation:")
            print(f"├── Threshold: {ver_metrics['threshold']:.4f}")
            print(f"├── Accuracy: {ver_metrics['ACC']:.2f}%")
            print(f"├── EER: {ver_metrics['EER']:.2f}%")
            print(f"└── AUC: {ver_metrics['AUC']:.2f}%")
            if 'TPRs' in ver_metrics:
                print("\nTPR at different FPRs:")
                for name, value in ver_metrics['TPRs']:
                    print(f"└── {name}: {value:.2f}%")
        print("\n" + "="*50)


    def _log_test_results(
        self,
        test_cls_metrics: Dict[str, float],
        test_ver_metrics: Dict[str, float]
    ) -> None:
        """Log test results."""
        print(f"\nTest Results:")
        print("-" * 50)

        print("\nClassification Test:")
        print(f"├── Loss: {test_cls_metrics['loss']:.4f}")
        print(f"├── Top-1 Acc: {test_cls_metrics['top1']:.2f}%")
        print(f"└── Top-5 Acc: {test_cls_metrics['top5']:.2f}%")

        print("\nVerification Test:")
        print(f"├── Accuracy: {test_ver_metrics['ACC']:.2f}%")
        print(f"├── EER: {test_ver_metrics['EER']:.2f}%")
        print(f"└── AUC: {test_ver_metrics['AUC']:.2f}%")

        if 'TPRs' in test_ver_metrics:
            print("\nTPR at different FPRs:")
            for name, value in test_ver_metrics['TPRs']:
                print(f"└── {name}: {value:.2f}%")

        print("\n" + "="*50)


    def _log_metrics(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """Log metrics to wandb if available."""
        if self.wandb_run is not None:
            # Add prefix to metric names if provided
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            self.wandb_run.log(metrics)

    
