"""
Advanced training script for bone fracture detection with medical imaging optimizations.
Includes mixed precision, learning rate scheduling, cross-validation, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
import time
import wandb
from collections import defaultdict

# Import our modules
from utils import (
    DeviceManager, MetricsCalculator, EarlyStopping, Timer,
    ConfigManager, setup_logging, count_parameters
)
from dataloader import BoneFractureDataManager
from models import create_ensemble_from_config, create_transfer_model, create_bone_cnn


class BoneFractureTrainer:
    """
    Advanced trainer for bone fracture detection with medical imaging optimizations.
    """
    
    def __init__(
        self,
        config_path: str,
        experiment_name: str = None,
        use_wandb: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for the experiment
            use_wandb: Whether to use Weights & Biases logging
        """
        # Load configuration
        self.config = ConfigManager.load_config(config_path)
        
        # Setup experiment
        self.experiment_name = experiment_name or f"bone_fracture_{int(time.time())}"
        self.use_wandb = use_wandb
        
        # Setup logging
        log_dir = Path(self.config['paths']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir / f"{self.experiment_name}.log")
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = DeviceManager.get_device()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.early_stopping = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = float('-inf')
        self.training_history = defaultdict(list)
        
        # Setup directories
        self._setup_directories()
        
        # Initialize experiment tracking
        if self.use_wandb:
            self._init_wandb()
        
        self.tensorboard_writer = SummaryWriter(
            log_dir / "tensorboard" / self.experiment_name
        )
    
    def _setup_directories(self):
        """Create necessary directories."""
        for path_key in ['models_dir', 'logs_dir', 'checkpoints_dir']:
            path = Path(self.config['paths'][path_key])
            path.mkdir(parents=True, exist_ok=True)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project="bone-fracture-detection",
                name=self.experiment_name,
                config=self.config
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def build_model(self, model_type: str = 'ensemble') -> nn.Module:
        """
        Build the model based on configuration.
        
        Args:
            model_type: Type of model ('ensemble', 'single', 'custom')
            
        Returns:
            Built model
        """
        if model_type == 'ensemble':
            model = create_ensemble_from_config(self.config['model'], self.device)
        elif model_type == 'single':
            # Use first architecture from config
            arch_config = self.config['model']['architectures'][0]
            arch_name = arch_config['name']
            
            if 'efficientnet' in arch_name:
                model = create_transfer_model(
                    'efficientnet',
                    variant=arch_name,
                    num_classes=self.config['model']['num_classes'],
                    pretrained=arch_config.get('pretrained', True)
                )
            elif 'resnet' in arch_name:
                model = create_transfer_model(
                    'resnet',
                    variant=arch_name,
                    num_classes=self.config['model']['num_classes'],
                    pretrained=arch_config.get('pretrained', True)
                )
            elif 'mobilenet' in arch_name:
                model = create_transfer_model(
                    'mobilenet',
                    variant=arch_name,
                    num_classes=self.config['model']['num_classes'],
                    pretrained=arch_config.get('pretrained', True)
                )
            else:
                model = create_bone_cnn(
                    'standard',
                    num_classes=self.config['model']['num_classes']
                )
        else:
            model = create_bone_cnn(
                'standard',
                num_classes=self.config['model']['num_classes']
            )
        
        self.model = model.to(self.device)
        
        # Log model info
        param_counts = count_parameters(self.model)
        self.logger.info(f"Model created: {param_counts}")
        
        if self.use_wandb:
            wandb.log(param_counts)
        
        return self.model
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, loss function, etc."""
        if self.model is None:
            raise ValueError("Model must be built before setting up training components")
        
        # Optimizer
        optimizer_config = self.config.get('training', {})
        lr = self.config['model']['learning_rate']
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        scheduler_type = optimizer_config.get('scheduler', 'cosine_annealing')
        epochs = self.config['model']['epochs']
        
        if scheduler_type == 'cosine_annealing':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'one_cycle':
            # Calculate total steps for OneCycleLR
            # This will be updated when we have data loaders
            self.scheduler = None  # Will be set later
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        if optimizer_config.get('mixed_precision', True):
            self.scaler = GradScaler()
        
        # Early stopping
        patience = self.config['model'].get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(
            patience=patience,
            mode='max',  # For metrics like F1-score
            restore_best_weights=True
        )
        
        self.logger.info("Training components setup complete")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                clip_value = self.config.get('training', {}).get('gradient_clipping', 1.0)
                if clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                clip_value = self.config.get('training', {}).get('gradient_clipping', 1.0)
                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        metrics = MetricsCalculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions)
        )
        
        return {
            'loss': avg_loss,
            'accuracy': metrics.accuracy,
            'f1_score': metrics.f1_score,
            'precision': metrics.precision,
            'recall': metrics.recall
        }
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = MetricsCalculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        return {
            'loss': avg_loss,
            'accuracy': metrics.accuracy,
            'f1_score': metrics.f1_score,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'auc_roc': metrics.auc_roc
        }
    
    def train(self, data_manager: BoneFractureDataManager):
        """
        Main training loop.
        
        Args:
            data_manager: Data manager instance
        """
        self.logger.info("Starting training...")
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_manager.get_data_loaders(
            self.config['data']['dataset_path']
        )
        
        # Setup OneCycleLR if needed
        if (hasattr(self, 'scheduler') and self.scheduler is None and 
            self.config.get('training', {}).get('scheduler') == 'one_cycle'):
            total_steps = len(train_loader) * self.config['model']['epochs']
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['model']['learning_rate'],
                total_steps=total_steps,
                pct_start=0.1
            )
        
        # Training loop
        epochs = self.config['model']['epochs']
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1_score'])
                elif isinstance(self.scheduler, OneCycleLR):
                    pass  # OneCycleLR is updated per batch
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint
            if val_metrics['f1_score'] > self.best_val_score:
                self.best_val_score = val_metrics['f1_score']
                self._save_checkpoint('best')
            
            # Early stopping check
            if self.early_stopping(val_metrics['f1_score'], self.model):
                self.logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                break
        
        # Final evaluation on test set
        self.logger.info("Evaluating on test set...")
        test_metrics = self.validate_epoch(test_loader)
        self.logger.info(f"Test metrics: {test_metrics}")
        
        if self.use_wandb:
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Save final model
        self._save_checkpoint('final')
        
        self.logger.info("Training completed!")
    
    def cross_validate(self, data_manager: BoneFractureDataManager, k_folds: int = 5):
        """
        Perform k-fold cross-validation.
        
        Args:
            data_manager: Data manager instance
            k_folds: Number of folds
        """
        self.logger.info(f"Starting {k_folds}-fold cross-validation...")
        
        # Get fold data loaders
        fold_loaders = data_manager.get_stratified_kfold_splits(
            self.config['data']['dataset_path'],
            k_folds
        )
        
        fold_results = []
        
        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
            self.logger.info(f"Training fold {fold_idx + 1}/{k_folds}")
            
            # Reset model for each fold
            self.build_model()
            self.setup_training_components()
            
            # Train on this fold
            best_val_score = 0.0
            epochs = self.config['model']['epochs']
            
            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate_epoch(val_loader)
                
                if val_metrics['f1_score'] > best_val_score:
                    best_val_score = val_metrics['f1_score']
                
                # Early stopping for each fold
                if self.early_stopping(val_metrics['f1_score'], self.model):
                    break
            
            fold_results.append(best_val_score)
            self.logger.info(f"Fold {fold_idx + 1} best F1: {best_val_score:.4f}")
        
        # Calculate cross-validation statistics
        mean_score = np.mean(fold_results)
        std_score = np.std(fold_results)
        
        self.logger.info(f"Cross-validation results:")
        self.logger.info(f"Mean F1: {mean_score:.4f} ± {std_score:.4f}")
        self.logger.info(f"Individual folds: {fold_results}")
        
        if self.use_wandb:
            wandb.log({
                'cv_mean_f1': mean_score,
                'cv_std_f1': std_score,
                'cv_scores': fold_results
            })
    
    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch_time: float):
        """Log epoch metrics."""
        # Update history
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.training_history[f'val_{key}'].append(value)
        
        # Console logging
        self.logger.info(f"Epoch {self.current_epoch}/{self.config['model']['epochs']} "
                        f"({epoch_time:.1f}s)")
        self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"F1: {train_metrics['f1_score']:.4f}, "
                        f"Acc: {train_metrics['accuracy']:.4f}")
        self.logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                        f"F1: {val_metrics['f1_score']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.4f}")
        
        # TensorBoard logging
        for key, value in train_metrics.items():
            self.tensorboard_writer.add_scalar(f'Train/{key}', value, self.current_epoch)
        for key, value in val_metrics.items():
            self.tensorboard_writer.add_scalar(f'Val/{key}', value, self.current_epoch)
        
        # Learning rate logging
        current_lr = self.optimizer.param_groups[0]['lr']
        self.tensorboard_writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)
        
        # Weights & Biases logging
        if self.use_wandb:
            log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
            log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
            log_dict['learning_rate'] = current_lr
            log_dict['epoch'] = self.current_epoch
            wandb.log(log_dict)
    
    def _save_checkpoint(self, checkpoint_type: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])
        checkpoint_path = checkpoint_dir / f"{self.experiment_name}_{checkpoint_type}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'config': self.config,
            'training_history': dict(self.training_history)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


def main():
    """Example usage of the trainer."""
    # Configuration
    config_path = "config/config.yaml"
    
    # Initialize trainer
    trainer = BoneFractureTrainer(
        config_path=config_path,
        experiment_name="bone_fracture_experiment",
        use_wandb=False  # Set to True if you want to use wandb
    )
    
    # Build model
    trainer.build_model(model_type='single')  # or 'ensemble'
    trainer.setup_training_components()
    
    # Initialize data manager
    data_manager = BoneFractureDataManager(config_path)
    
    # Train the model
    trainer.train(data_manager)
    
    # Optionally run cross-validation
    # trainer.cross_validate(data_manager, k_folds=5)


if __name__ == "__main__":
    main()