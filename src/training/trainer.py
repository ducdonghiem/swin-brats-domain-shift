import os
import torch
from tqdm import tqdm

from src.utils.metrics import BraTSMetrics

class SwinTrainer():

    def __init__(self, model, training_loader, val_loader, loss_fn, optimizer, scheduler, config):
        '''
        Args:
            model: Model to be trained
            training_loader: Data loader for training data
            val_loader: Data loader for validation data
            loss_fn: Loss function
            optimizer: Optimization algorithm
            scheduler: Learning rate scheduler
            config: Configuration dictionary
        '''

        self.model = model
        self.training_loader = training_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = config['training']['epochs']
        self.warmup_epochs = config['training']['warmup_epochs']
        self.learning_rate = config['training']['learning_rate']

        requested_device = config.get('device', 'cpu')
        if requested_device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = requested_device

        self.model.to(self.device)

        self.log_interval = config['training']['log_interval']
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.checkpoint_interval = config['training']['checkpoint_interval']

        # Mixed precision training
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device == 'cuda' else None
        
        # Enable gradient checkpointing if model supports it (saves memory during training)
        if config['training'].get('gradient_checkpointing', False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
            elif hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        self.best_metric = 0.0  # Best validation mean Dice observed during training
        
        # Early stopping setup
        self.early_stopping_enabled = config.get('early_stopping', {}).get('enabled', False)
        self.early_stopping_patience = config.get('early_stopping', {}).get('patience', 15)
        self.early_stopping_min_delta = config.get('early_stopping', {}).get('min_delta', 0.001)
        self.early_stopping_metric = config.get('early_stopping', {}).get('metric', 'val_mean_dice')
        self.early_stopping_mode = config.get('early_stopping', {}).get('mode', 'max')
        self.early_stopping_counter = 0
        self.early_stopping_best_score = None
        
        # History of training/validation metrics for analysis and visualization
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mean_dice': [],
            'val_dice_wt': [],
            'val_dice_tc': [],
            'val_dice_et': [],
            'val_hd95_wt': [],
            'val_hd95_tc': [],
            'val_hd95_et': [],
            'val_mean_hd95': [],
            'learning_rate': []
        }

        self.metrics = BraTSMetrics(device=self.device)

    def _warmup(self, epoch):
        '''
        Warmup learning rate for the first `n` epochs to stabilize training.

        Args:
            epoch (int): Current epoch
        '''

        if epoch < self.warmup_epochs:
            warmup_factor = self.learning_rate * \
                (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_factor

            return warmup_factor
        return None

    def _train_epoch(self):
        '''
        Iteration for one training epoch. 

        Returns:
            Average loss for the epoch.
        '''

        self.model.train()  # Set model to training mode

        train_loss = 0.0

        train_loader = tqdm(self.training_loader, desc="Training", leave=False)
        for _, (inputs, labels) in enumerate(train_loader):
            # Move each modality in inputs to the target device; inputs is a list of tensors.
            inputs = [x.to(self.device) for x in inputs]
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Track loss
            train_loss += loss.item()

            avg_loss = train_loss / max(1, len(train_loader))
            train_loader.set_postfix(loss=f"{avg_loss:.4f}")
            
            # Clear references to free memory
            del outputs, loss

        # Calculate epoch metrics
        avg_loss = train_loss / len(self.training_loader)

        return avg_loss

    def _validate_epoch(self):
        '''
        Iteration for one validation epoch.
        Returns:
            Average loss and metrics for the epoch.
        '''

        self.model.eval()  # Set model to evaluation mode

        val_loss = 0.0
        metric_sums = {
            'dice_wt': 0.0,
            'dice_tc': 0.0,
            'dice_et': 0.0,
            'mean_dice': 0.0,
            'hd95_wt': 0.0,
            'hd95_tc': 0.0,
            'hd95_et': 0.0,
            'mean_hd95': 0.0,
        }
        metric_count = 0

        with torch.no_grad():  # Disable gradient computation for validation
            val_loader = tqdm(self.val_loader, desc="Validating", leave=False)
            for inputs, labels in val_loader:
                # Move inputs to device
                inputs = [x.to(self.device) for x in inputs]
                labels = labels.to(self.device)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                # Track metrics - immediately move to CPU to save GPU memory
                val_loss += loss.item()
                
                # Detach outputs and labels before computing metrics to free computation graph
                outputs_detached = outputs.detach()
                labels_detached = labels.detach()
                
                # Compute metrics
                batch_metrics = self.metrics.compute_metrics(outputs_detached, labels_detached)
                batch_size = labels.size(0)
                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key] * batch_size
                metric_count += batch_size

                avg_loss = val_loss / max(1, len(val_loader))
                avg_mean_dice = metric_sums['mean_dice'] / max(1, metric_count)
                val_loader.set_postfix(loss=f"{avg_loss:.4f}", mean_dice=f"{avg_mean_dice:.4f}")
                
                # Clear GPU memory after each batch - delete everything
                del outputs, outputs_detached, labels_detached, loss, inputs, labels

        # Calculate epoch metrics
        avg_loss = val_loss / len(self.val_loader)

        avg_metrics = {
            key: (metric_sums[key] / metric_count if metric_count else 0.0)
            for key in metric_sums
        }

        return avg_loss, avg_metrics

    def _check_early_stopping(self, current_score):
        '''
        Check if early stopping criteria are met.
        
        Args:
            current_score (float): Current validation metric score
            
        Returns:
            bool: True if training should stop, False otherwise
        '''
        if not self.early_stopping_enabled:
            return False
            
        if self.early_stopping_best_score is None:
            self.early_stopping_best_score = current_score
            return False
            
        # Check if there's improvement based on mode
        if self.early_stopping_mode == 'max':
            improved = current_score > (self.early_stopping_best_score + self.early_stopping_min_delta)
        else:  # mode == 'min'
            improved = current_score < (self.early_stopping_best_score - self.early_stopping_min_delta)
            
        if improved:
            self.early_stopping_best_score = current_score
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                return True
            return False

    def _save_checkpoint(self, epoch, is_best=False):
        '''
        Save model checkpoint for the current epoch.

        Args:
            epoch (int): Current epoch number
            is_best (bool): Unused, kept for compatibility
        '''

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch + 1}.pth'
        )

        # Save checkpoint for this epoch
        torch.save(checkpoint, checkpoint_path)

    def train(self):
        '''
        Main training loop.

        Returns:
            History of training and validation metrics across epochs.
        '''

        for epoch in range(self.epochs + self.warmup_epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs + self.warmup_epochs}")
            
            _warmup_lr = self._warmup(epoch)

            # Train and validate one epoch
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate_epoch()

            # Step the learning rate scheduler after warmup epochs
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mean_dice'].append(val_metrics['mean_dice'])
            self.history['val_dice_wt'].append(val_metrics['dice_wt'])
            self.history['val_dice_tc'].append(val_metrics['dice_tc'])
            self.history['val_dice_et'].append(val_metrics['dice_et'])
            self.history['val_hd95_wt'].append(val_metrics['hd95_wt'])
            self.history['val_hd95_tc'].append(val_metrics['hd95_tc'])
            self.history['val_hd95_et'].append(val_metrics['hd95_et'])
            self.history['val_mean_hd95'].append(val_metrics['mean_hd95'])
            self.history['learning_rate'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Mean Dice: {val_metrics['mean_dice']:.4f} | Val Dice WT: {val_metrics['dice_wt']:.4f} | Val Dice TC: {val_metrics['dice_tc']:.4f} | Val Dice ET: {val_metrics['dice_et']:.4f}")
            print(f"Val Mean HD95: {val_metrics['mean_hd95']:.4f} | Val HD95 WT: {val_metrics['hd95_wt']:.4f} | Val HD95 TC: {val_metrics['hd95_tc']:.4f} | Val HD95 ET: {val_metrics['hd95_et']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            is_best = val_metrics['mean_dice'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['mean_dice']
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_metric': self.best_metric,
                    'history': self.history
                }
                if self.scaler is not None:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                    
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"New best model saved! Mean Dice: {self.best_metric:.4f}")

            # Save checkpoint every checkpoint_interval epochs after warmup
            if epoch >= self.warmup_epochs and (epoch - self.warmup_epochs + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch, is_best=False)
                print(f"Checkpoint saved at epoch {epoch + 1}")
                
            # Check early stopping
            early_stop_metric = val_metrics.get(self.early_stopping_metric.replace('val_', ''), val_metrics['mean_dice'])
            if self._check_early_stopping(early_stop_metric):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"No improvement in {self.early_stopping_metric} for {self.early_stopping_patience} epochs")
                break

        return self.history

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.history = checkpoint['history']

        return checkpoint['epoch']

    def test(self, test_loader):
        '''
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of test metrics
        '''
        self.model.eval()
        
        test_loss = 0.0
        metric_sums = {
            'dice_wt': 0.0,
            'dice_tc': 0.0,
            'dice_et': 0.0,
            'mean_dice': 0.0,
            'hd95_wt': 0.0,
            'hd95_tc': 0.0,
            'hd95_et': 0.0,
            'mean_hd95': 0.0,
        }
        metric_count = 0
        
        with torch.no_grad():
            test_loader_iter = tqdm(test_loader, desc="Testing", leave=False)
            for inputs, labels in test_loader_iter:
                inputs = [x.to(self.device) for x in inputs]
                labels = labels.to(self.device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                
                test_loss += loss.item()
                
                batch_metrics = self.metrics.compute_metrics(outputs, labels)
                batch_size = labels.size(0)
                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key] * batch_size
                metric_count += batch_size
                
                avg_loss = test_loss / max(1, len(test_loader_iter))
                avg_mean_dice = metric_sums['mean_dice'] / max(1, metric_count)
                test_loader_iter.set_postfix(loss=f"{avg_loss:.4f}", mean_dice=f"{avg_mean_dice:.4f}")
                
                # Clear GPU memory
                del outputs, loss, inputs, labels
        
        avg_loss = test_loss / len(test_loader)
        avg_metrics = {
            key: (metric_sums[key] / metric_count if metric_count else 0.0)
            for key in metric_sums
        }
        
        print("Test Results")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Mean Dice: {avg_metrics['mean_dice']:.4f} | Test Dice WT: {avg_metrics['dice_wt']:.4f} | Test Dice TC: {avg_metrics['dice_tc']:.4f} | Test Dice ET: {avg_metrics['dice_et']:.4f}")
        print(f"Test Mean HD95: {avg_metrics['mean_hd95']:.4f} | Test HD95 WT: {avg_metrics['hd95_wt']:.4f} | Test HD95 TC: {avg_metrics['hd95_tc']:.4f} | Test HD95 ET: {avg_metrics['hd95_et']:.4f}")
        
        return avg_metrics

    def save_results(self, results_dir, test_metrics=None):
        '''
        Save training results to a file.
        
        Args:
            results_dir: Directory to save results
            test_metrics: Optional test metrics dictionary
        '''
        import json
        from datetime import datetime
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f'results_{timestamp}.json')
        
        results = {
            'timestamp': timestamp,
            'config': {
                'epochs': self.epochs,
                'warmup_epochs': self.warmup_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': len(self.training_loader.dataset) // len(self.training_loader),
                'device': self.device,
            },
            'best_validation_metric': float(self.best_metric),
            'final_epoch': len(self.history['train_loss']),
            'training_history': {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_mean_dice': [float(x) for x in self.history['val_mean_dice']],
                'val_dice_wt': [float(x) for x in self.history['val_dice_wt']],
                'val_dice_tc': [float(x) for x in self.history['val_dice_tc']],
                'val_dice_et': [float(x) for x in self.history['val_dice_et']],
                'val_hd95_wt': [float(x) for x in self.history['val_hd95_wt']],
                'val_hd95_tc': [float(x) for x in self.history['val_hd95_tc']],
                'val_hd95_et': [float(x) for x in self.history['val_hd95_et']],
                'val_mean_hd95': [float(x) for x in self.history['val_mean_hd95']],
                'learning_rate': [float(x) for x in self.history['learning_rate']],
            }
        }
        
        if test_metrics is not None:
            results['test_metrics'] = {k: float(v) for k, v in test_metrics.items()}
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return results_file