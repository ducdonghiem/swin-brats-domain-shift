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

        # HD95 is expensive (scipy float64 distance transforms, ~5 GB CPU RAM per batch).
        # Hence compute it only every hd95_interval epochs during training, not every epoch.
        self.hd95_interval = config['training'].get('hd95_interval', 0)

        # Mixed precision training
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device == 'cuda' else None
        
        # Enable gradient checkpointing
        if config['training'].get('gradient_checkpointing', False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
            elif hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                # Fallback: apply to all SwinEncoderStage / SwinDecoderStage / Bottleneck submodules
                # that expose a use_checkpoint flag (standard in timm-style Swin implementations)
                _gc_applied = False
                for module in self.model.modules():
                    if hasattr(module, 'use_checkpoint'):
                        module.use_checkpoint = True
                        _gc_applied = True
                if _gc_applied:
                    print("Gradient checkpointing enabled via module.use_checkpoint flags.")
                else:
                    print("Warning: gradient_checkpointing=True in config but model has no "
                          "supported checkpointing interface. Checkpointing NOT active.")

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
        self.model.train()

        train_loss = 0.0

        train_loader = tqdm(self.training_loader, desc="Training", leave=False)
        for _, (inputs, labels) in enumerate(train_loader):
            # Move each modality in inputs to the target device; inputs is a list of tensors.
            inputs = [x.to(self.device) for x in inputs]
            labels = labels.to(self.device)

            # Zero gradients before forward pass to avoid accumulating stale gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
            
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
            
            del outputs, loss

        # Calculate epoch metrics
        avg_loss = train_loss / len(self.training_loader)

        return avg_loss

    def _validate_epoch(self, epoch=0):
        self.model.eval()

        # Compute HD95 only every hd95_interval epochs
        compute_hd95 = (
            self.hd95_interval > 0 and
            (epoch + 1) % self.hd95_interval == 0
        )

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

        with torch.no_grad(): 
            val_loader = tqdm(self.val_loader, desc="Validating", leave=False)
            for inputs, labels in val_loader:
                # Move inputs to device
                inputs = [x.to(self.device) for x in inputs]
                labels = labels.to(self.device)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                val_loss += loss.item()
                del loss

                # Move outputs and labels to CPU before metric computation.
                outputs_cpu = outputs.detach().cpu()
                labels_cpu = labels.detach().cpu()

                # Free GPU tensors
                del outputs, inputs, labels
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

                # Compute metrics on CPU
                batch_metrics = self.metrics.compute_metrics(outputs_cpu, labels_cpu, compute_hd95=compute_hd95)
                batch_size = labels_cpu.size(0)
                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key] * batch_size
                metric_count += batch_size

                del outputs_cpu, labels_cpu

                avg_loss = val_loss / max(1, len(val_loader))
                avg_mean_dice = metric_sums['mean_dice'] / max(1, metric_count)
                val_loader.set_postfix(loss=f"{avg_loss:.4f}", mean_dice=f"{avg_mean_dice:.4f}")

        # Calculate epoch metrics
        avg_loss = val_loss / len(self.val_loader)

        avg_metrics = {
            key: (metric_sums[key] / metric_count if metric_count else 0.0)
            for key in metric_sums
        }

        return avg_loss, avg_metrics

    def _check_early_stopping(self, current_score):
        if not self.early_stopping_enabled:
            return False
            
        if self.early_stopping_best_score is None:
            self.early_stopping_best_score = current_score
            return False
            
        if self.early_stopping_mode == 'max':
            improved = current_score > (self.early_stopping_best_score + self.early_stopping_min_delta)
        else:
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

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch + 1}.pth'
        )

        torch.save(checkpoint, checkpoint_path)

    def train(self):
        for epoch in range(self.epochs + self.warmup_epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs + self.warmup_epochs}")
            
            # Apply learning rate warmup if needed
            _warmup_lr = self._warmup(epoch)

            # Train and validate one epoch
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate_epoch(epoch)

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

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Mean Dice: {val_metrics['mean_dice']:.4f} | Val Dice WT: {val_metrics['dice_wt']:.4f} | Val Dice TC: {val_metrics['dice_tc']:.4f} | Val Dice ET: {val_metrics['dice_et']:.4f}")
            if not (val_metrics['mean_hd95'] != val_metrics['mean_hd95']):
                print(f"Val Mean HD95: {val_metrics['mean_hd95']:.4f} | Val HD95 WT: {val_metrics['hd95_wt']:.4f} | Val HD95 TC: {val_metrics['hd95_tc']:.4f} | Val HD95 ET: {val_metrics['hd95_et']:.4f}")
            else:
                print("Val HD95: n/a this epoch (computed every hd95_interval epochs)")
            print(f"Learning Rate: {current_lr:.6f}")

            is_best = val_metrics['mean_dice'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['mean_dice']
                # Save best model immediately
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
        print("\n" + "="*50)
        print("Testing on test set...")
        print("="*50)
        
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
                del loss

                outputs_cpu = outputs.detach().cpu()
                labels_cpu = labels.detach().cpu()
                del outputs, inputs, labels
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

                batch_metrics = self.metrics.compute_metrics(outputs_cpu, labels_cpu, compute_hd95=True)
                batch_size = labels_cpu.size(0)
                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key] * batch_size
                metric_count += batch_size

                del outputs_cpu, labels_cpu
        
        avg_loss = test_loss / len(test_loader)
        avg_metrics = {
            key: (metric_sums[key] / metric_count if metric_count else 0.0)
            for key in metric_sums
        }
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Mean Dice: {avg_metrics['mean_dice']:.4f} | Test Dice WT: {avg_metrics['dice_wt']:.4f} | Test Dice TC: {avg_metrics['dice_tc']:.4f} | Test Dice ET: {avg_metrics['dice_et']:.4f}")
        print(f"Test Mean HD95: {avg_metrics['mean_hd95']:.4f} | Test HD95 WT: {avg_metrics['hd95_wt']:.4f} | Test HD95 TC: {avg_metrics['hd95_tc']:.4f} | Test HD95 ET: {avg_metrics['hd95_et']:.4f}")
        print("="*50)
        
        return avg_metrics

    def save_loss_plot(self, results_dir):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping loss plot.")
            return

        os.makedirs(results_dir, exist_ok=True)

        train_loss = self.history['train_loss']
        val_loss   = self.history['val_loss']
        epochs     = list(range(1, len(train_loss) + 1))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, label='Train Loss',      color='steelblue', linewidth=1.5)
        ax.plot(epochs, val_loss,   label='Validation Loss', color='darkorange', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training vs Validation Loss ({len(epochs)} epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if self.history['val_mean_dice']:
            best_epoch = int(max(range(len(self.history['val_mean_dice'])),
                                 key=lambda i: self.history['val_mean_dice'][i])) + 1
            best_dice  = max(self.history['val_mean_dice'])
            ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.6,
                       label=f'Best Dice epoch {best_epoch} ({best_dice:.4f})')
            ax.legend()

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(results_dir, f'loss_plot_{timestamp}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Loss plot saved to {plot_path}")
        return plot_path

    def save_results(self, results_dir, test_metrics=None):
        import json
        from datetime import datetime

        def _safe_float(v):
            import math
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return float(v)
        
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
            'best_validation_metric': _safe_float(self.best_metric),
            'final_epoch': len(self.history['train_loss']),
            'training_history': {
                'train_loss':    [_safe_float(x) for x in self.history['train_loss']],
                'val_loss':      [_safe_float(x) for x in self.history['val_loss']],
                'val_mean_dice': [_safe_float(x) for x in self.history['val_mean_dice']],
                'val_dice_wt':   [_safe_float(x) for x in self.history['val_dice_wt']],
                'val_dice_tc':   [_safe_float(x) for x in self.history['val_dice_tc']],
                'val_dice_et':   [_safe_float(x) for x in self.history['val_dice_et']],
                'val_hd95_wt':   [_safe_float(x) for x in self.history['val_hd95_wt']],
                'val_hd95_tc':   [_safe_float(x) for x in self.history['val_hd95_tc']],
                'val_hd95_et':   [_safe_float(x) for x in self.history['val_hd95_et']],
                'val_mean_hd95': [_safe_float(x) for x in self.history['val_mean_hd95']],
                'learning_rate': [_safe_float(x) for x in self.history['learning_rate']],
            }
        }
        
        if test_metrics is not None:
            results['test_metrics'] = {k: _safe_float(v) for k, v in test_metrics.items()}
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return results_file