
import os
import torch

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
            epochs: Number of training epochs (excluding warmup)
            warmup_epochs: Number of warmup epochs for learning rate scheduler
            learning_rate: Initial learning rate for optimizer
            device: e.g. 'cuda' or 'cpu'
            log_interval: How often to log training progress (in batches)
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

        self.log_interval = config['training'].get('log_interval', 10)
        self.checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')


        self.best_acc = 0.0 # Best validation accuracy observed during training
        # History of training/validation metrics for analysis and visualization
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def _warmup(self, epoch):
        '''
        Warmup learning rate for the first `n` epochs to stabilize training.

        Args:
            epoch (int): Current epoch
        '''

        if epoch < self.warmup_epochs:
            warmup_factor = self.learning_rate * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_factor

            return warmup_factor
        return None

    def _train_epoch(self):
        '''
        Iteration for one training epoch. 

        Returns:
            Average loss and accuracy for the epoch.
        '''

        self.model.train() # Set model to training mode

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for _, (inputs,labels) in enumerate(self.training_loader):
            # Move each modality in inputs to the target device; inputs is a list of tensors.
            inputs = [x.to(self.device) for x in inputs]
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(*inputs)
            loss = self.loss_fn(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute training accuracy for this batch
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Calculate epoch metrics
        avg_loss = train_loss / len(self.training_loader)
        avg_acc = 100. * train_correct / train_total

        return avg_loss, avg_acc
    
    def _validate_epoch(self):
        '''
        Iteration for one validation epoch.
        Returns:
            Average loss and accuracy for the epoch.
        '''

        self.model.eval() # Set model to evaluation mode

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # Disable gradient computation for validation
            for inputs, labels in self.val_loader:
                # Move inputs to device
                inputs = [x.to(self.device) for x in inputs]
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(*inputs)
                loss = self.loss_fn(outputs, labels)

                # Track metrics
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Calculate epoch metrics
        avg_loss = val_loss / len(self.val_loader)
        avg_acc = 100. * val_correct / val_total

        return avg_loss, avg_acc

    def _save_checkpoint(self, epoch, is_best=False):
        '''
        Save model checkpoint for the current epoch. If this is the best model so far, also save a copy as 'best_model.pth'.

        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this checkpoint has the best validation accuracy
        '''

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )

        torch.save(checkpoint, checkpoint_path) # Save checkpoint for this epoch

        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pth'))

    def train(self):
        '''
        Main training loop.

        Returns:
            History of training and validation metrics across epochs.
        '''

        for epoch in range(self.epochs + self.warmup_epochs):
            # Apply learning rate warmup if needed
            _warmup_lr = self._warmup(epoch)

            # Train and validate one epoch
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            # Step the learning rate scheduler after warmup epochs
            if epoch >= self.warmup_epochs:
                self.scheduler.step() 

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc

            if (epoch + 1) % self.checkpoint_interval == 0 or is_best:
                self._save_checkpoint(epoch, is_best=is_best)

        return self.history

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        self.history = checkpoint['history']

        return checkpoint['epoch']
