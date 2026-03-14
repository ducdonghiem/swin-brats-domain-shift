from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import torchvision.transforms.v2 as T # v2 can apply the same transformation to multiple inputs at once

from .trainer import SwinTrainer
from data import MRIDataset, collate_modalities
from utils import BraTSLoss, load_config
from models import SwinBraTS

if __name__ == "__main__":
    '''
    Test example
    '''
    train_config = load_config('configs/train_config.yml')

    # Data loading
    train_dir = Path(train_config['data']['train_dir'])
    val_dir = Path(train_config['data']['val_dir'])
    test_dir = Path(train_config['data']['test_dir'])
    modality_order = train_config['data']['modality_order']

    # Data augmentation
    transform = T.Compose([T.RandomRotation(train_config['data']['transform']['rotate'])])

    train_dataset = MRIDataset(
        data_dir=train_dir,
        modalities=modality_order,
        transform=transform
    )
    val_dataset = MRIDataset(
        data_dir=val_dir,
        modalities=modality_order
    )
    test_dataset = MRIDataset(
        data_dir=test_dir,
        modalities=modality_order
    )

    batch_size = train_config['training']['batch_size']
    num_workers = train_config['training']['num_workers']
    pin_memory = train_config['data']['pin_memory']
    shuffle_train = train_config['data']['shuffle_train']

    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_modalities
    )
    if len(val_dataset) == 0:
        val_loader = training_loader
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_modalities
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_modalities
    )

    # Hyperparameters and model setup
    model = SwinBraTS(
        in_channels=train_config['model']['in_channels'],
        num_classes=train_config['model']['num_classes'],
        embed_dim=train_config['model']['embed_dim'],
        window_size=train_config['model']['window_size'],
        patch_size=train_config['model']['patch_size']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    proj_params  = sum(p.numel() for p in model.projection_block.parameters())
    swin_params  = sum(p.numel() for p in model.swin_backbone.parameters())
    recon_params = sum(p.numel() for p in model.reconstruction_block.parameters())
    print(f"  ProjectionBlock:      {proj_params:,}")
    print(f"  SwinUNet backbone:    {swin_params:,}")
    print(f"  ReconstructionBlock:  {recon_params:,}")

    loss = BraTSLoss(device=train_config['device'])
    optimizer = AdamW(model.parameters(),
        lr=train_config['training']['learning_rate'])

    # eta_min prevents LR from decaying all the way to 0, which causes the
    # loss spike seen near the end of cosine annealing.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config['training']['epochs'],
        eta_min=1e-6
    )

    trainer = SwinTrainer(
        model=model,
        training_loader=training_loader,
        val_loader=val_loader,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config
    )

    history = trainer.train()

    print("\n" + "="*50)
    print(f"Best validation Mean Dice: {trainer.best_metric:.4f}")
    print("="*50)

    # Save loss plot immediately after training
    trainer.save_loss_plot(train_config['training']['results_dir'])
    
    # Load best model and evaluate on test set
    best_checkpoint_path = Path(train_config['training']['checkpoint_dir']) / 'best_model.pth'
    test_metrics = None
    if best_checkpoint_path.exists():
        print(f"\nLoading best model from {best_checkpoint_path}")
        trainer.load_checkpoint(best_checkpoint_path)
        test_metrics = trainer.test(test_loader)
    else:
        print("\nWarning: No best model checkpoint found, skipping test evaluation")
    
    # Save results
    results_dir = train_config['training']['results_dir']
    trainer.save_results(results_dir, test_metrics)