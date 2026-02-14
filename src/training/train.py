import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.trainer import SwinTrainer
from src.data.data_loader import MRIDataset, collate_modalities
from src.utils.losses import BraTSLoss
from src.models.swinBraTS_full import SwinBraTS
from src.utils.config import load_config

if __name__ == "__main__":
    '''
    Test example
    '''
    train_config = load_config('configs/train_config.yml')

    # TODO: Data augmentation

    # === Data loading ===
    train_dir = Path("src/data/processed/train")
    val_dir = Path("src/data/processed/val")
    modality_order = ["flair", "t1", "t1ce", "t2"]

    train_dataset = MRIDataset(
        data_dir=train_dir,
        modalities=modality_order
    )
    val_dataset = MRIDataset(
        data_dir=val_dir,
        modalities=modality_order
    )

    batch_size = train_config["training"].get("batch_size", 1)
    num_workers = train_config["training"].get("num_workers", 0)

    training_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
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
            pin_memory=False,
            collate_fn=collate_modalities
        )

    # === Hyperparameters and model setup ===
    model = SwinBraTS(
        in_channels=train_config['model']['in_channels'],
        num_classes=train_config['model']['num_classes'],
        embed_dim=train_config['model']['embed_dim'],
        window_size=train_config['model']['window_size'],
        patch_size=train_config['model']['patch_size']
    )
    loss = BraTSLoss(device=train_config.get('device', 'cpu'))
    optimizer = AdamW(model.parameters(),
        lr=train_config['training']['learning_rate'])

    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_config['training']['epochs'])

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

    print("Training complete. Final validation mean Dice:", history['val_mean_dice'][-1])

    # NOTE: optionally plot metrics after training
