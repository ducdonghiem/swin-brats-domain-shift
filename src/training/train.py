from utils.config import load_config
from data.data_loader import MRIDataset, collate_modalities
from models.swinUNet import SwinUNet
from trainer import SwinTrainer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "models" / "SwinTransformers"))

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

    # FIXME: Requires projection and reconstruction block to wrap SwinUNet
    model = SwinUNet()
    loss = CrossEntropyLoss()
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

    # NOTE: optionally plot metrics after training
