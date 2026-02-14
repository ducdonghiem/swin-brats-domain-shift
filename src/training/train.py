from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models" / "SwinTransformers"))
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss

def main():
    train_config = load_config('src/configs/train_config.yml')

    # TODO: Data augmentation
    # TODO: Data loading

    device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')
    model = None # TODO: Initialize model
    loss = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=train_config['training']['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config['training']['epochs'])
    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=None,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config
    )

    history = trainer.train()

    # NOTE: optionally plot metrics after training

if __name__ == "__main__":
    main()