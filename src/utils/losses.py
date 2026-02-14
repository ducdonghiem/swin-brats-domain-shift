import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss

class BraTSLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(BraTSLoss, self).__init__()
        # DiceFocalLoss combines:
        # 1. Dice Loss: Good for global overlap and class imbalance.
        # 2. Focal Loss: Forces the model to focus on "hard" pixels (like ET).
        self.main_loss = DiceFocalLoss(
            softmax=True,            # Apply softmax to model logits
            to_onehot_y=True,        # Convert (240,240,155) labels to 4-channel one-hot
            include_background=True, # In training, we include background to stabilize CE/Focal
            gamma=2.0,               # Focal loss focusing parameter (standard is 2.0)
            lambda_dice=1.0,         # Weight for Dice component
            lambda_focal=1.0         # Weight for Focal component
        )
        self.device = device

    def forward(self, prediction, target):
        """
        Args:
            prediction: Raw logits from model (Batch, 4, 240, 240, 155)
            target: Ground truth labels (Batch, 1, 240, 240, 155) or (Batch, 240, 240, 155)
        """
        # Ensure target has the channel dimension for MONAI (Batch, 1, H, W, D)
        if len(target.shape) == 4:
            target = target.unsqueeze(1)
            
        return self.main_loss(prediction, target)

# Helper function for external use
def compute_loss(prediction, target, device="cuda"):
    criterion = BraTSLoss(device=device)
    return criterion(prediction, target)