import torch.nn as nn
from monai.losses import DiceFocalLoss

class BraTSLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(BraTSLoss, self).__init__()
        self.main_loss = DiceFocalLoss(
            softmax=True,            
            to_onehot_y=True,        
            include_background=True, 
            gamma=2.0,               
            lambda_dice=1.0,         
            lambda_focal=1.0         
        )
        self.device = device

    def forward(self, prediction, target):
        """
        prediction: (Batch, 4, 155, 240, 240)
        target: (Batch, 155, 240, 240) with values {0, 1, 2, 4}
        """
        # Remap label 4 to 3 so it matches the 4th channel of prediction
        # Use clone to avoid corrupting the original metadata
        target_remapped = target.clone()
        target_remapped[target == 4] = 3
        
        if len(target_remapped.shape) == 3:
            target_remapped = target_remapped.unsqueeze(1)
        elif len(target_remapped.shape) == 4:
            # If batch dim is present but channel dim is missing
            target_remapped = target_remapped.unsqueeze(1)
            
        return self.main_loss(prediction, target_remapped)