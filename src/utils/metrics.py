import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class BraTSMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        # 'mean_batch' returns a tensor of size [3] (one for each region) 
        # instead of a single scalar, allowing per-region access.
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, 
            percentile=95, 
            reduction="mean_batch"
        )

    def _get_brats_regions(self, mask):
        """
        Input mask: (B, D, H, W) or (B, 1, D, H, W) with values {0, 1, 2, 4}
        Returns: (B, 3, D, H, W) with binary channels for WT, TC, ET
        """
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
            
        b, _, d, h, w = mask.shape
        # Ensure we create the regions tensor on the same device as the mask
        regions = torch.zeros((b, 3, d, h, w), device=mask.device)
        
        # Region logic: WT (1+2+4), TC (1+4), ET (4)
        regions[:, 0] = (mask == 1) | (mask == 2) | (mask == 4)
        regions[:, 1] = (mask == 1) | (mask == 4)
        regions[:, 2] = (mask == 4)
        
        return regions

    @torch.no_grad()
    def compute_metrics(self, logits, ground_truth):
        """
        Args:
            logits: (B, 4, D, H, W) - Raw model output
            ground_truth: (B, D, H, W) - Manual labels {0, 1, 2, 4}
        """
        # 1. Convert logits to class indices (0, 1, 2, 3)
        prediction_map = torch.argmax(logits, dim=1) 
        
        # 2. Remap index 3 back to BraTS label 4
        # We use a copy to avoid in-place issues with autograd (though we are in no_grad)
        prediction_map = prediction_map.clone()
        prediction_map[prediction_map == 3] = 4
        
        # 3. Create 3-channel binary regions
        p_regions = self._get_brats_regions(prediction_map)
        g_regions = self._get_brats_regions(ground_truth)

        # 4. Dice Calculation
        self.dice_metric(y_pred=p_regions, y=g_regions)
        dice_agg = self.dice_metric.aggregate() # Shape: [3]
        self.dice_metric.reset()
        
        # Convert to numpy for the dictionary
        dice_values = dice_agg.cpu().numpy()

        # 5. HD95 Calculation
        # spacing=(1,1,1) is critical for BraTS mm units
        self.hd95_metric(y_pred=p_regions, y=g_regions, spacing=(1, 1, 1))
        hd95_agg = self.hd95_metric.aggregate() # Shape: [3]
        self.hd95_metric.reset()
        
        hd95_values = hd95_agg.cpu().numpy()

        return {
            "dice_wt": float(dice_values[0]),
            "dice_tc": float(dice_values[1]),
            "dice_et": float(dice_values[2]),
            "mean_dice": float(np.mean(dice_values)),
            "hd95_wt": float(hd95_values[0]),
            "hd95_tc": float(hd95_values[1]),
            "hd95_et": float(hd95_values[2]),
            "mean_hd95": float(np.mean(hd95_values))
        }

def compute_metrics(logits, ground_truth, device="cuda"):
    calculator = BraTSMetrics(device=device)
    return calculator.compute_metrics(logits, ground_truth)