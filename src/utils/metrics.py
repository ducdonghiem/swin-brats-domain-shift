import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class BraTSMetrics:
    def __init__(self, device="cuda"):
        # We use include_background=False because we handle the 
        # WT, TC, ET grouping manually as 3 separate binary channels.
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, 
            percentile=95, 
            reduction="mean"
        )
        self.device = device

    def _get_brats_regions(self, mask):
        """
        Converts a 4-class label map (0, 1, 2, 4) into 3 overlapping 
        binary channels for BraTS regions: WT, TC, ET.
        Input shape: (Batch, 240, 240, 155) or (240, 240, 155)
        Output shape: (Batch, 3, 240, 240, 155)
        """
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0) # Add batch dim if missing
            
        # Create empty tensor for 3 regions
        batch_size, h, w, d = mask.shape
        regions = torch.zeros((batch_size, 3, h, w, d), device=self.device)
        
        # Region 1: Whole Tumor (WT) = Labels 1 + 2 + 4
        regions[:, 0] = (mask == 1) | (mask == 2) | (mask == 4)
        
        # Region 2: Tumor Core (TC) = Labels 1 + 4
        regions[:, 1] = (mask == 1) | (mask == 4)
        
        # Region 3: Enhancing Tumor (ET) = Label 4
        regions[:, 2] = (mask == 4)
        
        return regions

    @torch.no_grad()
    def compute_metrics(self, prediction, ground_truth):
        """
        Args:
            prediction: Tensor of shape (240, 240, 155) with values {0, 1, 2, 4}
            ground_truth: Tensor of shape (240, 240, 155) with values {0, 1, 2, 4}
        Returns:
            Dictionary with Dice and HD95 for each region + Mean Dice
        """
        # Ensure they are on the right device and grouped correctly
        p_regions = self._get_brats_regions(prediction)
        g_regions = self._get_brats_regions(ground_truth)

        # 1. Compute Dice
        self.dice_metric(y_pred=p_regions, y=g_regions)
        dice_values = self.dice_metric.aggregate().cpu().numpy()[0] # [dsc_wt, dsc_tc, dsc_et]
        self.dice_metric.reset()

        # 2. Compute HD95
        # spacing=(1,1,1) because BraTS data is resampled to 1mm isotropic
        self.hd95_metric(y_pred=p_regions, y=g_regions, spacing=(1, 1, 1))
        hd95_values = self.hd95_metric.aggregate().cpu().numpy()[0]
        self.hd95_metric.reset()

        return {
            "dice_wt": dice_values[0],
            "dice_tc": dice_values[1],
            "dice_et": dice_values[2],
            "mean_dice": np.mean(dice_values),
            "hd95_wt": hd95_values[0],
            "hd95_tc": hd95_values[1],
            "hd95_et": hd95_values[2],
            "mean_hd95": np.mean(hd95_values)
        }

# Helper function for external use
def compute_metrics(prediction, ground_truth, device="cuda"):
    calculator = BraTSMetrics(device=device)
    return calculator.compute_metrics(prediction, ground_truth)