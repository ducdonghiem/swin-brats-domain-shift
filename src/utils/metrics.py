import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class BraTSMetrics:
    def __init__(self, device="cuda"):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, 
            percentile=95, 
            reduction="mean"
        )
        self.device = device

    def _get_brats_regions(self, mask):
        # mask shape: (B, 1, 155, 240, 240) or (B, 155, 240, 240)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
            
        batch_size, h, w, d = mask.shape
        regions = torch.zeros((batch_size, 3, h, w, d), device=self.device)
        
        # Region logic remains the same (WT, TC, ET)
        regions[:, 0] = (mask == 1) | (mask == 2) | (mask == 4)
        regions[:, 1] = (mask == 1) | (mask == 4)
        regions[:, 2] = (mask == 4)
        return regions

    @torch.no_grad()
    def compute_metrics(self, logits, ground_truth):
        """
        Args:
            logits: (Batch, 4, 155, 240, 240) -> From Model
            ground_truth: (Batch, 155, 240, 240) -> Values {0, 1, 2, 4}
        """
        # 1. Convert logits to label map (0, 1, 2, 3)
        prediction_map = torch.argmax(logits, dim=1) 
        
        # 2. Remap prediction 3 back to 4 for BraTS logic
        prediction_map[prediction_map == 3] = 4
        
        # 3. Get regions
        p_regions = self._get_brats_regions(prediction_map)
        g_regions = self._get_brats_regions(ground_truth)

        # 4. Dice
        self.dice_metric(y_pred=p_regions, y=g_regions)
        dice_values = self.dice_metric.aggregate().cpu().numpy()[0] 
        self.dice_metric.reset()

        # 5. HD95
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

# Helper
def compute_metrics(logits, ground_truth, device="cuda"):
    calculator = BraTSMetrics(device=device)
    return calculator.compute_metrics(logits, ground_truth)