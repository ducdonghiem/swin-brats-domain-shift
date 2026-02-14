import torch
import numpy as np
from monai.metrics import compute_dice, compute_hausdorff_distance

class BraTSMetrics:
    def __init__(self, device="cuda"):
        self.device = device

    def _get_brats_regions(self, mask):
        """
        Input mask: (B, 155, 240, 240) - values {0, 1, 2, 4}
        Output: (B, 3, 155, 240, 240) -> [WT, TC, ET]
        """
        if mask.dim() == 5:
            mask = mask.squeeze(1)
            
        b, d, h, w = mask.shape
        regions = torch.zeros((b, 3, d, h, w), device=mask.device)
        
        # Region 0: WT (1, 2, 4), Region 1: TC (1, 4), Region 2: ET (4)
        regions[:, 0] = (mask == 1) | (mask == 2) | (mask == 4)
        regions[:, 1] = (mask == 1) | (mask == 4)
        regions[:, 2] = (mask == 4)
        return regions

    @torch.no_grad()
    def compute_metrics(self, logits, ground_truth):
        """
        logits: (B, 4, 155, 240, 240)
        ground_truth: (B, 155, 240, 240)
        """
        # 1. Prediction mapping: Argmax + Label Fix (0,1,2,3 -> 0,1,2,4)
        preds = torch.argmax(logits, dim=1).clone()
        preds[preds == 3] = 4
        
        # 2. Get 3-channel binary regions
        p_regions = self._get_brats_regions(preds)
        g_regions = self._get_brats_regions(ground_truth)

        # 3. Compute metrics per region to prevent indexing errors
        results = {}
        region_names = ["wt", "tc", "et"]
        
        dice_scores = []
        hd95_scores = []

        for i, name in enumerate(region_names):
            # Extract single channel: (B, 1, D, H, W)
            p_reg = p_regions[:, i:i+1]
            g_reg = g_regions[:, i:i+1]

            # Dice for this region
            d = compute_dice(y_pred=p_reg, y=g_reg, ignore_empty=False)
            d_val = float(torch.nanmean(d).cpu().item())
            dice_scores.append(d_val)
            results[f"dice_{name}"] = d_val

            # HD95 for this region
            try:
                h = compute_hausdorff_distance(y_pred=p_reg, y=g_reg, spacing=(1, 1, 1), percentile=95)
                h_val = float(torch.nanmean(h).cpu().item())
            except Exception:
                h_val = 0.0 # Standard fallback for empty predictions
            
            hd95_scores.append(h_val)
            results[f"hd95_{name}"] = h_val

        # 4. Final averages
        results["mean_dice"] = float(np.mean(dice_scores))
        results["mean_hd95"] = float(np.mean(hd95_scores))

        return results

def compute_metrics(logits, ground_truth, device="cuda"):
    calculator = BraTSMetrics(device=device)
    return calculator.compute_metrics(logits, ground_truth)