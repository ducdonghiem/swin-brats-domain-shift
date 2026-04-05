import torch
import torch.nn.functional as F
import math


class GPUAugmentation:
    """
    GPU-based augmentation using PyTorch: flip, affine rotation, MRI bias field.

    All ops run on already-loaded GPU tensors using torch.nn.functional.grid_sample,
    which natively supports batched (B, C, D, H, W) input.

    Spatial transforms (flip, affine rotation) are applied jointly to all modalities
    and the mask in a single grid_sample call to ensure identical transformations.

    Bias field is implemented natively in PyTorch via a random low-order polynomial
    evaluated on a spatial grid, applied to modalities only (not the mask).

    Args:
        flip_prob (float): Probability of applying flip. Probability is determined per axis. Default 0.5
        affine_prob (float): Probability of applying rotation. Default 0.2
        affine_degrees (float): Max rotation angle in degrees. Default 15
        bias_field_prob (float): Probability of bias field per modality. Default 0.2
        bias_field_order (int): Polynomial order for bias field. Default 3
        device (str): Device. Default 'cuda'
    """

    def __init__(
        self,
        flip_prob=0.5,
        affine_prob=0.2,
        affine_degrees=15,
        bias_field_prob=0.2,
        bias_field_order=3,
        device='cuda',
    ):
        self.flip_prob = flip_prob
        self.affine_prob = affine_prob
        self.affine_degrees_rad = affine_degrees * math.pi / 180
        self.bias_field_prob = bias_field_prob
        self.bias_field_order = bias_field_order
        self.device = device

    @torch.no_grad()
    def __call__(self, modalities, labels):
        """
        Args:
            modalities: list of 4 tensors, each (B, D, H, W) float32
            labels:     (B, D, H, W) int64

        Returns:
            modalities: augmented list of 4 tensors, each (B, D, H, W) float32
            labels:     augmented (B, D, H, W) int64
        """
        B, D, H, W = modalities[0].shape

        # Stack all modalities + mask into one tensor for joint spatial transforms.
        # Shape: (B, 5, D, H, W)
        mask_float = labels.float().unsqueeze(1)       # (B, 1, D, H, W)
        vol = torch.stack(modalities, dim=1)           # (B, 4, D, H, W)
        vol = torch.cat([vol, mask_float], dim=1)      # (B, 5, D, H, W)

        # Flips
        # Independent per-sample flip along H (axis 3) and D (axis 2)
        for spatial_axis in [2, 3]:
            flip_mask = torch.rand(B, device=self.device) < self.flip_prob
            for b in range(B):
                if flip_mask[b]:
                    vol[b] = vol[b].flip(spatial_axis - 1)  # per-item axes: (C,D,H,W)

        # Affine rotation (in-plane HxW only)
        # Rotate in the HxW plane independently per sample.
        # D (depth) is not rotated - it represents axial slices and our model
        # treats it as a channel dimension, so only in-plane coherence matters.
        affine_mask = torch.rand(B, device=self.device) < self.affine_prob
        if affine_mask.any():
            angles = (torch.rand(B, device=self.device) * 2 - 1) * self.affine_degrees_rad
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            zeros = torch.zeros(B, device=self.device)

            # (B, 2, 3) affine matrix for 2D rotation
            theta = torch.stack([
                torch.stack([cos_a, -sin_a, zeros], dim=1),
                torch.stack([sin_a,  cos_a, zeros], dim=1),
            ], dim=1)

            # Merge C and D into one channel dim so grid_sample sees (B, C*D, H, W)
            vol_2d = vol.view(B, 5 * D, H, W)
            grid = F.affine_grid(theta, vol_2d.shape, align_corners=False)

            rotated = F.grid_sample(
                vol_2d.float(),
                grid,
                mode='nearest',       # nearest is safe for both float and int labels
                padding_mode='zeros',
                align_corners=False,
            )  # (B, 5*D, H, W)

            for b in range(B):
                if affine_mask[b]:
                    vol[b] = rotated[b].view(5, D, H, W)

        # Unstack
        mod_vols  = [vol[:, i] for i in range(4)]   # list of (B, D, H, W)
        mask_vol  = vol[:, 4].round().long()          # (B, D, H, W) int64

        # Bias field
        aug_modalities = []
        for mod in mod_vols:
            if torch.rand(1).item() < self.bias_field_prob:
                mod = self._apply_bias_field(mod)
            aug_modalities.append(mod)

        return aug_modalities, mask_vol

    def _apply_bias_field(self, vol):
        """
        Smooth multiplicative bias field via random polynomial over HxW grid.
        """
        B, D, H, W = vol.shape
        order = self.bias_field_order

        yy = torch.linspace(-1, 1, H, device=self.device)
        xx = torch.linspace(-1, 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')  # (H, W)

        # Polynomial basis: all terms x^i * y^j where i+j <= order
        basis = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                basis.append((grid_x ** i) * (grid_y ** j))
        basis = torch.stack(basis, dim=0)  # (num_terms, H, W)

        # Small random coefficients -> gentle field
        num_terms = basis.shape[0]
        coeffs = torch.randn(B, num_terms, device=self.device) * 0.1

        # Field shape: (B, H, W)
        field = torch.einsum('bt,thw->bhw', coeffs, basis).exp()

        # Apply multiplicatively across all D slices
        return vol * field.unsqueeze(1)  # (B, 1, H, W) broadcasts over D