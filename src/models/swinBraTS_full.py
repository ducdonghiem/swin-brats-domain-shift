import torch
import torch.nn as nn
from .projection_block import ProjectionBlock
from .swinUNet import SwinUNet
from .reconstruction_block import ReconstructionBlock


class SwinBraTS(nn.Module):
    """
    Complete end-to-end SwinBraTS model for brain tumor segmentation.
    
    Args:
        in_channels (int): Input channels per modality. Default: 155
        num_classes (int): Number of segmentation classes. Default: 4
        embed_dim (int): Initial embedding dimension for Swin. Default: 48
        window_size (int): Swin window size. Default: 7
        patch_size (int): Patch size for initial partition. Default: 4
    """
    
    def __init__(
        self,
        in_channels=155,
        num_classes=4,
        embed_dim=48,
        window_size=7,
        patch_size=4
    ):
        super(SwinBraTS, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # projection block
        # Fuses 4 MRI modalities (240×240 each) into 3-channel representation (224×224)
        self.projection_block = ProjectionBlock(
            in_channels=in_channels,
            hidden_channels=8,
            out_channels=3,
            num_modalities=4
        )
        
        # SwinUNet backbone
        # Processes fused features through encoder-decoder with skip connections
        # Output: (B, 96, 224, 224)
        self.swin_backbone = SwinUNet(
            in_channels=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size
        )
        
        # reconstruction block
        # Upsamples from Swin decoder output back to original resolution
        # Output: (B, 4, 155, 240, 240)
        self.reconstruction_block = ReconstructionBlock(
            in_channels=96,
            hidden_channels=32,
            num_classes=num_classes,
            original_depth=155
        )
    
    def forward(self, modalities):
        """
        Forward pass through the complete SwinBraTS pipeline.
        
        Args:
            modalities (list of torch.Tensor): List of 4 modality tensors, each shape (B, 155, 240, 240)
                - modalities[0]: FLAIR
                - modalities[1]: T1
                - modalities[2]: T1ce
                - modalities[3]: T2
        
        Returns:
            (B, 4, 155, 240, 240): Class-specific segmentation logits for each of the 4 classes
        """
        fused = self.projection_block(modalities)
        features = self.swin_backbone(fused)
        logits = self.reconstruction_block(features)
        
        return logits


if __name__ == "__main__":
    model = SwinBraTS(
        in_channels=155,
        num_classes=4,
        embed_dim=48,
        window_size=7,
        patch_size=4
    )
    
    # Create dummy input
    batch_size = 2
    modalities = [
        torch.randn(batch_size, 155, 240, 240),  # FLAIR
        torch.randn(batch_size, 155, 240, 240),  # T1
        torch.randn(batch_size, 155, 240, 240),  # T1ce
        torch.randn(batch_size, 155, 240, 240),  # T2
    ]
    
    print("Input shapes:")
    for i, mod in enumerate(modalities):
        print(f"  Modality {i}: {tuple(mod.shape)}")
    print()
    
    # Forward pass
    try:
        logits = model(modalities)
        print("Output shapes (class-specific logits):")
        
        # remake the test for the new output shape (B, 155, 240, 240)
        print(f"  Logits: {tuple(logits.shape)}")  # Should be (B, 4, 155, 240, 240)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Parameters:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print()
        
        # Component breakdown
        proj_params = sum(p.numel() for p in model.projection_block.parameters())
        swin_params = sum(p.numel() for p in model.swin_backbone.parameters())
        recon_params = sum(p.numel() for p in model.reconstruction_block.parameters())
        
        print(f"Component Breakdown:")
        print(f"  ProjectionBlock:     {proj_params:,}")
        print(f"  SwinUNet Backbone:   {swin_params:,}")
        print(f"  ReconstructionBlock: {recon_params:,}")
        print()
        
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
