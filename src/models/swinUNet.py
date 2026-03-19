import torch
import torch.nn as nn
try:
    from swinEncoder import PatchPartition, PatchMerging, SwinEncoderStage
    from bottleneck import Bottleneck
    from swinDecoder import PatchExpanding, FinalPatchExpanding, SwinDecoderStage
    from skipConnection import SkipConnection
except ImportError:
    from .swinEncoder import PatchPartition, PatchMerging, SwinEncoderStage
    from .bottleneck import Bottleneck
    from .swinDecoder import PatchExpanding, FinalPatchExpanding, SwinDecoderStage
    from .skipConnection import SkipConnection


class SwinUNet(nn.Module):
    """
    Complete Swin-UNet architecture with proper skip connections.
    
    Architecture:
        Input: (B, 3, 224, 224)
        
        Encoder:
            Patch Partition -> (B, 56*56, 48) -> Linear -> (B, 56*56, 96)
            Stage 1: 56x56x96, heads=3 -> skip1
            Stage 2: 28x28x192, heads=6 -> skip2
            Stage 3: 14x14x384, heads=12 -> skip3
        
        Bottleneck: 7x7x768, heads=24
        
        Decoder (with skip connections):
            Stage 1: 7x7x768 -> Expand -> 14x14x384 + skip3 -> 14x14x384
            Stage 2: 14x14x384 -> Expand -> 28x28x192 + skip2 -> 28x28x192
            Stage 3: 28x28x192 -> Expand -> 56x56x96 + skip1 -> 56x56x96
            Final: 56x56x96 -> Expand 4x -> (B, 96, 224, 224)
    
    Args:
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of output classes for segmentation. Default: 4
        embed_dim (int): Initial embedding dimension. Default: 48
        window_size (int): Window size for Swin blocks. Default: 7
        patch_size (int): Patch size for initial partition. Default: 4
    """
    
    def __init__(self, in_channels=3, num_classes=4, embed_dim=48, window_size=7, patch_size=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size
        
        # ============ ENCODER ============
        # Patch Partition: (B, 3, 224, 224) -> (B, 56*56, 48)
        self.patch_partition = PatchPartition(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # Linear projection: 48 -> 96
        self.linear_proj = nn.Linear(embed_dim, 96)
        
        # Encoder Stage 1: 56x56x96, heads=3
        self.encoder_stage1 = SwinEncoderStage(dim=96, num_heads=3, window_size=window_size)
        self.patch_merging1 = PatchMerging(dim=96)
        
        # Encoder Stage 2: 28x28x192, heads=6
        self.encoder_stage2 = SwinEncoderStage(dim=192, num_heads=6, window_size=window_size)
        self.patch_merging2 = PatchMerging(dim=192)
        
        # Encoder Stage 3: 14x14x384, heads=12
        self.encoder_stage3 = SwinEncoderStage(dim=384, num_heads=12, window_size=window_size)
        self.patch_merging3 = PatchMerging(dim=384)
        
        # ============ BOTTLENECK ============
        # 7x7x768, heads=24
        self.bottleneck = Bottleneck(dim=768, num_heads=24, window_size=window_size)
        
        # ============ DECODER ============
        # Decoder Stage 1: 7x7x768 -> 14x14x384
        self.patch_expanding1 = PatchExpanding(dim=768)
        self.skip_connection1 = SkipConnection(
            in_channels_encoder=384,
            in_channels_decoder=384,
            out_channels=384
        )
        self.decoder_stage1 = SwinDecoderStage(dim=384, num_heads=12, window_size=window_size)
        
        # Decoder Stage 2: 14x14x384 -> 28x28x192
        self.patch_expanding2 = PatchExpanding(dim=384)
        self.skip_connection2 = SkipConnection(
            in_channels_encoder=192,
            in_channels_decoder=192,
            out_channels=192
        )
        self.decoder_stage2 = SwinDecoderStage(dim=192, num_heads=6, window_size=window_size)
        
        # Decoder Stage 3: 28x28x192 -> 56x56x96
        self.patch_expanding3 = PatchExpanding(dim=192)
        self.skip_connection3 = SkipConnection(
            in_channels_encoder=96,
            in_channels_decoder=96,
            out_channels=96
        )
        self.decoder_stage3 = SwinDecoderStage(dim=96, num_heads=3, window_size=window_size)
        
        # Final Patch Expanding: 56x56x96 -> 224x224x96
        self.final_expanding = FinalPatchExpanding(dim=96)
    
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, 224, 224)
        
        Returns:
            out: Output features (B, 96, 224, 224)
        """
        # ============ ENCODER ============
        # Patch Partition: (B, 3, 224, 224) -> (B, 56*56, 48)
        x, H, W = self.patch_partition(x)  # H=56, W=56
        
        # Linear projection: 48 -> 96
        x = self.linear_proj(x)  # (B, 56*56, 96)
        
        # Encoder Stage 1: 56x56x96
        enc1 = self.encoder_stage1(x, H, W)  # (B, 56*56, 96)
        x = self.patch_merging1(enc1, H, W)
        H, W = H // 2, W // 2  # Now 28x28
        
        # Encoder Stage 2: 28x28x192
        enc2 = self.encoder_stage2(x, H, W)  # (B, 28*28, 192)
        x = self.patch_merging2(enc2, H, W)
        H, W = H // 2, W // 2  # Now 14x14
        
        # Encoder Stage 3: 14x14x384
        enc3 = self.encoder_stage3(x, H, W)  # (B, 14*14, 384)
        x = self.patch_merging3(enc3, H, W)
        H, W = H // 2, W // 2  # Now 7x7
        
        # ============ BOTTLENECK ============
        # 7x7x768
        x = self.bottleneck(x, H, W)  # (B, 7*7, 768)
        
        # ============ DECODER ============
        # Correct order per Swin-UNet paper:
        #   1. Patch expanding (upsample)
        #   2. Fuse skip connection (so transformer sees encoder context)
        #   3. Swin Transformer blocks

        # Decoder Stage 1: 7x7x768 -> 14x14x384
        x = self.patch_expanding1(x, H, W)  # (B, 14*14, 384)
        H, W = H * 2, W * 2  # Now 14x14

        # Skip connection BEFORE transformer blocks
        x = x.view(-1, H, W, 384).permute(0, 3, 1, 2)          # (B, 384, 14, 14)
        enc3_img = enc3.view(-1, H, W, 384).permute(0, 3, 1, 2) # (B, 384, 14, 14)
        x = self.skip_connection1(enc3_img, x)                   # (B, 384, 14, 14)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, H * W, 384)  # (B, 14*14, 384)

        # Transformer blocks now attend to fused encoder+decoder features
        x = self.decoder_stage1(x, H, W)  # (B, 14*14, 384)

        # Decoder Stage 2: 14x14x384 -> 28x28x192
        x = self.patch_expanding2(x, H, W)  # (B, 28*28, 192)
        H, W = H * 2, W * 2  # Now 28x28

        # Skip connection BEFORE transformer blocks
        x = x.view(-1, H, W, 192).permute(0, 3, 1, 2)          # (B, 192, 28, 28)
        enc2_img = enc2.view(-1, H, W, 192).permute(0, 3, 1, 2) # (B, 192, 28, 28)
        x = self.skip_connection2(enc2_img, x)                   # (B, 192, 28, 28)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, H * W, 192)  # (B, 28*28, 192)

        # Transformer blocks now attend to fused encoder+decoder features
        x = self.decoder_stage2(x, H, W)  # (B, 28*28, 192)

        # Decoder Stage 3: 28x28x192 -> 56x56x96
        x = self.patch_expanding3(x, H, W)  # (B, 56*56, 96)
        H, W = H * 2, W * 2  # Now 56x56

        # Skip connection BEFORE transformer blocks
        x = x.view(-1, H, W, 96).permute(0, 3, 1, 2)           # (B, 96, 56, 56)
        enc1_img = enc1.view(-1, H, W, 96).permute(0, 3, 1, 2)  # (B, 96, 56, 56)
        x = self.skip_connection3(enc1_img, x)                   # (B, 96, 56, 56)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, H * W, 96)  # (B, 56*56, 96)

        # Transformer blocks now attend to fused encoder+decoder features
        x = self.decoder_stage3(x, H, W)  # (B, 56*56, 96)
        
        # Final Patch Expanding: 56x56x96 -> 224x224x96
        x = self.final_expanding(x, H, W)  # (B, 96, 224, 224)
        
        return x


if __name__ == "__main__":
    # Test the full model
    batch_size = 2
    input_image = torch.randn(batch_size, 3, 224, 224)
    
    model = SwinUNet(in_channels=3, num_classes=4)
    
    print("=" * 70)
    print("Testing Complete Swin-UNet with Skip Connections")
    print("=" * 70)
    print(f"Input shape: {input_image.shape}")
    print()
    
    output = model(input_image)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (B, 96, 224, 224)")
    assert output.shape == (batch_size, 96, 224, 224), f"Wrong output shape: {output.shape}"
    print("✓ Output shape is correct!")
    print(f"Output stats - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    print("=" * 70)
    print("Architecture verified! ✓")
    print("=" * 70)
    print()
    print("Skip connections implemented at:")
    print("  - 1/4 scale: 14x14x384 (encoder stage 3 ↔ decoder stage 1)")
    print("  - 1/2 scale: 28x28x192 (encoder stage 2 ↔ decoder stage 2)")
    print("  - 1/1 scale: 56x56x96  (encoder stage 1 ↔ decoder stage 3)")
    print("=" * 70)