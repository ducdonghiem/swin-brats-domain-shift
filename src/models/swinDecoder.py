import torch
import torch.nn as nn
from .SwinTransformers import SwinTransformerBlock

class PatchExpanding(nn.Module):
    """
    Patch Expanding layer adaptation. Upsamples by 2x and decreases channels by 2x
    using a linear projection.
    
    Args:
        dim (int): Number of input channels
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
    
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        
        Returns:
            x: Output feature, tensor size (B, 2H*2W, C/2).
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = self.expand(x)  # (B, H*W, 2*C)
        
        x = x.view(B, H, W, 2 * C)
        
        # Rearrange to increase spatial resolution
        # Split channels and rearrange into 2x2 spatial pattern
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H, 2, W, 2, C/2)
        x = x.view(B, H * 2, W * 2, C // 2)  # (B, 2H, 2W, C/2)
        x = x.view(B, -1, C // 2)  # (B, 2H*2W, C/2)
        
        # norm after linear projection (normalize the 4 newly created pixels (each size C/2))
        x = self.norm(x)
        
        return x

class FinalPatchExpanding(nn.Module):
    """
    Final Patch Expanding Layer. Performs 4x upsampling using linear projection 
    to expand to 16*C channels, then rearrange and obtain (B, C, 224, 224) from (B, 56*56, C).
    In other words, it transforms (B, H*W, C) to (B, C, 4H, 4W).
    
    Args:
        dim (int): Number of input channels (96)
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Expand to 16*C (for 4x4 upsampling)
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)
    
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C) where H=56, W=56, C=96
            H, W: Spatial resolution of the input feature.
        
        Returns:
            x: Output feature, tensor size (B, C, 4H, 4W) = (B, 96, 224, 224)
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = self.expand(x)  # (B, H*W, 16*C)
        
        x = x.view(B, H, W, 16 * C)
        
        # Rearrange to increase spatial resolution by 4x
        # Split into 4x4 spatial pattern
        x = x.view(B, H, W, 4, 4, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H, 4, W, 4, C)
        x = x.view(B, H * 4, W * 4, C)  # (B, 4H, 4W, C)
        x = x.view(B, -1, C)  # (B, 4H*4W, C)
        
        x = self.norm(x)
        
        # Reshape to (B, C, H, W) format
        x = x.view(B, H * 4, W * 4, C)  # (B, 4H, 4W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, 4H, 4W)
        
        return x


class SwinDecoderStage(nn.Module):
    """
    A single Swin Decoder stage with 2 successive Swin Transformer blocks.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        
        # Two successive blocks: W-MSA then SW-MSA
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,  # W-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path
            ),
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2,  # SW-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path
            )
        ])
    
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        
        Returns:
            x: Output feature, tensor size (B, H*W, C).
        """
        for block in self.blocks:
            x = block(x, H, W)
        return x


if __name__ == "__main__":
    # Test the decoder
    batch_size = 2
    
    # Simulate bottleneck output
    bottleneck_output = torch.randn(batch_size, 7 * 7, 768)
    
    # Simulate skip connections from encoder
    skip_connections = [
        torch.randn(batch_size, 56 * 56, 96),   # Stage 1
        torch.randn(batch_size, 28 * 28, 192),  # Stage 2
        torch.randn(batch_size, 14 * 14, 384),  # Stage 3
    ]
    skip_dims = [(56, 56), (28, 28), (14, 14)]
    
    decoder = SwinDecoder()
    
    print("=" * 60)
    print("Testing Swin Decoder")
    print("=" * 60)
    print(f"Bottleneck input shape: {bottleneck_output.shape} (7x7x768)")
    print()
    print("Skip connections:")
    for i, (skip, (h, w)) in enumerate(zip(skip_connections, skip_dims)):
        print(f"  Stage {i+1}: {skip.shape} (resolution: {h}x{w})")
    print()
    
    output = decoder(bottleneck_output, skip_connections, skip_dims, 7, 7)
    
    print(f"Decoder output shape: {output.shape}")
    print(f"Expected shape: (B, 96, 224, 224)")
    assert output.shape == (batch_size, 96, 224, 224), f"Wrong output shape: {output.shape}"
    print("✓ Output shape is correct!")
    print(f"Output statistics - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print()
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)