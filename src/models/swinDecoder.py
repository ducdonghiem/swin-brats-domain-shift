import torch
import torch.nn as nn
from .SwinTransformers import SwinTransformerBlock


class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer - upsamples by 2x and decreases channels by 2x.
    
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
        
        x = self.norm(x)
        
        return x


class FinalPatchExpanding(nn.Module):
    """
    Final Patch Expanding Layer - performs 4x upsampling.
    Upsamples from (B, H*W, C) to (B, C, 4H, 4W)
    
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

# unused. Ignore.
class SwinDecoder(nn.Module):
    """
    Swin Transformer Decoder with 3 stages and patch expanding.
    
    Architecture (symmetric to encoder):
        Input: 7x7, C=768 (from bottleneck)
        Stage 1: 7->14, C=768->384, num_heads=12 (x2 Swin blocks)
        Stage 2: 14->28, C=384->192, num_heads=6 (x2 Swin blocks)
        Stage 3: 28->56, C=192->96, num_heads=3 (x2 Swin blocks)
        Final Expanding: 56->224, C=96 (4x upsampling)
        Output: (B, 96, 224, 224)
    
    Args:
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
    """
    
    def __init__(self, window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.num_stages = 3
        self.window_size = window_size
        
        # Stochastic depth - linearly decreasing drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)][::-1]  # Reverse for decoder
        
        # Stage 1: 7x7x768 -> 14x14x384
        self.patch_expanding1 = PatchExpanding(dim=768)
        # Concatenation with skip connection: 384 + 384 = 768
        self.concat_linear1 = nn.Linear(768, 384)
        self.stage1 = SwinDecoderStage(
            dim=384,
            num_heads=12,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[0]
        )
        
        # Stage 2: 14x14x384 -> 28x28x192
        self.patch_expanding2 = PatchExpanding(dim=384)
        # Concatenation with skip connection: 192 + 192 = 384
        self.concat_linear2 = nn.Linear(384, 192)
        self.stage2 = SwinDecoderStage(
            dim=192,
            num_heads=6,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[2]
        )
        
        # Stage 3: 28x28x192 -> 56x56x96
        self.patch_expanding3 = PatchExpanding(dim=192)
        # Concatenation with skip connection: 96 + 96 = 192
        self.concat_linear3 = nn.Linear(192, 96)
        self.stage3 = SwinDecoderStage(
            dim=96,
            num_heads=3,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[4]
        )
        
        # Final Patch Expanding: 56x56x96 -> (B, 96, 224, 224) (4x upsampling)
        self.final_expanding = FinalPatchExpanding(dim=96)
    
    def forward(self, x, skip_connections, skip_dims, H, W):
        """
        Args:
            x: Input feature from bottleneck, tensor size (B, 7*7, 768)
            skip_connections: List of skip connection features from encoder
                [stage1 (B, 56*56, 96), stage2 (B, 28*28, 192), stage3 (B, 14*14, 384)]
            skip_dims: List of (H, W) for each skip connection
                [(56, 56), (28, 28), (14, 14)]
            H, W: Spatial resolution of bottleneck input (7, 7)
        
        Returns:
            x: Output feature, tensor size (B, 96, 224, 224)
        """
        # Reverse skip connections to match decoder order
        # Encoder: stage1(56x56x96), stage2(28x28x192), stage3(14x14x384)
        # Decoder needs: stage3(14x14x384), stage2(28x28x192), stage1(56x56x96)
        skip_connections = skip_connections[::-1]  # Reverse
        skip_dims = skip_dims[::-1]  # Reverse
        
        # Stage 1: 7x7x768 -> 14x14x384
        x = self.patch_expanding1(x, H, W)  # (B, 14*14, 384)
        H, W = H * 2, W * 2  # Now 14x14
        
        # Skip connection from encoder stage 3 (14x14x384)
        skip_h, skip_w = skip_dims[0]
        assert (H, W) == (skip_h, skip_w), f"Resolution mismatch: decoder {(H, W)} vs skip {(skip_h, skip_w)}"
        x = torch.cat([x, skip_connections[0]], dim=-1)  # (B, 14*14, 768)
        x = self.concat_linear1(x)  # (B, 14*14, 384)
        x = self.stage1(x, H, W)
        
        # Stage 2: 14x14x384 -> 28x28x192
        x = self.patch_expanding2(x, H, W)  # (B, 28*28, 192)
        H, W = H * 2, W * 2  # Now 28x28
        
        # Skip connection from encoder stage 2 (28x28x192)
        skip_h, skip_w = skip_dims[1]
        assert (H, W) == (skip_h, skip_w), f"Resolution mismatch: decoder {(H, W)} vs skip {(skip_h, skip_w)}"
        x = torch.cat([x, skip_connections[1]], dim=-1)  # (B, 28*28, 384)
        x = self.concat_linear2(x)  # (B, 28*28, 192)
        x = self.stage2(x, H, W)
        
        # Stage 3: 28x28x192 -> 56x56x96
        x = self.patch_expanding3(x, H, W)  # (B, 56*56, 96)
        H, W = H * 2, W * 2  # Now 56x56
        
        # Skip connection from encoder stage 1 (56x56x96)
        skip_h, skip_w = skip_dims[2]
        assert (H, W) == (skip_h, skip_w), f"Resolution mismatch: decoder {(H, W)} vs skip {(skip_h, skip_w)}"
        x = torch.cat([x, skip_connections[2]], dim=-1)  # (B, 56*56, 192)
        x = self.concat_linear3(x)  # (B, 56*56, 96)
        x = self.stage3(x, H, W)
        
        # Final Patch Expanding: 56x56x96 -> (B, 96, 224, 224)
        x = self.final_expanding(x, H, W)  # (B, 96, 224, 224)
        
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