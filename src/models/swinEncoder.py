import torch
import torch.nn as nn
from .SwinTransformers import SwinTransformerBlock

# use convolution2D with kernel_size=patch_size and stride=patch_size to implement patch partition. 
# This is more efficient than manually slicing the image into patches and applying a linear layer to each patch. 
# The convolution will automatically extract non-overlapping patches and project them to the embedding dimension in one step.
class PatchPartition(nn.Module):
    """
    Patch Partition Layer - converts image to patches and projects to embedding dimension.
    Input: (B, 3, 224, 224)
    Output: (B, 56*56, 48) -> after first Swin block: (B, 56*56, 96)
    
    Args:
        in_channels (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 48
        patch_size (int): Patch token size. Default: 4
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, in_channels=3, embed_dim=48, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Linear projection: 4x4 patches with 3 channels -> embed_dim
        # This is equivalent to Conv2d with kernel_size=patch_size, stride=patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input image, tensor size (B, 3, 224, 224)
        
        Returns:
            x: Patch embeddings, tensor size (B, 56*56, 48)
            H, W: Output spatial dimensions (56, 56)
        """
        B, C, H, W = x.shape
        
        # Apply patch projection
        x = self.proj(x)  # (B, 48, 56, 56)
        B, C, H, W = x.shape
        
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)  # (B, 56*56, 48)
        x = self.norm(x)
        
        return x, H, W

# use linear projection to reduce the number of channels by half.
class PatchMerging(nn.Module):
    """
    Patch Merging Layer - downsamples by 2x and increases channels by 2x.
    
    Args:
        dim (int): Number of input channels
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        
        Returns:
            x: Output feature, tensor size (B, H/2*W/2, 2*C).
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)
        
        # Partition into 2x2 patches and concatenate
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C # top-left
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C # top-right
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C # bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C # local concatenation of 4 patches
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        # norm first then linear projection.
        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C
        
        return x


class SwinEncoderStage(nn.Module):
    """
    A single Swin Encoder stage with 2 successive Swin Transformer blocks.
    
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
    # Test the encoder
    batch_size = 2
    
    x = torch.randn(batch_size, 3, 224, 224)
    
    encoder = SwinEncoder(in_channels=3)
    
    print("=" * 60)
    print("Testing Swin Encoder")
    print("=" * 60)
    print(f"Input shape: {x.shape}")
    print()
    
    skip_connections, bottleneck_input, skip_dims, bottleneck_dim = encoder(x)
    
    print("Skip Connections:")
    for i, (skip, (h, w)) in enumerate(zip(skip_connections, skip_dims)):
        print(f"  Stage {i+1}: {skip.shape} (resolution: {h}x{w})")
    
    print()
    print(f"Bottleneck Input: {bottleneck_input.shape} (resolution: {bottleneck_dim[0]}x{bottleneck_dim[1]})")
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print()
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)