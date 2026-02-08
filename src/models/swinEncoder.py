import torch
import torch.nn as nn
from SwinTransformers.swinTransformerBlock import SwinTransformerBlock


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
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
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


class SwinEncoder(nn.Module):
    """
    Swin Transformer Encoder with patch partition and 3 stages.
    
    Architecture:
        Patch Partition: (B, 3, 224, 224) -> (B, 56*56, 48)
        First Linear: (B, 56*56, 48) -> (B, 56*56, 96)
        Stage 1: 56x56, C=96,  num_heads=3  (x2 Swin blocks)
        Stage 2: 28x28, C=192, num_heads=6  (x2 Swin blocks)
        Stage 3: 14x14, C=384, num_heads=12 (x2 Swin blocks)
        Output: 7x7, C=768 (after final patch merging, input to bottleneck)
    
    Args:
        in_channels (int): Number of input image channels. Default: 3
        embed_dim (int): Initial embedding dimension after patch partition. Default: 48
        patch_size (int): Patch size for patch partition. Default: 4
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
    """
    
    def __init__(self, in_channels=3, embed_dim=48, patch_size=4, window_size=7, 
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.1):
        super().__init__()
        
        self.num_stages = 3
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Patch Partition: (B, 3, 224, 224) -> (B, 56*56, 48)
        self.patch_partition = PatchPartition(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # Linear projection to increase channels: 48 -> 96
        self.linear_proj = nn.Linear(embed_dim, 96)
        
        # Stochastic depth - linearly increasing drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]  # 6 blocks total (2 per stage)
        
        # Stage 1: 56x56, C=96, heads=3
        self.stage1 = SwinEncoderStage(
            dim=96,
            num_heads=3,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[0]
        )
        self.patch_merging1 = PatchMerging(dim=96)
        
        # Stage 2: 28x28, C=192, heads=6
        self.stage2 = SwinEncoderStage(
            dim=192,
            num_heads=6,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[2]
        )
        self.patch_merging2 = PatchMerging(dim=192)
        
        # Stage 3: 14x14, C=384, heads=12
        self.stage3 = SwinEncoderStage(
            dim=384,
            num_heads=12,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[4]
        )
        self.patch_merging3 = PatchMerging(dim=384)
    
    def forward(self, x):
        """
        Args:
            x: Input image, tensor size (B, 3, 224, 224)
        
        Returns:
            skip_connections: List of features for skip connections
                [stage1_out (B, 56*56, 96), stage2_out (B, 28*28, 192), stage3_out (B, 14*14, 384)]
            x: Output feature for bottleneck (B, 7*7, 768)
            output_dims: List of (H, W) for each skip connection
            bottleneck_dim: (H, W) for bottleneck input
        """
        skip_connections = []
        output_dims = []
        
        # Patch Partition: (B, 3, 224, 224) -> (B, 56*56, 48)
        x, H, W = self.patch_partition(x)  # (B, 3136, 48), H=56, W=56
        
        # Linear projection: 48 -> 96
        x = self.linear_proj(x)  # (B, 3136, 96)
        
        # Stage 1: 56x56x96
        x = self.stage1(x, H, W)
        skip_connections.append(x)
        output_dims.append((H, W))
        x = self.patch_merging1(x, H, W)
        H, W = H // 2, W // 2  # Now 28x28
        
        # Stage 2: 28x28x192
        x = self.stage2(x, H, W)
        skip_connections.append(x)
        output_dims.append((H, W))
        x = self.patch_merging2(x, H, W)
        H, W = H // 2, W // 2  # Now 14x14
        
        # Stage 3: 14x14x384
        x = self.stage3(x, H, W)
        skip_connections.append(x)
        output_dims.append((H, W))
        x = self.patch_merging3(x, H, W)
        H, W = H // 2, W // 2  # Now 7x7, C=768
        
        return skip_connections, x, output_dims, (H, W)


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