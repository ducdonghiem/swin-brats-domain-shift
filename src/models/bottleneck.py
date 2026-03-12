import torch
import torch.nn as nn
try:
    from SwinTransformers.swinTransformerBlock import SwinTransformerBlock
except ImportError:
    from .SwinTransformers.swinTransformerBlock import SwinTransformerBlock


class Bottleneck(nn.Module):
    """
    Bottleneck layer with 2 successive Swin Transformer blocks.
    
    Args:
        dim (int): Number of input channels. Default: 768
        num_heads (int): Number of attention heads. Default: 24
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.1
    """
    
    def __init__(self, dim=768, num_heads=24, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
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
        for block in self.blocks:
            x = block(x, H, W)
        return x


if __name__ == "__main__":
    batch_size = 2
    H, W = 7, 7
    C = 768
    
    x = torch.randn(batch_size, H * W, C)
    
    bottleneck = Bottleneck(dim=C, num_heads=24, window_size=7)
    
    print(f"Input shape: {x.shape}")
    print(f"Input resolution: {H}x{W}, channels: {C}")
    print(f"Number of heads: 24")
    print(f"Window size: 7")
    print()
    
    output = bottleneck(x, H, W)
    
    print(f"Output shape: {output.shape}")
    print(f"Output statistics - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
    
    total_params = sum(p.numel() for p in bottleneck.parameters())
    trainable_params = sum(p.numel() for p in bottleneck.parameters() if p.requires_grad)
    
    print()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")