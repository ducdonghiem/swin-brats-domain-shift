import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    """
    Skip connection module for SwinBraTS encoder-decoder fusion.
    
    Concatenates encoder and decoder features, then applies:
    - Conv2d (1x1) to reduce channels
    - LayerNorm
    - GELU activation
    
    Note: Input features are in (B, C, H, W) format.
    
    Args:
        in_channels_encoder (int): Number of channels from encoder
        in_channels_decoder (int): Number of channels from decoder
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels_encoder: int, in_channels_decoder: int, out_channels: int):
        super(SkipConnection, self).__init__()
        
        total_in = in_channels_encoder + in_channels_decoder
        
        # 1x1 conv to reduce concatenated channels -> out_channels
        self.conv1 = nn.Conv2d(total_in, out_channels, kernel_size=1, padding=0, bias=True)
        
        # LayerNorm over channels
        self.ln = nn.LayerNorm(out_channels)
        
        self.act = nn.GELU()
    
    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc: Encoder features (B, C_enc, H, W)
            dec: Decoder features (B, C_dec, H, W)
        
        Returns:
            out: Fused features (B, out_channels, H, W)
        """
        if enc.shape[0] != dec.shape[0]:
            raise ValueError("Batch size mismatch between encoder and decoder inputs")
        if enc.shape[2:] != dec.shape[2:]:
            raise ValueError(f"Spatial dimensions must match: enc {enc.shape[2:]} vs dec {dec.shape[2:]}")
        
        # Concatenate along channel dimension
        x = torch.cat([enc, dec], dim=1)  # (B, C_enc + C_dec, H, W)
        
        # 1x1 convolution
        x = self.conv1(x)  # (B, out_channels, H, W)
        
        # LayerNorm over channels per spatial location
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x)
        x = self.act(x)
        
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x


if __name__ == "__main__":
    # Test the skip connection
    batch_size = 2
    H, W = 56, 56
    
    # Simulate encoder and decoder features
    enc_features = torch.randn(batch_size, 96, H, W)
    dec_features = torch.randn(batch_size, 96, H, W)
    
    skip = SkipConnection(in_channels_encoder=96, in_channels_decoder=96, out_channels=96)
    
    output = skip(enc_features, dec_features)
    
    print("=" * 60)
    print("Testing Skip Connection")
    print("=" * 60)
    print(f"Encoder features: {enc_features.shape}")
    print(f"Decoder features: {dec_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output stats - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
    print("=" * 60)