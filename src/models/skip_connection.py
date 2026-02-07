import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    """
    Generic skip connection module for SwinBraTS encoder-decoder fusion.

    Behavior:
    - Accepts two inputs (encoder features, decoder features) with matching spatial
      resolution (H, W).
    - Concatenates them along the channel dimension.
    - Applies a 1x1 convolution to reduce channels to `out_channels`.
    - Applies LayerNorm over the channel dimension (performed per spatial location)
      and a GELU activation.

    Usage:
        skip = SkipConnection(in_channels_encoder=56, in_channels_decoder=56, out_channels=56)
        out = skip(encoder_feats, decoder_feats)  # (B, out_channels, H, W)
    """

    def __init__(self, in_channels_encoder: int, in_channels_decoder: int, out_channels: int):
        super(SkipConnection, self).__init__()

        total_in = in_channels_encoder + in_channels_decoder

        # 1x1 conv to reduce concatenated channels -> out_channels
        self.conv1 = nn.Conv2d(total_in, out_channels, kernel_size=1, padding=0, bias=True)

        # LayerNorm over channels: we'll apply it to the channel axis by
        # permuting to (B, H, W, C), normalizing, then permuting back.
        self.ln = nn.LayerNorm(out_channels)

        self.act = nn.GELU()

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        if enc.shape[0] != dec.shape[0]:
            raise ValueError("Batch size mismatch between encoder and decoder inputs")
        if enc.shape[2:] != dec.shape[2:]:
            raise ValueError("Spatial dimensions of encoder and decoder must match (H,W)")

        x = torch.cat([enc, dec], dim=1)
        x = self.conv1(x)  # (B, out_channels, H, W)

        # LayerNorm over channels per spatial location
        # permute to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = self.act(x)
        # permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x