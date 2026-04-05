import torch.nn as nn

class ReconstructionBlock(nn.Module):
    '''
    Custom reconstruction block adaptation for BraTS dataset.

    The reconstruction block takes in the final feature map from the decoder in three stages.
    Each stage upsamples the feature map with transposed convolutions, 
    followed by group normalization and GELU activation. 
    
    Finally, it projects the features to the original number of channels with a 1x1 convolution,
    and reshapes it to a 3D volume.

    Args:
        in_channels (int): Number of input channels from UNet. Default: 96.
        hidden_channels (int): Number of hidden channels. Default: 32.
        num_classes (int): Number of output classes. Default: 4.
        original_depth (int): Original depth of the input volumes. Default: 155.
    '''

    def __init__(self, in_channels=96, hidden_channels=32, num_classes=4, original_depth=155):
        super(ReconstructionBlock, self).__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # Transposed convs to upsample spatially: 224 -> 240
        self.tconv_3  = nn.ConvTranspose2d(in_channels,     hidden_channels, kernel_size=3,  stride=1, padding=0)
        self.tconv_5  = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5,  stride=1, padding=0)
        self.tconv_11 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=11, stride=1, padding=0)
        
        # GroupNorm: batch-size independent, consistent with transformer norms upstream
        self.gn_3  = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gn_5  = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gn_11 = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.gelu = nn.GELU()

        # Project to (num_classes * depth) channels, then view as 3D volume
        self.final_logits = nn.Conv2d(hidden_channels, num_classes * original_depth, kernel_size=1)

        # Add a small 3D conv to let adjacent slices communicate
        self.depth_refine = nn.Conv3d(num_classes, num_classes, 
                               kernel_size=(3, 1, 1),  # only along depth
                               padding=(1, 0, 0))

    def forward(self, x):
        """
        Args:
            x: Input features from Swin backbone, shape (B, C=96, 224, 224)
        Returns:
            (B, num_classes=4, depth=155, 240, 240): Class-specific segmentation logits for each voxel
        """
        x = self.gelu(self.gn_3(self.tconv_3(x)))    # (B, 32, 226, 226)
        x = self.gelu(self.gn_5(self.tconv_5(x)))    # (B, 32, 230, 230)
        x = self.gelu(self.gn_11(self.tconv_11(x)))  # (B, 32, 240, 240)
        x = self.final_logits(x)                      # (B, 4*155, 240, 240)
        b, _, h, w = x.shape
        x = x.view(b, self.num_classes, -1, h, w)     # (B, 4, 155, 240, 240)
        x = self.depth_refine(x)                     # (B, 4, 155, 240, 240) — adds inter-slice context
        return x