import torch.nn as nn

class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels=48, hidden_channels=32, num_classes=4, original_depth=155):
        super(ReconstructionBlock, self).__init__()
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # Upsampling layers
        self.tconv_3 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0)
        self.tconv_5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=0)
        self.tconv_11 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=11, stride=1, padding=0)
        
        self.bn_3 = nn.BatchNorm2d(hidden_channels)
        self.bn_5 = nn.BatchNorm2d(hidden_channels)
        self.bn_11 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        # The Final Transition to 3D Volume
        self.final_logits = nn.Conv2d(hidden_channels, num_classes * original_depth, kernel_size=1)

    def forward(self, x):
        # Upsample spatially
        x = self.relu(self.bn_3(self.tconv_3(x)))
        x = self.relu(self.bn_5(self.tconv_5(x)))
        x = self.relu(self.bn_11(self.tconv_11(x)))

        x = self.final_logits(x)
        
        # Reshape to 5D Tensor
        b, _, h, w = x.shape
        x = x.view(b, self.num_classes, -1, h, w)
        
        return x