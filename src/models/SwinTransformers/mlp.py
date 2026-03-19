import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in Swin Transformer.
    
    Args:
        in_features (int): Number of input channels
        hidden_features (int, optional): Number of hidden channels. Default: None (same as in_features)
        out_features (int, optional): Number of output channels. Default: None (same as in_features)
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        drop (float, optional): Dropout rate. Default: 0.0
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x