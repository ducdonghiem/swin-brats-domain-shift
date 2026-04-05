import torch
import torch.nn as nn
from .mlp import MLP
from .window_attention import WindowAttention
from .window_utils import window_partition, window_reverse


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block. Implementation sourced from original author.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        window_size (int): Window size
        shift_size (int): Shift size for SW-MSA. Use 0 for W-MSA
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: `nn.GELU`
        norm_layer (nn.Module, optional): Normalization layer. Default: `nn.LayerNorm`
    """
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Cache for attention mask - recomputed only when spatial dims change
        self._attn_mask_cache = {}  # key: (H, W) -> mask tensor
    
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size (if input image is 224*224 (56->28->14->7), then there is no padding)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            # shift the feature map to the left and up by shift_size, so that the windows will be shifted by shift_size.
            # This allows the model to capture interactions between neighboring windows in the next attention layer.
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Use cached mask; create and cache if not seen this (Hp, Wp) before
            cache_key = (Hp, Wp)
            # will be cached for every forward pass except the first forward pass with a new (Hp, Wp) size, which only happens when the input spatial dimensions change.
            if cache_key not in self._attn_mask_cache:
                self._attn_mask_cache[cache_key] = self.create_mask(Hp, Wp).to(x.device)
            attn_mask = self._attn_mask_cache[cache_key]
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        
        # FFN with residual connection
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def create_mask(self, H, W):
        """
        Create attention mask for SW-MSA. It prevents attention between 
        tokens that are not neighbours in the original feature map, 
        which would happen after the cyclic shift. This is done by assigning 
        a large negative value to the attention scores of non-neighbouring tokens, 
        so that after softmax they become zero and do not contribute to the output.
        
        Args:
            H (int): Height of padded feature map
            W (int): Width of padded feature map
        
        Returns:
            attn_mask: Attention mask
        """
        # Calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        # slice() is like range() but not a list, it creates a slice object that can be used to index tensors. 

        # Here we create 3 slices for height and width: one for the first part of the feature map, 
        # one for the middle part (after the shift), and one for the last part (after the shift).
        
        # The cnt variable is used to assign a unique value to each region of the feature map, 
        # so that tokens in the same region will have the same value and tokens in different regions will have different values.
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # unsqueeze adds dimensions for broadcasting: nW, 1, window_size*window_size and nW, window_size*window_size, 1. 
        # subtracting them means comparing every token in the window to every other token: if they are in the same region (same value) then 0, otherwise non-zero.
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask

# drop the entire f(x) in the residual x = x + f(x) with probability drop_prob, during training. Prevents deep networks from overfitting.
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output