import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA) module. Implementation sourced from original author.
    
    Args:
        dim (int): Number of input channels
        window_size (tuple[int]): The height and width of the window
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5   # scaling factor for attention scores (/sqrt(d_k))
        
        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # uses broadcasting to find the distance between every pair. The values range from -6 to +6 for 7*7 window.
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        
        # relative_position_index is a `window_size`*`window_size` matrix where each entry is an ID between 0 and the number of relative positions.
        # For a 7x7 window, the entry at (0, 48) corresponds to the relative position of the top-left token and the bottom-right token in the window, which is (-6, -6) and gets mapped to an ID of 0. 
        # The entry at (48, 0) corresponds to the relative position of the bottom-right token and the top-left token, which is (6, 6) and gets mapped to an ID of 168. 
        # The entry at (24, 24) corresponds to the relative position of the center token with itself, which is (0, 0) and gets mapped to an ID of 84.
        # This ID represents the relative position of the two tokens in the window, and is used to look up the relative position bias from the table.
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1  # Shift -6...6 to 0...12
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1     # Multiply Y by 13
        # # Add X and Y together
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        # register as buffer, not a parameter. This way it gets saved in the state dict but is not updated by the optimizer.
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Linear layers for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias table
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) where N = window_size[0] * window_size[1] (=7*7)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale query
        q = q * self.scale
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) # shape: (Batch*num_windows, num_heads, 49, 49). A 49x49 "score" for how much every pixel in the window cares about every other pixel.
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)   # flatten to 49*49
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # adds the learned spatial "bonus" to the raw attention scores.
        # Logic: If the model learned that "pixels 1 step apart" are important, the bias table will have a high value for that ID, boosting the attn score for all pixel pairs that are 1 step apart.
        
        # Apply mask if provided (for shifted window attention)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # adds -100 to the scores of "fake" neighbors.
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn) # Now, pixels only attend to "true" neighbors within the shifted window.
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)    # concatenate attention output from all heads. Shape: (Batch*num_windows, 49, C)
        x = self.proj(x)    # fusion of attention output from all heads. Shape: (Batch*num_windows, 49, C)
        # notes: if C=96, each head produces a vector of size 32, and the proj layer learns how to combine the 3 heads' outputs into a single 96-dim output.

        x = self.proj_drop(x)
        
        return x