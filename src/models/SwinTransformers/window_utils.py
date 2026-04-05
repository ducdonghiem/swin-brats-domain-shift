'''
Helper functions for shifted window attention.

Implementation sourced from original author.
'''

def window_partition(x, window_size):
    """
    Partition input into non-overlapping windows. Reshapes `x` and ensures 
    a `window_size` * `window_size` local "tokens" in a window attention. 
    
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse of `window_partition`. Reshapes `windows` back to original feature map. 
    Implementation sourced from original author.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x