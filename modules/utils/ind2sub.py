"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import numpy as np
import torch
"""array_shape (rows, cols)
ind = row * cols + col
"""
def ind2coords(array_shape, ind):
    row = torch.div(ind, array_shape[1], rounding_mode='trunc')
    col = ind % array_shape[1] # or numpy.mod(ind.astype('int'), array_shape[1])
    # [MODIFIED] Create coords on the same device as ind (when ind is a torch.Tensor)
    # to avoid CPU<->GPU round-trips during the forward pass.
    # When ind is a numpy array (used in test()), we fall back to CPU as before.
    device = ind.device if isinstance(ind, torch.Tensor) else torch.device('cpu')
    coords = torch.zeros((1, 1, len(ind), 2), dtype=torch.float32, device=device)
    coords[:, :, :, 1] = 2 * row.to(torch.float32) / array_shape[0] - 1
    coords[:, :, :, 0] = 2 * col.to(torch.float32) / array_shape[1] - 1
    return coords