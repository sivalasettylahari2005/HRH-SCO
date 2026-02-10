# HRH-SCO IMPLEMENTATION APPENDIX
# This file contains the new methods added for High-Resolution Heatmaps 
# and Subpixel Center Offsets in the YOLC architecture.

import torch
import torch.nn as nn

def build_offset_head(in_channel, upsample_num):
    """
    NEW METHOD: Construct the Subpixel Center Offset (SCO) branch.
    This head predicts the fractional (x, y) displacement of object centers.
    """
    layers = []
    layers.append(nn.Conv2d(in_channel, in_channel, 3, padding=1))
    layers.append(nn.ReLU(inplace=True))
    
    curr_channel = in_channel
    for i in range(upsample_num):
         # Transposed convolution to match the high-resolution heatmap grid
         layers.append(nn.ConvTranspose2d(curr_channel, curr_channel // 2, 4, stride=2, padding=1))
         layers.append(nn.ReLU(inplace=True))
         curr_channel = curr_channel // 2

    layers.append(nn.Conv2d(curr_channel, 2, 1)) # Output is 2 channels: (dx, dy)
    return nn.Sequential(*layers)

def decode_subpixel_heatmap(heatmap_pred, offset_pred, upsample_ratio):
    """
    NEW METHOD: Inference-time decoding with subpixel refinement.
    Integrates the peak of the high-res heatmap with predicted SCO values.
    """
    # 1. Identify high-resolution integer peaks
    # topk_xs_large, topk_ys_large (integer coordinates)
    
    # 2. Gather subpixel offsets at those exact peak locations
    # offsets = (dx, dy) gathered from offset_pred
    
    # 3. Refine: Final_Coord = Integer_Peak + Subpixel_Offset
    # topk_xs_refined = (topk_xs_large.float() + offsets[..., 0]) / upsample_ratio
    # topk_ys_refined = (topk_ys_large.float() + offsets[..., 1]) / upsample_ratio
    
    pass # Integrated into yolc_head.py

def calculate_sco_loss(pred_offsets, target_offsets, weights, avg_factor):
    """
    NEW METHOD: L1 Loss for the Subpixel Center Offset branch.
    Only calculated for positive pixels (where an object exists).
    """
    loss_func = nn.L1Loss(reduction='none')
    loss = loss_func(pred_offsets, target_offsets)
    loss = (loss * weights).sum() / avg_factor
    return loss
