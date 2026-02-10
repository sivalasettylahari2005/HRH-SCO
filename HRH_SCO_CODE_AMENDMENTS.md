# HRH-SCO Technical Documentation: Code Amendments

This document outlines the specific methods and architectural changes implemented to upgrade the YOLC model to the **HRH-SCO (High-Resolution Heatmap with Subpixel Center Offsets)** version.

## 1. High-Resolution Heatmap Construction
The following method in `models/dense_heads/yolc_head.py` was amended to support dynamic upsampling of the detection heatmap.

```python
def _build_loc_head(self, in_channel, out_channel):
    """Build head for high resolution heatmap branch with HRH architecture."""
    if not getattr(self, 'use_hrh_sco', False):
        # Baseline Architecture (Standard YOLC)
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel, self.num_classes * 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.num_classes * 8, self.num_classes * 8, 4, stride=2, padding=1),
            nn.Conv2d(self.num_classes * 8, out_channel, 1, groups=self.num_classes)
        )
    else:
        # Proposed HRH-SCO Architecture: Dynamic Multi-stage Upsampling
        layers = []
        layers.append(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        curr_channel = in_channel
        for i in range(self.heatmap_upsample_num):
            layers.append(nn.ConvTranspose2d(curr_channel, self.num_classes * 8, 4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            curr_channel = self.num_classes * 8
        
        layers.append(nn.Conv2d(curr_channel, out_channel, 1, groups=self.num_classes))
        return nn.Sequential(*layers)
```

## 2. Subpixel Center Offset (SCO) Branch
A new method was added to create the head responsible for predicting the fractional displacement of object centers.

```python
def _build_offset_head(self, in_channel, out_channel):
    """New Method: Build head for subpixel center offsets (SCO)."""
    layers = []
    layers.append(nn.Conv2d(in_channel, in_channel, 3, padding=1))
    layers.append(nn.ReLU(inplace=True))
    
    curr_channel = in_channel
    for i in range(self.heatmap_upsample_num):
         # Matches the resolution of the heatmap for pixel-perfect alignment
         layers.append(nn.ConvTranspose2d(curr_channel, curr_channel // 2, 4, stride=2, padding=1))
         layers.append(nn.ReLU(inplace=True))
         curr_channel = curr_channel // 2

    layers.append(nn.Conv2d(curr_channel, out_channel, 1))
    return nn.Sequential(*layers)
```

## 3. SCO Loss Integration
The training logic was amended to include the L1 loss for the subpixel refinement branch.

```python
if self.use_hrh_sco:
    center_offset_target = target_result['center_offset_target']
    center_offset_target_weight = target_result['center_offset_target_weight']
    
    # L1 Loss between predicted fractional offsets and ground truth
    loss_center_offset = self.loss_center_offset(
        center_offset_pred, center_offset_target, center_offset_target_weight, avg_factor=avg_factor)
    losses['loss_center_offset'] = loss_center_offset
```

## 4. Inference Refinement
The `decode_heatmap` logic was updated to apply the learned offsets during its final coordinate calculation.

```python
# HRH-SCO: Subpixel localization
if center_offset_pred is not None:
    # Gather subpixel offsets from the high-res map at predicted peak locations
    offsets = transpose_and_gather_feat(center_offset_pred, batch_index_large)
    
    # Refine the high-resolution integer peaks with decimal offsets
    topk_xs_high_res = topk_xs_large.float() + offsets[..., 0]
    topk_ys_high_res = topk_ys_large.float() + offsets[..., 1]
    
    # Convert back to global coordinates
    topk_xs = topk_xs_high_res / upsample_ratio
    topk_ys = topk_ys_high_res / upsample_ratio
```
