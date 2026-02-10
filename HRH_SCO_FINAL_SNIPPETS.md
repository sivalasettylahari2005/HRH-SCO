# HRH-SCO Extension: Core Code Implementation Snippets

This document consolidates the key structural and logical amendments made to implement the **High-Resolution Heatmap with Subpixel Center Offsets (HRH-SCO)** extension.

---

### **1. Dynamic High-Resolution Heatmap Head**
*File: `models/dense_heads/yolc_head.py`*

This method replaces the static upsampling of the baseline model with a dynamic loop that scales the heatmap resolution based on the `heatmap_upsample_num` configuration.

```python
def _build_loc_head(self, in_channel, out_channel):
    """Build head for high resolution heatmap branch."""
    if not getattr(self, 'use_hrh_sco', False):
        # Baseline Architecture: Fixed x4 upsample
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
        # HRH-SCO Architecture: Dynamic Multi-stage Upsampling
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

---

### **2. Subpixel Center Offset (SCO) Branch (New Method)**
*File: `models/dense_heads/yolc_head.py`*

An entirely new architectural branch designed to predict the fractional displacement $(\Delta x, \Delta y)$ of object centers on the high-resolution grid.

```python
def _build_offset_head(self, in_channel, out_channel):
    """Build head for subpixel center offsets (SCO)."""
    layers = []
    layers.append(nn.Conv2d(in_channel, in_channel, 3, padding=1))
    layers.append(nn.ReLU(inplace=True))
    
    curr_channel = in_channel
    for i in range(self.heatmap_upsample_num):
         layers.append(nn.ConvTranspose2d(curr_channel, curr_channel // 2, 4, stride=2, padding=1))
         layers.append(nn.ReLU(inplace=True))
         curr_channel = curr_channel // 2

    layers.append(nn.Conv2d(curr_channel, out_channel, 1))
    return nn.Sequential(*layers)
```

---

### **3. Subpixel Target Generation Logic**
*File: `models/dense_heads/yolc_head.py > get_targets()`*

This snippet shows how the ground-truth fractional offsets are calculated during training to supervise the SCO branch.

```python
if self.use_hrh_sco:
    # center_x_hm is the continuous coordinate on the upsampled grid
    # ctx_int is the integer pixel index
    # The target is the residual: Fractional position - Integer pixel
    center_offset_target[batch_id, 0, cty_int, ctx_int] = center_x_hm[j] - ctx_int
    center_offset_target[batch_id, 1, cty_int, ctx_int] = center_y_hm[j] - cty_int
    center_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1.0
```

---

### **4. Subpixel Inference Decoding**
*File: `models/dense_heads/yolc_head.py > decode_heatmap()`*

The decoding logic was amended to apply the learned offsets to the integer heatmap peaks, providing microscopic localization accuracy.

```python
if center_offset_pred is not None:
    # 1. Gather the subpixel offsets (dx, dy) at predicted heatmap peaks
    offsets = transpose_and_gather_feat(center_offset_pred, batch_index_large)
    
    # 2. Refine high-resolution peaks: Integer Coordinate + Subpixel Offset
    topk_xs_high_res = topk_xs_large.float() + offsets[..., 0]
    topk_ys_high_res = topk_ys_large.float() + offsets[..., 1]
    
    # 3. Map back to the global image scale
    topk_xs = topk_xs_high_res / upsample_ratio
    topk_ys = topk_ys_high_res / upsample_ratio
```

---

### **5. SCO Loss Function Integration**
*File: `models/dense_heads/yolc_head.py > loss()`*

Incorporation of the L1 loss for the Subpixel Center Offset branch into the global loss calculation.

```python
if self.use_hrh_sco:
    # Predict decimal refinements and supervise with L1 Loss
    loss_center_offset = self.loss_center_offset(
        center_offset_pred, 
        center_offset_target, 
        center_offset_target_weight, 
        avg_factor=avg_factor)
    
    losses['loss_center_offset'] = loss_center_offset
```
