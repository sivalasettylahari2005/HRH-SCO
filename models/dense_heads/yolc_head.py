# Copyright (c) OpenMMLab. All rights reserved.
from operator import gt
import re
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms, soft_nms, DeformConv2d
from mmcv.runner import force_fp32

from mmdet.core import bbox, multi_apply, MlvlPointGenerator
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

from kornia.filters import gaussian_blur2d


@HEADS.register_module()
class YOLCHead(BaseDenseHead, BBoxTestMixin):
    """YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images.
    Paper link <https://arxiv.org/abs/2404.06180>

    Modified for HRH-SCO: High-Resolution Heatmap with Subpixel Center Offsets.
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_local=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_xywh=dict(type='GWDLoss', loss_weight=2.0),
                 use_hrh_sco=False,
                 heatmap_upsample_num=2, # Default x4 upsample if enabled (Stride 4 if input Stride 16) - Adjust based on requirements
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YOLCHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.use_hrh_sco = use_hrh_sco
        self.heatmap_upsample_num = heatmap_upsample_num if use_hrh_sco else 0
        self.upsample_ratio = 2 ** self.heatmap_upsample_num

        # Build heads
        self.local_head = self._build_loc_head(in_channel, num_classes)
        
        if self.use_hrh_sco:
            self.offset_head = self._build_offset_head(in_channel, 2)
            self.loss_center_offset = build_loss(dict(type='L1Loss', loss_weight=1.0))
        else:
            self.offset_head = None
            self.loss_center_offset = None

        self._build_reg_head(in_channel, feat_channel)

        self.loss_center_local = build_loss(loss_center_local)
        self.loss_xywh_coarse = build_loss(loss_xywh)
        self.loss_xywh_refine = build_loss(loss_xywh)
        loss_l1 = dict(type='L1Loss', loss_weight=0.5)
        self.loss_xywh_coarse_l1 = build_loss(loss_l1)
        self.loss_xywh_refine_l1 = build_loss(loss_l1)

        strides=[1] # Should be replaced by actual stride if using MlvlPointGenerator relative
        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        dcn_base = np.arange(-1, 2).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, 3)
        dcn_base_x = np.tile(dcn_base, 3)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_loc_head(self, in_channel, out_channel):
        """Build head for high resolution heatmap branch."""
        if not getattr(self, 'use_hrh_sco', False):
            # Baseline Architecture: EXACT match for yolc.pth
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
            # Proposed HRH-SCO Architecture: Dynamic upsampling
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

    def _build_offset_head(self, in_channel, out_channel):
        """Build head for subpixel center offsets."""
        # Matches resolution of loc_head
        layers = []
        layers.append(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        curr_channel = in_channel
        for i in range(self.heatmap_upsample_num):
             # Logic to match loc_head upsampling
             layers.append(nn.ConvTranspose2d(curr_channel, curr_channel // 2, 4, stride=2, padding=1))
             layers.append(nn.ReLU(inplace=True))
             curr_channel = curr_channel // 2

        layers.append(nn.Conv2d(curr_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _build_reg_head(self, in_channel, feat_channel):
        """Build head for regression branch (coarse resolution)."""
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, feat_channel, kernel_size=3, padding=1))
        
        self.xywh_init = nn.Conv2d(feat_channel, 4, kernel_size=1)
        self.bbox_offset = nn.Conv2d(feat_channel, 18, kernel_size=1)
        # Force standard Conv2d instead of DeformConv2d
        self.xywh_refine = nn.Conv2d(feat_channel, 4, kernel_size=3, padding=1)

    def init_weights(self):
        # Initialize reg_conv, xywh_init, bbox_offset, xywh_refine
        for head in [self.reg_conv, self.xywh_init, self.bbox_offset, self.xywh_refine]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Initialize offset_head if it exists
        if self.use_hrh_sco:
            for m in self.offset_head.modules():
                 if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                 elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)

        # Initialize local_head
        for m in self.local_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def get_points(self, featmap_sizes, img_metas, device):
        num_imgs = len(img_metas)
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=False)
        points_list = [multi_level_points[0].clone() for _ in range(num_imgs)]
        return points_list

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_local_pred = self.local_head(feat).sigmoid()
        
        if self.use_hrh_sco:
            center_offset_pred = self.offset_head(feat)
        else:
            center_offset_pred = None
        
        reg_feat = self.reg_conv(feat).contiguous()
        xywh_pred_coarse = self.xywh_init(reg_feat)
        featmap_sizes = [xywh_pred_coarse.size()[-2:]]
        device = xywh_pred_coarse.device
        
        center_points = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=False)[0]
        bbox_pred = xywh_pred_coarse.detach().permute(0, 2, 3, 1).reshape(xywh_pred_coarse.size(0), -1, 4).contiguous()
        bbox_pred[:, :, :2] = bbox_pred[:, :, :2] + center_points.unsqueeze(0)
        
        offset = self.bbox_offset(reg_feat).sigmoid()
        dcn_offset = self.gen_dcn_offset(bbox_pred.permute(0, 2, 1), offset, center_points)
        xywh_pred_refine = self.xywh_refine(reg_feat)
        
        return center_local_pred, center_offset_pred, xywh_pred_coarse, xywh_pred_refine
    
    def gen_dcn_offset(self, bbox_pred, offset, center_points):
        B, _, H, W = offset.shape
        dcn_offset = offset.new(B, 9*2, H, W)
        bbox_pred = bbox_pred.view(B, 4, H, W)
        bbox_pred[:, 0:2, :, :,] = bbox_pred[:, 0:2, :, :,] - bbox_pred[:, 2:4, :, :,]
        bbox_pred[:, 2:4, :, :,] = 2 * bbox_pred[:, 2:4, :, :,]
        
        dcn_offset[:, 0::2, :, :] = bbox_pred[:, 0, :, :].unsqueeze(1) + bbox_pred[:, 2, :, :].unsqueeze(1) * offset[:, 0::2, :, :]
        dcn_offset[:, 1::2, :, :] = bbox_pred[:, 1, :, :].unsqueeze(1) + bbox_pred[:, 3, :, :].unsqueeze(1) * offset[:, 1::2, :, :]

        dcn_base_offset = self.dcn_base_offset.type_as(dcn_offset)
        dcn_anchor_offset = center_points.view(H, W, 2).repeat(B, 1, 1, 1).repeat(1, 1, 1, 9).permute(0, 3, 1, 2)
        dcn_anchor_offset += dcn_base_offset
        return dcn_offset - dcn_anchor_offset

    @force_fp32(apply_to=('center_local_preds', 'center_offset_preds', 'xywh_preds_coarse', 'xywh_preds_refine'))
    def loss(self,
             center_local_preds,
             center_offset_preds,
             xywh_preds_coarse,
             xywh_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
             
        center_local_pred = center_local_preds[0]
        center_offset_pred = center_offset_preds[0] # May be None
        xywh_pred_coarse = xywh_preds_coarse[0]
        xywh_pred_refine = xywh_preds_refine[0]

        featmap_sizes = [featmap.size()[-2:] for featmap in xywh_preds_coarse]
        device = xywh_pred_coarse.device

        center_points = self.get_points(featmap_sizes, img_metas, device)[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     xywh_pred_coarse.shape,
                                                     img_metas[0]['pad_shape'])

        center_local_target = target_result['center_heatmap_target']
        xywh_target = target_result['xywh_target']
        xywh_target_weight = target_result['xywh_target_weight']

        # BBox L2
        B, _, H, W = xywh_pred_coarse.shape
        bbox_pred_coarse = xywh_pred_coarse.permute(0, 2, 3, 1).reshape(B, -1, 4)
        bbox_pred_coarse[:, :, :2] = bbox_pred_coarse[:, :, :2] + center_points.unsqueeze(0)
        
        bbox_pred_refine = xywh_pred_refine.permute(0, 2, 3, 1).reshape(B, -1, 4)
        bbox_pred_refine[:, :, :2] = bbox_pred_refine[:, :, :2] + center_points.unsqueeze(0)
        
        xywh_target = xywh_target.reshape(B, -1, 4)
        xywh_target_weight = xywh_target_weight.reshape(B, -1)
        xywh_l1target_weight = target_result['xywh_l1target_weight'].reshape(B, -1, 4)

        loss_center_heatmap = self.loss_center_local(
            center_local_pred, center_local_target, avg_factor=avg_factor)
        
        losses = dict(loss_center_heatmap=loss_center_heatmap)

        if self.use_hrh_sco:
            center_offset_target = target_result['center_offset_target']
            center_offset_target_weight = target_result['center_offset_target_weight']
            
            loss_center_offset = self.loss_center_offset(
                center_offset_pred, center_offset_target, center_offset_target_weight, avg_factor=avg_factor)
            losses['loss_center_offset'] = loss_center_offset

        loss_xywh_coarse = self.loss_xywh_coarse(
            bbox_pred_coarse, xywh_target, xywh_target_weight, avg_factor=avg_factor)
        loss_xywh_refine = self.loss_xywh_refine(
            bbox_pred_refine, xywh_target, xywh_target_weight, avg_factor=avg_factor)
        loss_xywh_coarse_l1 = self.loss_xywh_coarse_l1(
            bbox_pred_coarse, xywh_target, xywh_l1target_weight, avg_factor=avg_factor)
        loss_xywh_refine_l1 = self.loss_xywh_refine_l1(
            bbox_pred_refine, xywh_target, xywh_l1target_weight, avg_factor=avg_factor)
            
        losses.update(dict(
             loss_xywh_coarse=loss_xywh_coarse,
             loss_xywh_coarse_l1=loss_xywh_coarse_l1,
             loss_xywh_refine=loss_xywh_refine,
             loss_xywh_refine_l1=loss_xywh_refine_l1))
             
        return losses

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape
        
        # Calculate derived heatmap size
        heatmap_h = feat_h * self.upsample_ratio
        heatmap_w = feat_w * self.upsample_ratio
        
        # Ratios for mapping image to feature/heatmap
        # feat ratios (for bbox)
        width_ratio_feat = float(feat_w / img_w)
        height_ratio_feat = float(feat_h / img_h)

        # Heatmap ratios (for gaussian and offset)
        width_ratio_hm = float(heatmap_w / img_w)
        height_ratio_hm = float(heatmap_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, heatmap_h, heatmap_w])
        
        if self.use_hrh_sco:
             center_offset_target = gt_bboxes[-1].new_zeros([bs, 2, heatmap_h, heatmap_w])
             center_offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, heatmap_h, heatmap_w])
        else:
             center_offset_target = None
             center_offset_target_weight = None

        xywh_target = gt_bboxes[-1].new_zeros([bs, feat_h, feat_w, 4])
        xywh_target_weight = gt_bboxes[-1].new_zeros([bs, feat_h, feat_w])
        xywh_l1target_weight = gt_bboxes[-1].new_zeros([bs, feat_h, feat_w, 4])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            
            # Centers on Heatmap Grid
            center_x_hm = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio_hm / 2
            center_y_hm = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio_hm / 2
            gt_centers_hm = torch.cat((center_x_hm, center_y_hm), dim=1)

            # Centers on Feature Grid (for BBox)
            center_x_feat = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio_feat / 2
            center_y_feat = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio_feat / 2
            gt_centers_feat = torch.cat((center_x_feat, center_y_feat), dim=1)

            for j, ct_hm in enumerate(gt_centers_hm):
                # Heatmap Target Generation
                # Use High-Res Grid coordinates
                ctx_int, cty_int = ct_hm.int()
                
                # BBox scaling (use feature grid ratio, but can rely on image size)
                box_h = (gt_bbox[j][3] - gt_bbox[j][1])
                box_w = (gt_bbox[j][2] - gt_bbox[j][0])
                scale_box_h = box_h * height_ratio_feat # Scale to feature grid
                scale_box_w = box_w * width_ratio_feat # Scale to feature grid

                # Radius calculation: needs to be relative to the Heatmap Grid so it covers appropriate pixels
                # gaussian_radius expects box size. If we use scaled sizes, radius is in grid pixels.
                # If heatmap is 4x larger, box size on heatmap is 4x larger, so radius is 4x larger.
                # This is correct for high-res Gaussian.
                scale_box_h_hm = box_h * height_ratio_hm
                scale_box_w_hm = box_w * width_ratio_hm
                
                radius = gaussian_radius([scale_box_h_hm, scale_box_w_hm], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)
                
                if self.use_hrh_sco:
                    # HRH-SCO: Offset Targets
                    # Target is the fractional part of the coordinate on the high-res grid
                    center_offset_target[batch_id, 0, cty_int, ctx_int] = center_x_hm[j] - ctx_int
                    center_offset_target[batch_id, 1, cty_int, ctx_int] = center_y_hm[j] - cty_int
                    center_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1.0

                # BBox Targets (Low Res / Coarse Grid)
                # We need to map the high-res center back to the coarse grid to find the assignment
                # Coarse index should be consistent.
                # ct_feat is the float center on coarse grid.
                ct_feat = gt_centers_feat[j]
                ctx_feat_int, cty_feat_int = ct_feat.int()

                if cty_feat_int >= feat_h or ctx_feat_int >= feat_w:
                    continue

                xywh_target[batch_id, cty_feat_int, ctx_feat_int, 0] = ct_feat[0]
                xywh_target[batch_id, cty_feat_int, ctx_feat_int, 1] = ct_feat[1]
                xywh_target[batch_id, cty_feat_int, ctx_feat_int, 2] = scale_box_w/2
                xywh_target[batch_id, cty_feat_int, ctx_feat_int, 3] = scale_box_h/2

                xywh_target_weight[batch_id, cty_feat_int, ctx_feat_int] = 1
                xywh_l1target_weight[batch_id, cty_feat_int, ctx_feat_int, 0:2] = 1.0
                xywh_l1target_weight[batch_id, cty_feat_int, ctx_feat_int, 2:4] = 0.2

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            center_offset_target=center_offset_target,
            center_offset_target_weight=center_offset_target_weight,
            xywh_target=xywh_target,
            xywh_target_weight=xywh_target_weight,
            xywh_l1target_weight=xywh_l1target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   center_offset_preds,
                   xywh_preds_init,
                   xywh_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        if center_offset_preds is not None and len(center_offset_preds) > 0:
            center_offset_pred = center_offset_preds[0]
        else:
            center_offset_pred = None
        
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta.get('border', [0, 0, 0, 0]) for img_meta in img_metas]
        
        center_heatmap_preds[0] = gaussian_blur2d(center_heatmap_preds[0], (3, 3), sigma=(1, 1))
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            xywh_preds[0],
            img_metas[0]['batch_input_shape'],
            center_offset_pred=center_offset_pred,
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] -= batch_border

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results


    def decode_heatmap(self,
                       center_heatmap_pred,
                       xywh_pred,
                       img_shape,
                       center_offset_pred=None,
                       k=100,
                       kernel=3):
        
        # center_heatmap_pred [bs, 1, H, W] (High Res if enabled)
        # xywh_pred [bs, 4, h, w] (Low Res)
        
        height, width = xywh_pred.shape[2:] # Coarse dims
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys_large, topk_xs_large = self.get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index_large, batch_topk_labels = batch_dets
        
        # Map large grid indices to coarse grid
        # self.upsample_ratio should be available, or we derive it
        upsample_ratio = self.upsample_ratio
        
        topk_ys_coarse = (topk_ys_large / upsample_ratio).long()
        topk_xs_coarse = (topk_xs_large / upsample_ratio).long()
        
        # Clamp to bounds just in case
        topk_ys_coarse = topk_ys_coarse.clamp(max=height-1)
        topk_xs_coarse = topk_xs_coarse.clamp(max=width-1)
        
        batch_index_coarse = topk_ys_coarse * width + topk_xs_coarse
        
        xywh = transpose_and_gather_feat(xywh_pred, batch_index_coarse)

        # Base center coordinate on the grid
        # xywh[...] is (x,y, w/2, h/2). x,y are absolute on Coarse Grid? 
        # Check forward(): 
        # bbox_pred[:, :, :2] = bbox_pred[:, :, :2] + center_points.unsqueeze(0)
        # center_points are on Coarse Grid integers (0,0), (0,1)...
        # So xywh is absolute coordinate on Coarse Grid.
        
        # But we want to use the Hig-Res Peak + Offset for better localization
        if center_offset_pred is not None:
             # Gather subpixel offsets [B, K, 2] from high-res map
            offsets = transpose_and_gather_feat(center_offset_pred, batch_index_large)
            
            # Reconstruct high-res coordinate
            # Peak integer location (large) + offset (subpixel)
            topk_xs_high_res = topk_xs_large.float() + offsets[..., 0]
            topk_ys_high_res = topk_ys_large.float() + offsets[..., 1]
            
            # Convert back to Coarse Grid scale for consistent bbox operations (or Image scale directly)
            # Image Scale = High Res * (img_w / heatmap_w)
            # heatmap_w = width * upsample_ratio
            # img_w / heatmap_w is likely the global stride / upsample_ratio.
            
            # Using Coarse scale to calculate boxes:
            topk_xs = topk_xs_high_res / upsample_ratio
            topk_ys = topk_ys_high_res / upsample_ratio
        else:
            # Fallback to coarse prediction if offsets not available
            # Or use heatmap peak mapped to coarse
            topk_xs = topk_xs_large.float() / upsample_ratio
            topk_ys = topk_ys_large.float() / upsample_ratio
            # Prefer using xywh predicted centers? No, heatmap is main localization.
            # Usually CenterNet refines center using 'reg' (offset on coarse).
            # But here we have HRH-SCO.
            
            # If standard YOLC uses reg, we should use that from xywh.
            # xywh[..., 0] is the predicted center X on coarse grid.
            # If not using SCO, use that.
            if not self.use_hrh_sco:
                topk_xs = xywh[..., 0]
                topk_ys = xywh[..., 1]

        tl_x = (topk_xs - xywh[..., 2]) * (inp_w / width)
        tl_y = (topk_ys - xywh[..., 3]) * (inp_h / height)
        br_x = (topk_xs + xywh[..., 2]) * (inp_w / width)
        br_y = (topk_ys + xywh[..., 3]) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def get_topk_from_heatmap(self, center_heatmap, k=20):
        batch, _, height, width = center_heatmap.size()
        topk_scores, topk_inds = torch.topk(center_heatmap.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


    def get_local_minimum(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        heat = 1 - torch.div(heat, 10)
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4].contiguous(),
                                       bboxes[:, -1].contiguous(), labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def simple_test(self, feats, img_metas, rescale=False, crop=False):
        if crop:
            return self.simple_test_bboxes(feats, img_metas, rescale=rescale, crop=True)
        else:
            return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    def simple_test_bboxes(self, feats, img_metas, rescale=False, crop=False):
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        if crop:
            return self.LSM(outs[0], img_metas), results_list
        else:
            return results_list

    def LSM(self, center_heatmap_preds, img_metas):
        center_heatmap_pred = center_heatmap_preds[0]
        locmap = torch.max(center_heatmap_pred, dim=1, keepdim=True)[0].cpu().numpy()
        coord = self.findclusters(locmap, find_max=True, fname=["test"])
        border_pixs = [img_meta['border'] for img_meta in img_metas]
        coord[:, 0] = coord[:, 0] - border_pixs[0][2]
        coord[:, 1] = coord[:, 1] - border_pixs[0][0]
        return coord

    def findclusters(self, heatmap, find_max, fname):
        heatmap = 1 - heatmap
        heatmap = 255*heatmap / np.max(heatmap)
        heatmap = heatmap[0][0]

        gray = heatmap.astype(np.uint8)
        Thresh = 10.0/11.0 * 255.0
        ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)

        binmap = binary.copy()
        binmap[binmap==255] = 1
        density_map = np.zeros((16, 10))
        w_stride = binary.shape[1]//16
        h_stride = binary.shape[0]//10
        for i in range(16):
            for j in range(10):
                x1 = w_stride*i
                y1 = h_stride*j
                x2 = min(x1+w_stride, binary.shape[1])
                y2 = min(y1+h_stride, binary.shape[0])
                density_map[i][j] = binmap[y1:y2,x1:x2].sum()

        d = density_map.flatten()
        topk = 15
        idx = d.argsort()[-topk:][::-1]
        grid_idx = idx.copy()
        idx_x = idx // 10 * w_stride
        idx_x = idx_x.reshape((-1, 1))
        idx_y = idx % 10 * h_stride
        idx_y = idx_y.reshape((-1, 1))
        idx = np.concatenate((idx_x, idx_y), axis=1)
        grid = np.zeros((16, 10))
        for item in grid_idx:
            x1 = item // 10
            y1 = item % 10
            grid[x1, y1] = 255
        result = split_overlay_map(grid)
        result = np.array(result)
        result[:,0::2] = np.clip(result[:, 0::2]*w_stride, 0,  binary.shape[1])
        result[:,1::2] = np.clip(result[:, 1::2]*h_stride, 0,  binary.shape[0])
        
        result[:, 2] = result[:, 2] - result[:, 0]
        result[:, 3] = result[:, 3] - result[:, 1]
        return result


def split_overlay_map(grid):
    # This function is modified from https://github.com/Cli98/DMNet
    """
        Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
        :param grid: desnity mask to connect
        :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    if 0 <= i_cp < m and 0 <= j_cp < n and grid[i_cp][j_cp] == 255:
                        top = min(i_cp, top)
                        left = min(j_cp, left)
                        bot = max(i_cp, bot)
                        right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])

                            queue.append([i_cp - 1, j_cp - 1])
                            queue.append([i_cp - 1, j_cp + 1])
                            queue.append([i_cp + 1, j_cp - 1])
                            queue.append([i_cp + 1, j_cp + 1])
                count += 1
                result.append([max(0, top), max(0, left), min(bot+1, m), min(right+1, n)])
    return result
