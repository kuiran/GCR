import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models.builder import build_backbone, build_head, build_neck
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torchvision.transforms.functional import normalize

from ..builder import MODELS
from .base import BaseSingleObjectTracker

from .point_track import PointTracking
from .stark import Stark


@MODELS.register_module()
class PointTrackingV2(Stark):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 pt_reg_head=None,
                 head=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        super(PointTrackingV2, self).__init__(backbone,
                                              neck,
                                              head,
                                              init_cfg,
                                              frozen_modules,
                                              train_cfg,
                                              test_cfg)
        self.pt_reg_head = build_head(pt_reg_head)

    def forward_train(self,
                      img,
                      img_metas,
                      search_img,
                      search_img_metas,
                      points,
                      gt_bboxes,
                      padding_mask,
                      search_gt_bboxes,
                      search_padding_mask,
                      search_gt_labels=None,
                      **kwargs):
        """
        Args:
            imgs: (B, num_templates, C, H, W)
            img_metas: list[dict]

        """
        b, num_templates, c, h, w = img.shape
        img_ = img.reshape(-1, c, h, w)  # (B, num_templates, C, H, W) => (B*num_templates, C, H, W)
        gt_bboxes_ = []
        for bbox in gt_bboxes:
            gt_bboxes_.append(bbox[0][1:].unsqueeze(0))
            gt_bboxes_.append(bbox[1][1:].unsqueeze(0))
        gt_bboxes_ = [bbox.float() for bbox in gt_bboxes_]
        img_metas_ = []
        for i in range(len(img_metas)):
            img_metas_.extend(img_metas[i])
        img_feats = self.extract_feat(img_)  # img_feats(B*num_templates, C, H//stride, W//stride)
        img_feats_ = []
        for i in range(img_feats[0].shape[0]):
            img_feats_.append(img_feats[0][i].unsqueeze(0))
        # img_feats_:list(Tensor) (B*num_templates, (1, C, H, W))

        imgs_whwh = []
        for meta in img_metas_:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(img_[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        # gt_r_points: (B, 2)
        points = points.reshape(-1, points.shape[2])[:, 1:]
        pt_reg_losses, pred_bbox = self.pt_reg_head.forward_train(
            img_feats_,
            points,
            img_metas_,
            gt_bboxes_,
            imgs_whwh
        )

        losses = dict()
        losses.update(pt_reg_losses)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        pass
