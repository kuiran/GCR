from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmdet.models.utils import build_transformer
from mmcv.runner import auto_fp16, force_fp32
import torch.nn as nn
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.core import bbox2roi
import torch


import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.core import AnchorGenerator, build_prior_generator
from mmdet.models.builder import HEADS
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy

from torch.nn import functional as F
import clip


@HEADS.register_module()
class SinglePosAnchorRPNHead(BaseModule):
    """
    Args:

    """

    def __init__(self,
                 anchor_generator_cfg=dict(
                     type='AnchorGenerator',
                     strides=[4, 8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     scales=[1, 4, 8]
                 ),
                 num_proposals=36,
                 proposal_feature_channel=256,
                 w_ratio=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                 h_ratio=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                 init_cfg=None,
                 **kwargs):
        super(SinglePosAnchorRPNHead1, self).__init__(init_cfg)
        self.anchor_generator = build_prior_generator(anchor_generator_cfg)
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.point_init_proposal_features = nn.Embedding(self.num_proposals, self.proposal_feature_channel)
        self.bbox_init_proposal_features = nn.Embedding(self.num_proposals, self.proposal_feature_channel)
        self.pred_iou_proposal_features = nn.Embedding(self.num_proposals, self.proposal_feature_channel)

    def _decode_init_proposals(self, imgs, img_metas, points=None, bbox=None):
        """
        Args:
            imgs: list[Tensor]: FPN featurs. list(b,c,h, w)
            img_metas: list[dict]
            points: tensor(b, 1, 2)
            bbox: tensor(b, 4)
        Return:
            proposals (Tensor): Decoded proposal bboxes,
                  has shape (batch_size, num_proposals, 4).
            init_proposal_features (Tensor): Expanded proposal
                  features, has shape
                  (batch_size, num_proposals, proposal_feature_channel).
            imgs_whwh (Tensor): Tensor with shape
                  (b, num_proposals, 4), the dimension means
                  [img_width, img_height, img_width, img_height].
        """
        num_imgs = len(img_metas)
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        point_init_proposal_features = self.point_init_proposal_features.weight.clone()
        point_init_proposal_features = point_init_proposal_features[None].expand(
            num_imgs, *point_init_proposal_features.size()
        )
        box_init_proposal_features = self.bbox_init_proposal_features.weight.clone()
        box_init_proposal_features = box_init_proposal_features[None].expand(
            num_imgs, *box_init_proposal_features.size()
        )
        pred_iou_proposal_features = self.pred_iou_proposal_features.weight.clone()
        pred_iou_proposal_features = pred_iou_proposal_features[None].expand(
            num_imgs, *pred_iou_proposal_features.size()
        )
        if points is not None:
            proposals = self.anchor_generator.gen_base_anchors()
            proposals = torch.stack(proposals).reshape([-1, 4]).unsqueeze(0).repeat([num_imgs, 1, 1]).cuda()
            proposal_num = proposals.shape[1]
            points = points.repeat([1, proposal_num, 2]).cuda()
            proposals = proposals + points
        elif bbox is not None:
            bbox_xy_wh = bbox_xyxy_to_cxcywh(bbox)
            w, h = bbox_xy_wh[:, 2].unsqueeze(1), bbox_xy_wh[:, 3].unsqueeze(1)
            t_w_ratio = torch.tensor(self.w_ratio).unsqueeze(0)
            t_h_ratio = torch.tensor(self.h_ratio).unsqueeze(0)
            # only bbox test
            t_w_ratio = t_w_ratio.cuda()
            t_h_ratio = t_h_ratio.cuda()

            shake_w = torch.mm(w, t_w_ratio)
            shake_h = torch.mm(h, t_h_ratio)
            shake_w = torch.repeat_interleave(shake_w, 6, dim=1)
            shake_h = shake_h.repeat(1, 6)
            center_x, center_y = bbox_xy_wh[:, 0].unsqueeze(1), bbox_xy_wh[:, 1].unsqueeze(1)
            center_x, center_y = center_x.repeat(1, self.num_proposals), center_y.repeat(1, self.num_proposals)
            shake_bbox_xy_wh = torch.stack((center_x, center_y, shake_w, shake_h)).permute(1, 2, 0)
            proposals = bbox_cxcywh_to_xyxy(shake_bbox_xy_wh).cuda()
        else:
            raise NotImplementedError

        return proposals, imgs_whwh, point_init_proposal_features, box_init_proposal_features, pred_iou_proposal_features

    def forward_train(self, img, img_metas, points=None, bbox=None):
        return self._decode_init_proposals(img, img_metas, points, bbox)

    def simple_test_rpn(self, img, img_metas, points=None, bbox=None):
        return self._decode_init_proposals(img, img_metas, points, bbox)


from mmtrack.models import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector
import numpy as np
import random
import math


@MODELS.register_module()
class GCR_model(TwoStageDetector):
    def __init__(self, *args, **kwargs):
        super(UnifiedTrackersInput2, self).__init__(*args, **kwargs)

    """
    new_generation
    """
    def gen_points(self, bboxes):
        """
        Args:
            bboxes: list[tensor(1, 4)]
        Return:
            random_points: tensor(b, 2)
        """
        random_points = []
        for bbox in bboxes:
            cxcywh = bbox_xyxy_to_cxcywh(bbox)
            cx = cxcywh[:, 0].item()
            cy = cxcywh[:, 1].item()
            w = cxcywh[:, 2].item()
            h = cxcywh[:, 3].item()
    
            a = w / self.train_cfg.gen_point_factor
            b = h / self.train_cfg.gen_point_factor
    
            if a == 0:
                random_point_x = 0.
                random_point_y = 0.
            else:
                _random_x = random.uniform(-a, a)
                _random_y_range = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)
                _random_y = random.uniform(-_random_y_range, _random_y_range)
                random_x = _random_x
                random_y = _random_y
    
                random_point_x = round((cx + random_x), 2)
                random_point_y = round((cy + random_y), 2)
            random_points.append(torch.tensor([[random_point_x, random_point_y]]))
        random_points = torch.cat(random_points).unsqueeze(1)
        return random_points

    def gen_noisy_bbox(self, bboxes):
        """
        Args:
            bboxes: list[tensor(1, 4)]
        Return:
            noise_bboxes: tensor(b, 4)
        """
        noise_bboxes = []
        for bbox in bboxes:
            # bbox (1, 4)
            cxcywh = bbox_xyxy_to_cxcywh(bbox)
            cx = cxcywh[:, 0].item()
            cy = cxcywh[:, 1].item()
            w = cxcywh[:, 2].item()
            h = cxcywh[:, 3].item()
            cx1 = np.random.uniform(low=-0.4, high=0.4, size=None) * w + cx
            cy1 = np.random.uniform(low=-0.4, high=0.4, size=None) * h + cy
            w1 = (np.random.uniform(low=-0.4, high=0.4, size=None) + 1) * w
            h1 = (np.random.uniform(low=-0.4, high=0.4, size=None) + 1) * h
            xywh = bbox_cxcywh_to_xyxy(torch.tensor([[cx1, cy1, w1, h1]]))
            noise_bboxes.append(xywh)
        noise_bboxes = torch.cat(noise_bboxes)
        return noise_bboxes

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      points=None,
                      **kwargs):
        """
                Args:
                    img: (b, c, h, w) -> x:tuple(tensor(b, c, h ,w))
                    img_metas: list[dict]
                    gt_bboxes: (b, 4) [tl_x, tl_y, br_x, br_y]
                    points:
                    proposals: xxx
                Return:
                    roi_losses: dict
                    bbox_pred: (b, 4)
        """
        x = self.extract_feat(img)
        gt_bboxes = [bbox.float() for bbox in gt_bboxes]

        point_or_bbox = 0
        if point_or_bbox == 0:
            points = self.gen_points(gt_bboxes)
            proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
                self.rpn_head.forward_train(x, img_metas, points=points)
            gt_points = points
            input_mode = 'point'
        elif point_or_bbox == 1:
            raise NotImplementedError
            noisy_bboxes = self.gen_noisy_bbox(gt_bboxes)
            proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
                self.rpn_head.forward_train(x, img_metas, bbox=noisy_bboxes)
            input_mode = 'bbox'
        roi_losses, _ = self.roi_head.forward_train(
            x,
            gt_points,
            proposal_boxes,
            point_proposal_features,
            bbox_proposal_features,
            pred_iou_proposal_features,
            img_metas,
            gt_bboxes,
            input_mode=input_mode,
            imgs_whwh=imgs_whwh
        )
        return roi_losses

    def simple_test(self, img, img_metas, points, gt_bboxes, noisy_bbox=None, rescale=False):
        """
        Args:
            img:
            points: (b, 1, 2)
            noise_box: (b, 4)
        """
        x = self.extract_feat(img)
        if self.test_cfg.test_mode == 'point':
            points = points[0].unsqueeze(0)
            proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
                self.rpn_head.forward_train(x, img_metas, points=points)
            input_mode = 'point'
        elif self.test_cfg.test_mode == 'bbox':
            proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
                self.rpn_head.forward_train(x, img_metas, bbox=noisy_bbox)
            input_mode = 'bbox'
        else:
            raise NotImplementedError

        if input_mode == 'point':
            proposal_feat = point_proposal_features
        elif input_mode == 'bbox':
            proposal_feat = bbox_proposal_features

        result = self.roi_head.simple_test(
            x,
            points,
            proposal_boxes,
            point_proposal_features,
            bbox_proposal_features,
            pred_iou_proposal_features,
            input_mode,
            img_metas,
            imgs_whwh
        )
        return dict(ori_pred_bbox=result['ori_pred_bbox'])


class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class AttentionHead1(BaseModule):
    def __init__(self, in_channels, hide_channels, out_channels, temprature=30, init_weight=True):
        super(AttentionHead1, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_channels < hide_channels
        self.in_channels = in_channels
        self.hide_channels = hide_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hide_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hide_channels, out_channels, kernel_size=1, bias=False)
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)
        att = self.net(att).view(x.shape[0], -1)
        return F.softmax(att / self.temprature, -1)