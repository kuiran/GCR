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

# from . import SinglePosAnchorRPNHead
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
class GCR(TwoStageDetector):
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
        # use single fpn
        # x = tuple(x[2].unsqueeze(0))
        gt_bboxes = [bbox.float() for bbox in gt_bboxes]

        # gt_points = [_ for _ in points.float().squeeze(1)]

        # decide point or box
        # point_or_bbox = random.randint(0, 1)
        # print(point_or_bbox)
        point_or_bbox = 0
        if point_or_bbox == 0:
            points = self.gen_points(gt_bboxes)
            proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
                self.rpn_head.forward_train(x, img_metas, points=points)
            # gt_points = [_ for _ in points.float().squeeze(1)]
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
        # use single fpn
        
        # x = tuple(x[2].unsqueeze(0))

        # if points is not None:
        #     noisy_points = points[0].squeeze(1)
        #     proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features = \
        #         self.rpn_head.forward_train(x, img_metas, points=noisy_points)
        #     input_mode = 'point'
        # elif noisy_bbox is not None:
        #     noisy_bbox = noisy_bbox[0].squeeze(1)
        #     proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features = \
        #         self.rpn_head.forward_train(x, img_metas, bbox=noisy_bbox)
        #     input_mode = 'bbox'

        # noisy_points = points[0].squeeze(1)
        # proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
        #     self.rpn_head.forward_train(x, img_metas, points=noisy_points)
        # input_mode = 'point'
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
        # noisy_bbox = noisy_bbox[0].float().squeeze(1)
        # proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
        #     self.rpn_head.forward_train(x, img_metas, bbox=noisy_bbox)
        # input_mode = 'bbox'
        #
        # result = self.roi_head.simple_test(
        #     x,
        #     proposal_boxes,
        #     bbox_proposal_features,
        #     pred_iou_proposal_features,
        #     input_mode,
        #     img_metas,
        #     imgs_whwh
        # )
        """
        vis
        """
        if self.test_cfg.vis_path is not None:
            # vis
            import cv2
            # import numpy as np
            import copy
            import os
            prefix = self.test_cfg.vis_path
            # vis_gt_bbox = gt_bboxes[0].squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)
            vis_pred_bbox = result['pred_bbox'].squeeze(0).cpu().numpy().astype(np.int32)
            vis_points = points[0].squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)

            file_name = img_metas[0]['filename']
            video_name = file_name.split('/')[-3]
            h, w, _ = img_metas[0]['img_shape']
            img1 = cv2.imread(file_name)
            img1 = cv2.resize(img1, (w, h))
            igs1 = copy.deepcopy(img1)
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            # igs1 = cv2.rectangle(igs1, (vis_gt_bbox[0], vis_gt_bbox[1]),
            #                      (vis_gt_bbox[2], vis_gt_bbox[3]),
            #                      color=(0, 128, 0))
            igs1 = cv2.rectangle(igs1, (vis_pred_bbox[0], vis_pred_bbox[1]),
                                 (vis_pred_bbox[2], vis_pred_bbox[3]),
                                 color=(0, 255, 255), thickness=2)
            if input_mode == 'bbox':
                vis_noisy_bbox = noisy_bbox.squeeze(0).cpu().numpy().astype(np.int32)
                igs1 = cv2.rectangle(igs1, (vis_noisy_bbox[0], vis_noisy_bbox[1]),
                                     (vis_noisy_bbox[2], vis_noisy_bbox[3]),
                                     color=(255, 255, 0))
            elif input_mode == 'point':
                igs1 = cv2.circle(igs1, (vis_points[0], vis_points[1]), 2, (0, 255, 0), 6)
            cv2.imwrite('{}/{}_{}.jpg'.format(prefix, video_name, 0), igs1)

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


from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.core import bbox2roi
from . import UnifiedTrackersInputRoIHead
from mmdet.models.builder import build_backbone, build_head, build_neck


@HEADS.register_module()
class UnifiedTrackersInputRoIHead(CascadeRoIHead):
    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=None,
                 attn_head=dict(
                     type='AttentionHead1',
                     in_channels=256,
                     hide_channels=2048,
                     out_channels=256
                 ),
                 box_refinest1=None,
                 pred_iou_head=dict(
                     type='UTIDIIBoxHead',
                     num_classes=1,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)
                 ),
                 bbox_head=dict(
                     type='DIIBoxHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_head=None,
                 custom_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super(UnifiedTrackersInputRoIHead1, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg
        )
        self.pred_iou_head = build_head(pred_iou_head)
        self.attn_head = build_head(attn_head)

        # saved text features file
        self.clip_feat_file = torch.load("/home/ubuntu/kuiran/tracking/clip_features/lasot_vit-b-32_prompt-a.pt")
        self.text_mlp = MLP(input_dim=512, hidden_dim=256, output_dim=256, num_layers=4)
        # self.text_mlp = nn.Linear(512, 256)
        self.batch_size = 8

        # 64.899
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")

        # text
        # self.text_linear = nn.Linear(512, 256)

        #
        # self.clip_model, _ = clip.load("RN50", device="cuda")
        # # text
        # self.text_linear = nn.Linear(1024, 256)

        #
        # self.clip_model, _ = clip.load("ViT-L/14", device="cuda")
        # # # text
        # self.text_linear = nn.Linear(768, 256)

        # self.clip_model, _ = clip.load("ViT-B/16", device="cuda")

        # box_refine
        if box_refinest1 is not None:
            self.box_refinest1 = build_head(box_refinest1)

        self.custom_cfg = custom_cfg
        # self.point_feat_sampler = PointFeatExtractor()

    def _bbox_forward(self, points, stage, x, rois, object_feats, img_metas, imgs_whwh, filter_roi_feat):
        """
            Args:
                points: (b, 2)
                stage: int
                x: List[Tensor]
                rois: (img_index, x1, y1, x2, y2)
                object_feats: (b, num_proposals, c)
            Returns:
                dict:
                    decode_bbox_pred: (b, num_proposals, 4) [tl_x, tl_y, br_x, br_y]
                    object_feats: (b, num_proposals, c)
                    detach_proposal_list: (b, num_proposals, 4)
        """
        num_imgs = len(img_metas)
        # if stage == 100 and filter_roi_feat is not None:
        #     bbox_head = self.bbox_head[stage]
        #     bbox_feats = filter_roi_feat.permute(0, 2, 1).reshape(-1, 512, 7, 7)
        # else:
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        rois = rois.float()
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # dynamic proposal_feat
        # attn_proposal_feat = self.attn_head(bbox_feats).unsqueeze(1)
        # object_feats = object_feats * attn_proposal_feat
        if self.custom_cfg.roi_connect:
            bbox_pred = bbox_head(bbox_feats, object_feats)
            object_feats, attn_feats = None, None
        else:
            bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats, img_metas, rois, stage)

        """
            proposal_list: list[tensor(1, 4)]
            4 distance
        """
        # proposal_list = self.bbox_head[stage].refine_bboxes_points(
        #     points,
        #     bbox_pred,
        #     imgs_whwh,
        #     img_metas)

        """
            xywh
        """
        # xywh decode
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(bbox_pred.size(1)) for _ in range(num_imgs)],
            img_metas)

        # xywh2points
        # bbox2point = rois[:, 1:]
        # points___ = (bbox2point[:, :2] + bbox2point[:, 2:]) / 2
        # proposal_list = self.bbox_head[stage].refine_bboxes(
        #     points___,
        #     bbox_pred.view(-1, bbox_pred.size(-1)),
        #     img_metas)

        bbox_results = dict(
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _bbox_forward_st1(self, points, stage, x, rois, object_feats, img_metas, imgs_whwh):
        """
            Args:
                points: (b, 2)
                stage: int
                x: List[Tensor]
                rois: (img_index, x1, y1, x2, y2)
                object_feats: (b, num_proposals, c)
            Returns:
                dict:
                    decode_bbox_pred: (b, num_proposals, 4) [tl_x, tl_y, br_x, br_y]
                    object_feats: (b, num_proposals, c)
                    detach_proposal_list: (b, num_proposals, 4)
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.box_refinest1
        rois = rois.float()
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)

        # add text_feat
        # text_info = []
        # for img_meta in img_metas:
        #     text_info.append(img_meta['ori_filename'].split('/')[0])
        # text = clip.tokenize(text_info).cuda()
        # with torch.no_grad():
        #     text_feature = self.text_linear(self.clip_model.encode_text(text).float()).unsqueeze(1)
        # text_feature = torch.repeat_interleave(text_feature, self.custom_cfg.num_proposals, 0).squeeze(1).unsqueeze(2).unsqueeze(3)
        # bbox_feats = bbox_feats + text_feature

        # add text_feat with loading saved text feature
        # text_feature = []
        # for img_meta in img_metas:
        #     t = img_meta['ori_filename'].split('/')[0]
        #     text_feature.append(self.text_mlp(self.clip_feat_file[t].float().cuda()))
        # text_feature = torch.cat(text_feature)
        # text_feature = torch.repeat_interleave(text_feature, self.custom_cfg.num_proposals, 0).unsqueeze(2).unsqueeze(3)
        # bbox_feats = bbox_feats + text_feature
        
        # dynamic proposal_feat
        # attn_proposal_feat = self.attn_head(bbox_feats).unsqueeze(1)
        # object_feats = object_feats * attn_proposal_feat

        if self.custom_cfg.roi_connect:
            bbox_pred = bbox_head(bbox_feats, object_feats)
            object_feats, attn_feats = None, None
        else:
            # ori to vis roi feat add img_metas and roi box
            # bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats)

            bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats, img_metas, rois, stage)

        # bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats)

        """
            proposal_list: list[tensor(1, 4)]
            4 distance
        """
        # proposal_list = self.bbox_head[stage].refine_bboxes_points(
        #     points,
        #     bbox_pred,
        #     imgs_whwh,
        #     img_metas)

        """
            xywh
        """
        # xywh decode
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(bbox_pred.size(1)) for _ in range(num_imgs)],
            img_metas)

        # xywh2points
        # bbox2point = rois[:, 1:]
        # points___ = (bbox2point[:, :2] + bbox2point[:, 2:]) / 2
        # proposal_list = self.bbox_head[stage].refine_bboxes(
        #     points___,
        #     bbox_pred.view(-1, bbox_pred.size(-1)),
        #     img_metas)

        bbox_results = dict(
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _iou_forward(self, points, stage, x, rois, object_feats, img_metas, imgs_whwh):
        """
            Args:
                points: (b, 2)
                stage: int
                x: List[Tensor]
                rois: (img_index, x1, y1, x2, y2)
                object_feats: (b, num_proposals, c)
            Returns:
                dict:
                    decode_bbox_pred: (b, num_proposals, 4) [tl_x, tl_y, br_x, br_y]
                    object_feats: (b, num_proposals, c)
                    detach_proposal_list: (b, num_proposals, 4)
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        # bbox_head = self.bbox_head[stage]
        rois = rois.float()
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        ious_pred = self.pred_iou_head(bbox_feats, object_feats)
        return ious_pred

    def extra_point_feat(self, x, point):
        point_stride = (point / 16.0).int().squeeze(1)
        point_feature = []
        x = x[0]
        # b, c, h, w = x.shape
        # point_stride[:, 0] = torch.clamp(point_stride[:, 0], min=0, max=w-1)
        # point_stride[:, 1] = torch.clamp(point_stride[:, 1], min=0, max=h-1)
        for i in range(x.shape[0]):
            point_feature.append(x[i, :, point_stride[i, 1], point_stride[i, 0]])
        point_feature = torch.stack(point_feature)
        return point_feature

    """
    sparse select + sparse refine
    """
    def forward_train(self,
                      x,
                      points,
                      proposal_boxes,
                      point_proposal_features,
                      box_proposal_features,
                      pred_iou_proposal_features,
                      img_metas,
                      gt_bboxes,
                      input_mode='point',
                      imgs_whwh=None,
                      text_feature=None):
        """
            Args:
                x: tuple(4) (b, c, h, w)
                points: list, [(2)]
                proposal_boxes: (b, num_proposals, 4)
                proposal_features: (b, num_proposals, c)
                img_metas: list(dict)
                gt_bboxes: (b, 4)
                gt_bboxes_ignore: xxx
                imgs_whwh: (b, 4) [w, h, w, h]
            Return:
                all_stage_loss: dict
                decode_bbox_pred: (b, 4)
        """
        num_imgs = len(img_metas)
        # num_proposals = proposal_boxes.size(1)
        # imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        sample_points = []
        # for i in range(len(points)):
        #     sample_points.append(points[i].unsqueeze(0).unsqueeze(0))
        #     points[i] = points[i].unsqueeze(0)
        # points = torch.cat(points).unsqueeze(1)
        # if input_mode == 'point':
        #     object_feats = point_proposal_features
        # elif input_mode == 'bbox':
        #     object_feats = box_proposal_features
        # get text_info replace proposal_feat
        all_stage_loss = {}
        if text_feature is not None:
            object_feats = text_feature
            st1_refine_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
        elif self.custom_cfg.use_point_feat:
            point_feat = self.extra_point_feat(x, points).unsqueeze(1)
            object_feats = point_feat
            st1_refine_proposal_features = point_feat.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = point_feat.repeat(1, self.custom_cfg.num_proposals, 1)
        elif self.custom_cfg.use_text_feat:
            text_feature = []
            for img_meta in img_metas:
                t = img_meta['ori_filename'].split('/')[0]
                text_feature.append(self.text_mlp(self.clip_feat_file[t].float().cuda()))
                # text_info.append(img_meta['ori_filename'].split('/')[0])   
            text_feature = torch.stack(text_feature)
            # text = clip.tokenize(text_info).cuda()
            # with torch.no_grad():
            #     text_feature = self.text_linear(self.clip_model.encode_text(text).float()).unsqueeze(1)
            # 512 dims
            # text_feature = self.clip_model.encode_text(text).float().unsqueeze(1)
            object_feats = text_feature
            # object_feats = box_proposal_features
            st1_refine_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            # st1_refine_proposal_features = point_proposal_features
            pred_iou_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            """
            # old code with load text feat
            text_info = []
            for img_meta in img_metas:
                text_info.append(img_meta['ori_filename'].split('/')[0])    
            # use saved text features
            
            text = clip.tokenize(text_info).cuda()
            with torch.no_grad():
                text_feature = self.text_linear(self.clip_model.encode_text(text).float()).unsqueeze(1)
            # 512 dims
            # text_feature = self.clip_model.encode_text(text).float().unsqueeze(1)
            object_feats = text_feature
            # object_feats = box_proposal_features
            st1_refine_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            # st1_refine_proposal_features = point_proposal_features
            pred_iou_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            """
        else:
            point_proposal_features = point_proposal_features[:, 0, :].unsqueeze(1)
            object_feats = point_proposal_features
            st1_refine_proposal_features = point_proposal_features.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = point_proposal_features.repeat(1, self.custom_cfg.num_proposals, 1)

        if self.custom_cfg.with_prototype_selection:
            """
            st1 refine
            """
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward_st1(points, 0, x, rois, st1_refine_proposal_features, img_metas, imgs_whwh)
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            single_stage_loss = self.box_refinest1.loss(
                decode_bbox_pred.view(-1, 4),
                gt_bboxes,
                imgs_whwh=imgs_whwh
            )
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage_-1_{key}'] = value * self.stage_loss_weights[0]
            proposal_list = bbox_results['detach_proposal_list']
            # use max iou proposal box directly
            # mi_gt = torch.stack(gt_bboxes).repeat([1, 36, 1]).reshape(-1, 4)
            # mi_proposal = torch.stack(proposal_list).reshape(-1, 4)
            # mi_iou = bbox_overlaps(mi_proposal, mi_gt, is_aligned=True).reshape(2, 36, 1)
            # max_pred_ious, max_ious_index = torch.max(mi_iou, dim=1)

            # select
            rois = bbox2roi(proposal_list)
            pred_ious, filter_roi_feat = self._iou_forward(points, 0, x, rois, pred_iou_proposal_features, img_metas, imgs_whwh)
            pred_ious_loss = self.pred_iou_head.loss(proposal_boxes, gt_bboxes, pred_ious, imgs_whwh=imgs_whwh)
            all_stage_loss['loss_pred_ious'] = pred_ious_loss['loss_pred_ious']
            max_pred_ious, max_ious_index = torch.max(pred_ious, dim=1)

            # cal gt_iou
            # iou_gt = torch.stack(gt_bboxes).repeat(1, 36, 1).reshape(-1, 4)
            # iou_proposal_list = torch.stack(proposal_list).reshape(-1, 4)
            # gt_iou = bbox_overlaps(iou_proposal_list, iou_gt, is_aligned=True)

            _of = []
            """
            select proposal_feat
            """
            # _of.append(box_proposal_features[i, max_ious_index[i][0].item(), :])
            # if input_mode == 'point':
            #     for i in range(num_imgs):
            #         _of.append(point_proposal_features[i, max_ious_index[i][0].item(), :])
            # elif input_mode == 'bbox':
            #     for i in range(num_imgs):
            #         _of.append(box_proposal_features[i, max_ious_index[i][0].item(), :])
            # object_feats = torch.stack(_of).unsqueeze(1)
            filter_roi_feat = None
            _pl = []
            for i in range(num_imgs):
                _pl.append(proposal_list[i][max_ious_index[i][0].item(), :].unsqueeze(0))
            proposal_list = _pl

            # filter_roi_feat = filter_roi_feat.reshape(2, -1, 49, 512)
            # _filter_roi_feat = []
            # for i in range(num_imgs):
            #     _filter_roi_feat.append(filter_roi_feat[i, max_ious_index[i][0].item(), :, :])
            # filter_roi_feat = torch.stack(_filter_roi_feat)

            # cal iou
            proposal_bbox_ious = torch.cat(_pl)
            gt_bboxes_iou = torch.cat(gt_bboxes)
            pro_ious = bbox_overlaps(proposal_bbox_ious, gt_bboxes_iou, is_aligned=True)
            all_stage_loss['proposal_gt_ious'] = pro_ious
            all_stage_loss['proposal_pred_ious'] = max_pred_ious.unsqueeze(1)
        else:
            proposal_list = []
            for meta in img_metas:
                h, w, _ = meta['img_shape']
                proposal_list.append(x[0].new_tensor([[0, 0, w, h]]))

        if self.custom_cfg.with_iterative_refine:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(points, stage, x, rois, object_feats,
                                                  img_metas, imgs_whwh, None)
                all_stage_bbox_results.append(bbox_results)
                # if gt_bboxes_ignore is None:
                #     gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                # cls_pred_list = bbox_results['detach_cls_score_list']
                proposal_list = bbox_results['detach_proposal_list']
                # for i in range(num_imgs):
                #     normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                #                                               imgs_whwh[i])
                #     assign_result = self.bbox_assigner[stage].assign(
                #         normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                #         gt_labels[i], img_metas[i])
                #     sampling_result = self.bbox_sampler[stage].sample(
                #         assign_result, proposal_list[i], gt_bboxes[i])
                #     sampling_results.append(sampling_result)
                # bbox_targets = self.bbox_head[stage].get_targets(
                #     sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                #     True)
                # cls_score = bbox_results['cls_score']
                decode_bbox_pred = bbox_results['decode_bbox_pred']

                # ious_pred = bbox_results['ious_pred']
                single_stage_loss = self.bbox_head[stage].loss(
                    decode_bbox_pred.view(-1, 4),
                    gt_bboxes,
                    imgs_whwh=imgs_whwh)

                for key, value in single_stage_loss.items():
                    all_stage_loss[f'stage{stage}_{key}'] = value * \
                                                            self.stage_loss_weights[stage]
                """
                fix object_feats
                """
                if not self.custom_cfg.fix_proposal:
                    object_feats = bbox_results['object_feats']

            # calculate iou
            # pred_ious_stage6 = all_stage_bbox_results[5]['ious_pred']
            # max_pred_ious, max_ious_index = torch.max(pred_ious_stage6, dim=1)
            # max_ious_index = torch.argmax(pred_ious_stage6, dim=1)
            pred_bboxes_stage6 = all_stage_bbox_results[self.num_stages - 1]['decode_bbox_pred'].reshape(self.batch_size, -1, 4)
            # bbox_ious = []
            # for i in range(max_ious_index.shape[0]):
            #     bbox_ious.append(pred_bboxes_stage6[i, max_ious_index[i, :][0], :])
            bbox_ious = pred_bboxes_stage6
            # bbox_ious = torch.stack(bbox_ious)
            # # gt_bboxes_iou = torch.stack(gt_bboxes).repeat([1, 48, 1]).reshape(-1, 4)
            gt_bboxes_iou = torch.cat(gt_bboxes)
            ious = bbox_overlaps(bbox_ious.squeeze(1), gt_bboxes_iou, is_aligned=True)
            all_stage_loss['ious'] = ious
            # all_stage_loss['pred_ious'] = max_pred_ious

            return all_stage_loss, decode_bbox_pred
        else:
            decode_bbox_pred = torch.cat(proposal_list)
            return all_stage_loss, decode_bbox_pred

    def simple_test(self, x, points, proposal_boxes, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features,
                    input_mode, img_metas, img_whwh, rescale=False):
        num_imgs = 1
        proposal_boxes = proposal_boxes[0]
        proposal_list = []
        proposal_list.append(proposal_boxes)
        points = torch.tensor([1, 2]).cuda() # dummy arg

        """
        text info
        """
        if self.custom_cfg.use_text_feat:
            text_info = []
            for img_meta in img_metas:
                text_info.append('a {}'.format(img_meta['ori_filename'].split('/')[0]))

            """
            ###### extract text using clip model
            text = clip.tokenize(text_info).cuda()
            with torch.no_grad():
                # text_feature = self.text_linear(self.clip_model.encode_text(text).float()).unsqueeze(1)
                text_feature = self.text_mlp(self.clip_model.encode_text(text).float()).unsqueeze(1)
            """
            
            # 512 dims
            text_feature = []
            for img_meta in img_metas:
                t = img_meta['ori_filename'].split('/')[0]
                # t = text_info[0].split()[1]
                text_feature.append(self.text_mlp(self.clip_feat_file[t].float().cuda()))
                # text_info.append(img_meta['ori_filename'].split('/')[0])   
            text_feature = torch.stack(text_feature)
            # text_feature = self.clip_model.encode_text(text).float().unsqueeze(1)
            object_feats = text_feature
            st1_refine_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
        # elif self.custom_cfg.use_point_feat:
        #     point_feat = self.extra_point_feat(x, points).unsqueeze(1)
        #     object_feats = point_feat
        #     st1_refine_proposal_features = point_feat.repeat(1, self.custom_cfg.num_proposals, 1)
        #     pred_iou_proposal_features = point_feat.repeat(1, self.custom_cfg.num_proposals, 1)
        else:
            point_proposal_features = point_proposal_features[:, 0, :].unsqueeze(1)
            object_feats = point_proposal_features
            st1_refine_proposal_features = point_proposal_features.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = point_proposal_features.repeat(1, self.custom_cfg.num_proposals, 1)
        # st1_refine_proposal_features = bbox_proposal_features
        # pred_iou_proposal_features = text_feature.repeat(1, 36, 1)

        if self.custom_cfg.with_prototype_selection:
            """
            st1 refine
            """
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward_st1(points, 0, x, rois, st1_refine_proposal_features, img_metas, img_whwh)
            # decode_bbox_pred = bbox_results['decode_bbox_pred']
            proposal_list = bbox_results['detach_proposal_list']

            # select
            rois = bbox2roi(proposal_list)
            pred_ious, filter_roi_feat = self._iou_forward(points, 0, x, rois, pred_iou_proposal_features, img_metas, img_whwh)
            max_pred_ious, max_ious_index = torch.max(pred_ious, dim=1)
            # _of = []
            # for i in range(num_imgs):
            #     _of.append(bbox_proposal_features[i, max_ious_index[i][0].item(), :])
            # object_feats = torch.stack(_of).unsqueeze(1)

            _pl = []
            for i in range(num_imgs):
                _pl.append(proposal_list[i][max_ious_index[i][0].item(), :].unsqueeze(0))
            proposal_list = _pl

            # filter_roi_feat = filter_roi_feat.reshape(1, -1, 49, 256)
            # _filter_roi_feat = []
            # for i in range(num_imgs):
            #     _filter_roi_feat.append(filter_roi_feat[i, max_ious_index[i][0].item(), :, :])
            # filter_roi_feat = torch.stack(_filter_roi_feat)

            # select faster st2
            # max_ious_index, proposal_list = self.pred_iou_head.simple_test(x, proposal_list, img_metas, img_whwh)
            # _of = []
            # for i in range(num_imgs):
            #     _of.append(bbox_proposal_features[i, max_ious_index[i].item(), :])
            # object_feats = torch.stack(_of).unsqueeze(1)
            # object_feats = bbox_proposal_features

            # vis select anchor
            # show anchor
            # if True:
            #     import cv2
            #     import copy
            #     import os
            #     prefix = '/home/ubuntu/kuiran/github/mmtracking-master/mmtracking-master-cache/debug/vis_select_anchor_1101_512dims'
            #     # prefix = 'debug1'
            #     # gt_vis = torch.stack(gt_bboxes_, dim=0).squeeze(1)
            #     # gt_vis_1 = np.array(gt_vis.cpu().squeeze(0)).astype(np.int32)
            #     # pred_bbox_vis = self.vis_box.squeeze(0).clone().cpu().numpy().astype(np.int32)
            #     # ppppp = random_points[0].squeeze(0).squeeze(0).clone().cpu().numpy().astype(np.int32)
            #     # vis_noisy_bbox = noisy_bbox[0].squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)
            #     file_name = img_metas[0]['filename']
            #     # frame_id = i_m['frame_id']
            #     video_name = file_name.split('/')[-3]
            #     h, w, _ = img_metas[0]['img_shape']
            #     img1 = cv2.imread(file_name)
            #     img1 = cv2.resize(img1, (w, h))
            #     igs1 = copy.deepcopy(img1)
            #     if not os.path.isdir(prefix):
            #         os.makedirs(prefix)
            #     vis_anchor = proposal_list[0].squeeze(0).cpu().numpy().astype(np.int32)
            #     igs1 = cv2.rectangle(igs1, (vis_anchor[0], vis_anchor[1]),
            #                          (vis_anchor[2], vis_anchor[3]),
            #                          color=(0, 256, 0))
            #     # igs1 = cv2.putText(igs1, str(max_pred_ious.item()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #     # for b
            #     # igs1 = cv2.rectangle(igs1, (pred_bbox_vis[0], pred_bbox_vis[1]),
            #     #                      (pred_bbox_vis[2], pred_bbox_vis[3]),
            #     #                      color=(0, 256, 0))
            #     # if self.test_cfg.test_mode == 'point':
            #     #     igs1 = cv2.circle(igs1, (ppppp[0], ppppp[1]), 2, (0, 0, 255), 4)
            #     # elif self.test_cfg.test_mode == 'bbox':
            #     #     igs1 = cv2.rectangle(igs1, (vis_noisy_bbox[0], vis_noisy_bbox[1]),
            #     #                          (vis_noisy_bbox[2], vis_noisy_bbox[3]),
            #     #                          color=(255, 255, 0))
            #     # igs1 = cv2.rectangle(igs1, (gt_vis_1[0], gt_vis_1[1]),
            #     #                      (gt_vis_1[2], gt_vis_1[3]),
            #     #                      color=(0, 0, 0))
            #     cv2.imwrite('{}/{}_{}.jpg'.format(prefix, video_name, 0), igs1)
        else:
            proposal_list = []
            for meta in img_metas:
                h, w, _ = meta['img_shape']
                proposal_list.append(x[0].new_tensor([[0, 0, w, h]]))

        if self.custom_cfg.with_iterative_refine:
            for stage in range(self.num_stages):
                rois = bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(
                    points, stage, x, rois, object_feats, img_metas, img_whwh, None
                )
                """
                fix object_feat
                """
                if not self.custom_cfg.fix_proposal:
                    object_feats = bbox_results['object_feats']
                proposal_list = bbox_results['detach_proposal_list']
                # if stage == 0:
                #     temp_pred_bbox = bbox_results['decode_bbox_pred']

            # pred_ious_stage6 = bbox_results['ious_pred']
            # max_pred_ious, max_ious_index = torch.max(pred_ious_stage6, dim=1)

            pred_bbox = bbox_results['decode_bbox_pred']
            # pred_bbox = temp_pred_bbox

            # pred_bbox = pred_bbox[max_ious_index[0][0].item(), :]
        else:
            pred_bbox = proposal_list[0]
        scale_factor = img_metas[0]['scale_factor']
        ori_pred_bbox = pred_bbox / pred_bbox.new_tensor(scale_factor)

        return dict(pred_bbox=pred_bbox, ori_pred_bbox=ori_pred_bbox)


@HEADS.register_module()
class UTIDIIIOUHead(BBoxHead):
    def __init__(self,
                 num_classes=1,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 feedforward_channels=2048,
                 in_channels=256,
                 reg_dims=4,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_pred_ious=dict(type='SmoothL1Loss', loss_weight=2.0),
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(UTIDIIIOUHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_pred_ious = build_loss(loss_pred_ious)
        self.in_channels = in_channels
        self.reg_dims = reg_dims
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)

        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        self.fc_cls = nn.Linear(in_channels, self.num_classes)
        self.fc_cls_1 = nn.Sigmoid()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(UTIDIIIOUHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """
        Args:
            roi_feat: (b*num_proposals, c, h, w)
            proposal_feat: (b, num_proposals, c)
        Returns:
            bbox_delta: (b, num_proposals, 4)
            obj_feat: (b, num_proposals, c)
        """
        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        attn_feats = proposal_feat.permute(1, 0, 2)

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic, filter_roi_feat = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        # jump
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        # !jump
        # proposal_feat = self.instance_interactive_conv_dropout(proposal_feat_iic)

        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)

        # bbox_delta = self.reg_1(self.fc_reg(reg_feat)).view(N, num_proposals, self.reg_dims)
        # kuiran
        ious_pred = self.fc_cls_1(self.fc_cls(cls_feat)).view(N, num_proposals, self.num_classes)

        return ious_pred, filter_roi_feat

    @force_fp32(apply_to=('pred_ious',))
    def loss(self,
             proposal_bbox,
             bbox_targets,
             pred_ious,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        losses = dict()
        b = proposal_bbox.shape[1]
        bbox_targets = torch.stack(bbox_targets).repeat([1, b, 1]).reshape(-1, 4)

        # caculate_ious
        pred_bboxes_stage6 = proposal_bbox.reshape(-1, 4)
        gt_bboxes_iou = bbox_targets
        gt_ious = bbox_overlaps(pred_bboxes_stage6, gt_bboxes_iou, is_aligned=True)

        pred_ious = pred_ious.reshape(-1)
        # if pred_ious is not None:
        #     if pred_ious.numel() > 0:
        losses['loss_pred_ious'] = self.loss_pred_ious(pred_ious, gt_ious)
        return losses


@HEADS.register_module()
class UTIDIIBoxHead(BBoxHead):
    def __init__(self,
                 num_classes=1,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 reg_dims=4,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_pred_ious=dict(type='SmoothL1Loss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(UTIDIIBoxHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.in_channels = in_channels
        self.reg_dims = reg_dims
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.batch_size = 8
        self.num_proposals = 12

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)

        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.reg_fcs = nn.ModuleList()
        for i in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(in_channels, reg_dims)
        self.reg_1 = nn.Sigmoid()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(UTIDIIBoxHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    def roi_feature_vis(self, roi_feat, img_metas, rois, stage):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import os
        import cv2
        roi_feature = roi_feat.permute(0, 2, 1)
        N, C, HW = roi_feature.shape
        H, W = int(HW**0.5), int(HW**0.5)
        roi_feature = roi_feature.cpu().detach().numpy()
        prefix = '/home/ubuntu/kuiran/github/mmtracking-master'
        img_file_path = img_metas[0]['filename']
        img_file_path = os.path.join(prefix, img_file_path)
        original_image = cv2.imread(img_file_path)
        save_path_prefix = '/home/ubuntu/kuiran/github/mmtracking-master/mmtracking-master-cache/restart/tcr_base_1x/tcr_r50_1x_fpn_stage2_anchor4_3_frozen_backbone_fix_proposal/1x/heatmap_vis/replace3/'
        # img_save_name = '{}_{}.jpg'.format(img_file_path.split('/')[-3])
        resize_h, resize_w = img_metas[0]['img_shape'][0], img_metas[0]['img_shape'][1]

        if N == 1:
            stage = stage + 1

        heatmap_list = []
        for i in range(N):
            vis_roi_feature = roi_feature[i, :, :].T
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(vis_roi_feature)
            # principal_component = principal_component
            # heatmap = (principal_component - np.min(principal_component)) / (np.max(principal_component) - np.min(principal_component))
            # heatmap = np.uint8(255 * heatmap)
            heatmap_list.append(principal_component)
        heatmap_list = np.array(heatmap_list)
        global_min = np.min(heatmap_list)
        global_max = np.max(heatmap_list)

        for i in range(N):
            # vis_roi_feature = roi_feature[i, :, :].T
            # pca = PCA(n_components=1)
            # principal_component = pca.fit_transform(vis_roi_feature)
            # principal_component = principal_component.reshape(H, W)
            # heatmap = (principal_component - np.min(principal_component)) / (np.max(principal_component) - np.min(principal_component))
            # heatmap = np.uint8(255 * heatmap)

            heatmap = heatmap_list[i, :, :]
            heatmap = (heatmap - global_min) / (global_max - global_min)
            heatmap = np.uint8(255 * heatmap).reshape(H, W)

            roi = rois[i, 1:]
            x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
            x1, y1, x2, y2 = min(x1, resize_w), min(y1, resize_h), min(x2, resize_w), min(y2, resize_h)
            roi_h, roi_w = y2 - y1, x2 - x1
            heatmap_resize = cv2.resize(heatmap, (roi_w, roi_h))
            heatmap_color = cv2.applyColorMap(heatmap_resize, cv2.COLORMAP_JET)
            superimposed_img = original_image.copy()
            superimposed_img = cv2.resize(superimposed_img, (resize_w, resize_h))
            superimposed_img[y1:y2, x1:x2] = cv2.addWeighted(superimposed_img[y1:y2, x1:x2], 0.5, heatmap_color, 0.5, 0)

            img_save_name = '{}_{}_{}.jpg'.format(img_file_path.split('/')[-3], stage + 1, i)
            save_path = os.path.join(save_path_prefix, img_save_name)
            cv2.imwrite(save_path, superimposed_img)
            # cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    """
    ori def forward(self, roi_feat, proposal_feat)
    """
    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, img_metas, rois, stage):
        """
        Args:
            roi_feat: (b*num_proposals, c, h, w)
            proposal_feat: (b, num_proposals, c)
        Returns:
            bbox_delta: (b, num_proposals, 4)
            obj_feat: (b, num_proposals, c)
        """
        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        attn_feats = proposal_feat.permute(1, 0, 2)

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic, filter_roi_feature = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        """
        The heatmap visualize of roi feature
        """
        # self.roi_feature_vis(filter_roi_feature, img_metas, rois, stage)

        # jump
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        # !jump
        # proposal_feat = self.instance_interactive_conv_dropout(proposal_feat_iic)

        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        reg_feat = obj_feat

        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        # bbox_delta = self.reg_1(self.fc_reg(reg_feat)).view(N, num_proposals, self.reg_dims)
        # kuiran
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, self.reg_dims)

        return bbox_delta, obj_feat.view(
            N, num_proposals, self.in_channels), attn_feats
        # return bbox_delta, attn_feats

    @force_fp32(apply_to=('bbox_pred',))
    def loss(self,
             bbox_pred,
             bbox_targets,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        losses = dict()
        # 4是batch size
        b = int(bbox_pred.shape[0] / self.batch_size)
        avg_factor = bbox_pred.shape[0]
        bbox_targets = torch.stack(bbox_targets).reshape(-1, 4).repeat_interleave(b, dim=0)
        if bbox_pred is not None:
            pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)
            imgs_whwh = imgs_whwh.repeat_interleave(b, dim=0).reshape(-1, 4)
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred / imgs_whwh,
                bbox_targets / imgs_whwh,
                avg_factor=avg_factor)
            losses['loss_iou'] = self.loss_iou(
                pos_bbox_pred,
                bbox_targets,
                avg_factor=avg_factor)
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0
            losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    @force_fp32(apply_to=('bbox_pred',))
    def loss_st1_refine(self,
             bbox_pred,
             bbox_targets,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        losses = dict()
        avg_factor = bbox_pred.shape[0]
        bbox_targets = torch.stack(bbox_targets).reshape(-1, 4)
        if bbox_pred is not None:
            pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)
            imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0), 4)
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred / imgs_whwh,
                bbox_targets / imgs_whwh,
                avg_factor=avg_factor)
            losses['loss_iou'] = self.loss_iou(
                pos_bbox_pred,
                bbox_targets,
                avg_factor=avg_factor)
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0
            losses['loss_iou'] = bbox_pred.sum() * 0
        return losses


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