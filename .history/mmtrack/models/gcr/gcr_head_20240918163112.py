from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.core import bbox2roi
from . import UnifiedTrackersInputRoIHead
from mmdet.models.builder import build_backbone, build_head, build_neck


@HEADS.register_module()
class GCR_head(CascadeRoIHead):
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
        self.text_mlp = MLP(input_dim=512, hidden_dim=256, output_dim=256, num_layers=4)
        self.batch_size = 8

        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")

        # box_refine
        if box_refinest1 is not None:
            self.box_refinest1 = build_head(box_refinest1)

        self.custom_cfg = custom_cfg

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
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        rois = rois.float()
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        if self.custom_cfg.roi_connect:
            bbox_pred = bbox_head(bbox_feats, object_feats)
            object_feats, attn_feats = None, None
        else:
            bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats, img_metas, rois, stage)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(bbox_pred.size(1)) for _ in range(num_imgs)],
            img_metas)

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

        if self.custom_cfg.roi_connect:
            bbox_pred = bbox_head(bbox_feats, object_feats)
            object_feats, attn_feats = None, None
        else:
            bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats, img_metas, rois, stage)
        # xywh decode
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(bbox_pred.size(1)) for _ in range(num_imgs)],
            img_metas)

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
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        sample_points = []
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
            object_feats = text_feature
            st1_refine_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
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

            # select
            rois = bbox2roi(proposal_list)
            pred_ious, filter_roi_feat = self._iou_forward(points, 0, x, rois, pred_iou_proposal_features, img_metas, imgs_whwh)
            pred_ious_loss = self.pred_iou_head.loss(proposal_boxes, gt_bboxes, pred_ious, imgs_whwh=imgs_whwh)
            all_stage_loss['loss_pred_ious'] = pred_ious_loss['loss_pred_ious']
            max_pred_ious, max_ious_index = torch.max(pred_ious, dim=1)

            _of = []
            """
            select proposal_feat
            """
            filter_roi_feat = None
            _pl = []
            for i in range(num_imgs):
                _pl.append(proposal_list[i][max_ious_index[i][0].item(), :].unsqueeze(0))
            proposal_list = _pl

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
                sampling_results = []
                proposal_list = bbox_results['detach_proposal_list']
                decode_bbox_pred = bbox_results['decode_bbox_pred']

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
            pred_bboxes_stage6 = all_stage_bbox_results[self.num_stages - 1]['decode_bbox_pred'].reshape(self.batch_size, -1, 4)
            bbox_ious = pred_bboxes_stage6
            gt_bboxes_iou = torch.cat(gt_bboxes)
            ious = bbox_overlaps(bbox_ious.squeeze(1), gt_bboxes_iou, is_aligned=True)
            all_stage_loss['ious'] = ious

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

            text = clip.tokenize(text_info).cuda()
            with torch.no_grad():
                text_feature = self.text_mlp(self.clip_model.encode_text(text).float()).unsqueeze(1)
            
            object_feats = text_feature
            st1_refine_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
            pred_iou_proposal_features = text_feature.repeat(1, self.custom_cfg.num_proposals, 1)
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
            bbox_results = self._bbox_forward_st1(points, 0, x, rois, st1_refine_proposal_features, img_metas, img_whwh)
            # decode_bbox_pred = bbox_results['decode_bbox_pred']
            proposal_list = bbox_results['detach_proposal_list']

            # select
            rois = bbox2roi(proposal_list)
            pred_ious, filter_roi_feat = self._iou_forward(points, 0, x, rois, pred_iou_proposal_features, img_metas, img_whwh)
            max_pred_ious, max_ious_index = torch.max(pred_ious, dim=1)

            _pl = []
            for i in range(num_imgs):
                _pl.append(proposal_list[i][max_ious_index[i][0].item(), :].unsqueeze(0))
            proposal_list = _pl
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


import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple
from torch.nn.init import normal_

from mmdet.models.utils.builder import TRANSFORMER

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


@TRANSFORMER.register_module()
class GuidedConv(BaseModule):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 with_proj=True,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(GuidedConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        # self.dynamic_layer = nn.Linear(
        #     512, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        if self.with_proj:
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        input_feature = input_feature.flatten(2).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        filter_roi_features = features
        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features, filter_roi_features
