import torch
import torch.nn as nn
import numpy as np

from mmdet.models.builder import DETECTORS
from mmtrack.models import MODELS
from mmdet.models.detectors import SparseRCNN
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads import SparseRoIHead
# from mmdet.models.roi_heads.bbox_heads import DIIHead
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.models.utils import build_transformer
from mmdet.core.bbox.samplers import PseudoSampler

from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.core import multi_apply


@MODELS.register_module()
class PointRegression(SparseRCNN):

    def __init__(self, *args, **kwargs):
        super(PointRegression, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert proposals is None, 'Sparse R-CNN and QueryInst ' \
                                  'do not support external proposals'

        x = self.extract_feat(img)
        gt_bboxes = [bbox.float() for bbox in gt_bboxes]
        gt_points = [(box[:, :2] + box[:, 2:]) / 2 for box in gt_bboxes]
        # proposal_boxes, proposal_features, imgs_whwh = \
        #     self.rpn_head.forward_train(x, img_metas)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas, gt_points)
        roi_losses, bbox_pred = self.roi_head.forward_train(
            x,
            gt_points,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        return roi_losses, bbox_pred

    def simple_test(self, img, gt_bboxes, img_metas, rescale=False):
        x = self.extract_feat(img)
        gt_bboxes = [bbox.float() for bbox in gt_bboxes]
        gt_points = [(box[:, :2] + box[:, 2:]) / 2 for box in gt_bboxes]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas, gt_points)
        result = self.roi_head.forward_train(
            x,
            gt_points,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes)
        return result


@HEADS.register_module()
class PointRegressionHead(SparseRoIHead):

    async def async_simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False, **kwargs):
        pass

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        pass

    def __init__(self, **kwargs):
        super(PointRegressionHead, self).__init__(**kwargs)
        self.ppp = PointFeatExtractor(16)

    def _bbox_forward(self, points, stage, x, rois, object_feats, img_metas, imgs_whwh):
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats)

        """
            proposal_list: list[tensor(1, 4)]
            4 distance
        """
        proposal_list = self.bbox_head[stage].refine_bboxes_points(
            points,
            bbox_pred,
            imgs_whwh,
            img_metas)

        """
            xywh
        """
        # xywh decode
        # proposal_list = self.bbox_head[stage].refine_bboxes(
        #     rois,
        #     rois.new_zeros(len(rois)),  # dummy arg
        #     bbox_pred.view(-1, bbox_pred.size(-1)),
        #     [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
        #     img_metas)

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

    def tensor2list(self, tensor):
        """
            tensor: (B, ...)
        Return:
            tensor: list[tensor] (B, (1, ...))
        """
        B = tensor.shape[0]
        result = []
        for i in range(B):
            result.append(tensor[i, ...].unsqueeze(0))
        return result

    def forward_train(self,
                      x,
                      points,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        feat = self.tensor2list(x[0])
        for i in range(len(points)):
            points[i] = points[i].unsqueeze(0)

        object_feats = self.ppp(feat, points, 16).squeeze(-1).permute(0, 2, 1)
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(points, stage, x, rois, object_feats,
                                              img_metas, imgs_whwh)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
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
            single_stage_loss = self.bbox_head[stage].loss(
                decode_bbox_pred.view(-1, 4),
                gt_bboxes,
                imgs_whwh=imgs_whwh)

            if self.with_mask:
                mask_results = self._mask_forward_train(
                    stage, x, bbox_results['attn_feats'], sampling_results,
                    gt_masks, self.train_cfg[stage])
                single_stage_loss['loss_mask'] = mask_results['loss_mask']

            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                                        self.stage_loss_weights[stage]
            object_feats = bbox_results['object_feats']

        # calculate iou
        from mmdet.core.bbox.iou_calculators import bbox_overlaps
        pred_bboxes_stage6 = all_stage_bbox_results[5]['decode_bbox_pred']
        gt_bboxes_iou = torch.stack(gt_bboxes, dim=0).squeeze(1)
        ious = bbox_overlaps(pred_bboxes_stage6, gt_bboxes_iou, is_aligned=True)
        all_stage_loss['ious'] = ious

        return all_stage_loss, decode_bbox_pred

    def simple_test(self,
                    points,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        num_imgs = 1
        proposal_list = proposal_boxes[0]
        ori_shape = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        proposal_list_ = []
        proposal_list_.append(proposal_list)

        object_feats = proposal_features

        # use point feat
        feat = self.tensor2list(x[0])
        for i in range(len(points)):
            points[i] = points[i].unsqueeze(0)

        object_feats = self.ppp(feat, points, 16).squeeze(-1).permute(0, 2, 1)
        if all([proposal.shape[0] == 0 for proposal in proposal_list_]):
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs
            return bbox_results

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list_)
            bbox_results = self._bbox_forward(points, stage, x, rois, object_feats,
                                              img_metas, imgs_whwh)
            object_feats = bbox_results['object_feats']
            proposal_list_ = bbox_results['detach_proposal_list']

        # det_bboxes = []
        # for img_id in range(num_imgs):
        #     bbox_pred_per_img = proposal_list_[img_id]
        #     if rescale:
        #         scale_factor = img_metas[img_id]['scale_factor']
        #         bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
        #     det_bboxes.append(
        #         torch.cat([bbox_pred_per_img], dim=1))

        # bbox_results = [
        #     bbox2result(det_bboxes[i], det_labels[i], num_classes)
        #     for i in range(num_imgs)
        # ]
        pred_bbox = bbox_results['decode_bbox_pred']
        scale_factor = img_metas[0]['scale_factor']
        ori_pred_bbox = proposal_list_[0] / proposal_list_[0].new_tensor(scale_factor)
        # results = bbox_results
        return pred_bbox, ori_pred_bbox


@HEADS.register_module()
class SinglePointDIIHead(BBoxHead):
    def __init__(self,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 reg_dims=4,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 guide_conv_cfg=dict(
                        type='GuidedConv',
                        in_channels=256,
                        feat_channels=64,
                        out_channels=256,
                        input_feat_shape=7,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 **kwargs
                 ):
        super(SinglePointDIIHead, self).__init__(
            with_cls=False,
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

        self.instance_interactive_conv = build_transformer(guide_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

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

        """
        sigmoid
        """
        self.reg_1 = nn.Sigmoid()

        # kuiran
        # self.fc_reg = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        # )

        # xywh2points
        self.transform_method = 'moment'
        self.moment_transfer = nn.Parameter(
            data=torch.zeros(2), requires_grad=True)
        self.moment_mul = 0.01

        # kuiran regress wh
        # self.fc_reg = nn.Linear(in_channels, 2)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(SinglePointDIIHead, self).init_weights()
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

    # @auto_fp16()
    # def forward(self, roi_feat, proposal_feat):
    #     N, num_proposals = proposal_feat.shape[:2]
    #     B, C, H, W = roi_feat.shape
    #     # Self attention
    #     proposal_feat = proposal_feat.permute(1, 0, 2)
    #     proposal_feat = self.attention_norm(self.attention(proposal_feat))
    #     attn_feats = proposal_feat.permute(1, 0, 2)
    #
    #     # instance interactive
    #     proposal_feat = attn_feats.reshape(-1, self.in_channels)
    #     proposal_feat_iic = self.instance_interactive_conv(
    #         proposal_feat, roi_feat).reshape(B, H, W, C).permute(0, 3, 1, 2)
    #     proposal_feat = roi_feat + self.instance_interactive_conv_dropout(proposal_feat_iic)
    #     # proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
    #     #     proposal_feat_iic)
    #     obj_feat = self.instance_interactive_conv_norm(proposal_feat.permute(0, 2, 3, 1))
    #
    #     # FFN
    #     obj_feat = self.ffn_norm(self.ffn(obj_feat))
    #
    #     reg_feat = obj_feat.permute(0, 3, 1, 2)
    #
    #     # for reg_layer in self.reg_fcs:
    #     #     reg_feat = reg_layer(reg_feat)
    #
    #     # bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, self.reg_dims)
    #     # kuiran
    #     # bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, 2)
    #     bbox_delta = self.fc_reg(reg_feat).reshape(B, -1, 2)
    #
    #     return bbox_delta, attn_feats

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        attn_feats = proposal_feat.permute(1, 0, 2)

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        reg_feat = obj_feat

        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        bbox_delta = self.reg_1(self.fc_reg(reg_feat)).view(N, num_proposals, self.reg_dims)
        # kuiran
        # bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, self.reg_dims)

        return bbox_delta, obj_feat.view(
            N, num_proposals, self.in_channels), attn_feats
        # return bbox_delta, attn_feats

    # kuiran
    def refine_bboxes_points(self, points, bbox_pred, imgs_whwh, imgs_metas):
        bbox_pred = bbox_pred * imgs_whwh
        points = torch.cat(points)
        # imgs_w
        max_shape = []
        for img_meta in imgs_metas:
            max_shape.append(torch.tensor(img_meta['img_shape']).unsqueeze(0))

        max_shape = torch.cat(max_shape)
        refine_bboxes = self.bbox_coder.decode(points, bbox_pred, max_shape)

        proposal_list = []
        for i in range(len(refine_bboxes)):
            proposal_list.append(refine_bboxes[i])
        return proposal_list

    @force_fp32(apply_to=('bbox_pred',))
    def loss(self,
             bbox_pred,
             bbox_targets,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        losses = dict()
        # bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        # pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # num_pos = pos_inds.sum().float()
        avg_factor = bbox_pred.shape[0]
        bbox_targets = torch.stack(bbox_targets, 1).squeeze(0)
        # if cls_score is not None:
        #     if cls_score.numel() > 0:
        #         losses['loss_cls'] = self.loss_cls(
        #             cls_score,
        #             labels,
        #             label_weights,
        #             avg_factor=avg_factor,
        #             reduction_override=reduction_override)
        #         losses['pos_acc'] = accuracy(cls_score[pos_inds],
        #                                      labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
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

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_first=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                          ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                          ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        else:
            raise NotImplementedError
        return bbox

    def refine_points(self, points, offset, max_shape, refine_mode='mul', clip_border=True):
        """
        points: (B, num_proposals, 2)
        offset: (B, num_proposals, 2)
        max_shape: (B, 2) [W, H]
        """
        assert refine_mode in ['mul', 'add']
        B = offset.shape[0]
        offset = offset.squeeze(1).reshape(B, -1, 2)
        if refine_mode == 'mul':
            points = points * offset.exp()
        elif refine_mode == 'add':
            points = points + max_shape.unsqueeze(1).cuda() * offset
        else:
            raise NotImplementedError

        if clip_border and max_shape is not None:
            # points[:, :, 0] = torch.clamp(points[:, :, 0], max=max_shape[:, 0].unsqueeze(1).cuda())
            # points[:, :, 1] = torch.clamp(points[:, :, 1], max=max_shape[:, 1].unsqueeze(1).cuda())

            points[:, :, 0].clamp_(points[:, :, 0], max=max_shape[:, 0].unsqueeze(1).cuda())
            points[:, :, 1].clamp_(points[:, :, 1], max=max_shape[:, 1].unsqueeze(1).cuda())
        return points
    # @force_fp32(apply_to=('points_preds', ))
    # def refine_bboxes(self, points, points_preds, img_metas):
    #     """Refine bboxes during training.
    #
    #     Args:
    #         rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
    #             and bs is the sampled RoIs per image. The first column is
    #             the image id and the next 4 columns are x1, y1, x2, y2.
    #         labels (Tensor): Shape (n*bs, ).
    #         bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
    #         pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
    #             is a gt bbox.
    #         img_metas (list[dict]): Meta info of each image.
    #
    #     Returns:
    #         list[Tensor]: Refined bboxes of each image in a mini-batch.
    #     """
    #     num_points = int(points_preds.shape[1] / 2)
    #     points1 = points.repeat(1, num_points)
    #     wh = []
    #     for img_metas_ in img_metas:
    #         h, w = img_metas_['img_shape'][:2]
    #         wh.append(torch.tensor([w, h]))
    #     wh = torch.stack(wh).repeat(1, num_points).cuda()
    #     refine_points = points1 + points_preds * wh
    #     bbox_preds = self.points2bbox(refine_points, y_first=False)
    #     bboxes_list = []
    #     for bbox_preds_ in bbox_preds:
    #         bboxes_list.append(bbox_preds_.unsqueeze(0))
    #     return bboxes_list

    # @force_fp32(apply_to=('bbox_pred', ))
    # def regress_by_class(self, points, points_pred, img_meta):
    #     """Regress the bbox for the predicted class. Used in Cascade R-CNN.
    #
    #     Args:
    #         points (Tensor): Rois from `rpn_head` or last stage
    #             `bbox_head`, has shape (num_proposals, 2) or
    #             (num_proposals, 3).
    #         points_pred (Tensor): Regression prediction of
    #             current stage `bbox_head`. When `self.reg_class_agnostic`
    #             is False, it has shape (n, num_classes * 18), otherwise
    #             it has shape (n, 18).
    #         img_meta (dict): Image meta info.
    #
    #     Returns:
    #         Tensor: Regressed bboxes, the same shape as input rois.
    #     """
    #
    #     assert points.size(1) == 2 or points.size(1) == 3, repr(points.shape)
    #
    #     # kuiran
    #     # assert bbox_pred.size(1) == 4
    #
    #     max_shape = img_meta['img_shape']
    #
    #     if rois.size(1) == 4:
    #         new_rois = self.bbox_coder.decode(
    #             rois, bbox_pred, max_shape=max_shape)
    #     else:
    #         bboxes = self.bbox_coder.decode(
    #             rois[:, 1:], bbox_pred, max_shape=max_shape)
    #         new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
    #
    #     return new_rois


def build_from_type(cfg, **kwargs):
    cls = cfg.pop('type')
    return eval(cls)(**cfg, **kwargs)


from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import build_head, build_loss
from .cpr_head import CirclePtFeatGenerator, SinglePointExtractor
import copy


class PointCircleSampler(object):
    def __init__(self, radius, start_angle=0, base_num_point=8, same_num_all_radius=False, append_center=True):
        # self.radius = radius
        self.radius = nn.Parameter(torch.FloatTensor(radius), requires_grad=True)
        self.radius.data = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8])
        self.start_angle = start_angle
        self.base_num_point = base_num_point
        self.same_num_all_radius = same_num_all_radius
        self.append_center = append_center

    def clip_point(self, chosen_pts, img_wh):
        """
        chosen_pts: (B, num_chosen, 2)
        img_wh: (B, 2)
        """
        chosen_pts[:, :, 0] = torch.clamp(chosen_pts[:, :, 0], min=0)
        chosen_pts[:, :, 0] = torch.clamp(chosen_pts[:, :, 0],
                                          max=img_wh[:, 0].unsqueeze(1).cuda())
        chosen_pts[:, :, 1] = torch.clamp(chosen_pts[:, :, 1], min=0)
        chosen_pts[:, :, 1] = torch.clamp(chosen_pts[:, :, 1],
                                          max=img_wh[:, 1].unsqueeze(1).cuda())
        return chosen_pts

    def __call__(self, centers, stride, img_metas):
        """
        Args:
            centers: Tensor, shape=(num_imgs, 2)
            stride: int
        Returns:
            sampler_pts: list[Tensor] (num_imgs, (1, num_chosen, 2))
        """
        assert len(centers.shape) == 2
        chosen_pts = []
        for i in range(self.radius.data.shape[0]):
            r = self.radius[i].item() * stride
            num_pts = self.base_num_point if self.same_num_all_radius else (self.base_num_point * (i + 1))
            angles = torch.arange(num_pts).float().to(centers.device) / num_pts * 360 + self.start_angle
            angles = angles / 360 * np.pi * 2
            anchor_pts = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)
            chosen_pts.append(anchor_pts)
        # for i in range(self.radius):
        #     r = (i + 1) * stride
        #     num_pts = self.base_num_point if self.same_num_all_radius else (self.base_num_point * (i + 1))
        #     angles = torch.arange(num_pts).float().to(centers.device) / num_pts * 360 + self.start_angle
        #     angles = angles / 360 * np.pi * 2
        #     anchor_pts = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=-1)
        #     chosen_pts.append(anchor_pts)
        chosen_pts = torch.cat(chosen_pts).unsqueeze(dim=0) + centers.reshape(-1, 1, 2)

        img_wh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            img_wh.append(torch.tensor([w, h]))
        img_wh = torch.stack(img_wh, dim=0)
        chosen_pts = self.clip_point(chosen_pts, img_wh)
        # all_chosen_pts = torch.cat([chosen_pts, centers.reshape(-1, 1, 2)], dim=1)

        # return chosen_pts, all_chosen_pts
        return chosen_pts

import torch.nn.functional as F


def grid_sample(feat, chosen_pts, align_corners):
    """
    # (B=1, num_gt_pts, num_chosen, 2)
    Args:
        feat: shape=(B, C, H, W)
        chosen_pts:  shape=(B, num_gts, num_chosen, 2)
    Returns:
    """
    if align_corners:
        # [0, w-1] -> [-1, 1]
        grid_norm_func = lambda xy, wh: 2 * xy / (wh - 1) - 1
        padding_mode = 'zeros'
    else:
        # [-0.5, w-1+0.5] -> [-1, 1]
        # x -> x' => x' = (2x+1) / w - 1
        grid_norm_func = lambda xy, wh: (2 * xy + 1) / wh - 1  # align_corners=False
        padding_mode = 'border'
    h, w = feat.shape[2:]
    WH = feat.new_tensor([w, h])
    chosen_pts = grid_norm_func(chosen_pts, WH)
    return F.grid_sample(feat, chosen_pts, align_corners=align_corners, padding_mode=padding_mode)


class PointFeatExtractor(BaseModule):
    def __init__(self, strides: tuple, align_corners=False, init_cfg=None):
        super(PointFeatExtractor, self).__init__(init_cfg)
        self.align_corners = align_corners
        self.strides = strides

    def extract_point_feat(self, feat, chosen_pts, stride):
        """
            feat: (1, C, H, W)
            chosen_pts: (1, num_chosen, 2)
            stride: float
        Return:
            point_feats: (1, num_chosen, feat_channel)
        """
        # s = chosen_pts.shape[:-2]
        chosen_pts = chosen_pts.unsqueeze(0) / stride # => (B=1, num_gt_points=1, num_chosen, 2)
        # permute(0, 2, 3, 1)[0]: (B=1, feat_c, num_gt_pts, num_chosen) => (num_gt_pts, num_chosen, feat_c)
        # points_feat = grid_sample(feat, chosen_pts, self.align_corners).permute(0, 2, 3, 1)[0]
        points_feat = grid_sample(feat, chosen_pts, self.align_corners)
        # _, num_chosen, feat_c = points_feat.shape
        # points_feat = points_feat.reshape(*s, num_chosen, feat_c)
        return points_feat

    def forward(self, feats, chosen_points, stride):
        """
            feat: list[Tensor] (k, (1, C, H, W))
            chosen_points: list[Tensor] (k, (1, num_chosen, 2))
        Return:
            points_feat: list[Tensor] (k, (1, num_chosen, feat_channel))
        """
        points_feat = [self.extract_point_feat(feat, chosen_points_, stride) for feat, chosen_points_ in
                       zip(feats, chosen_points)]
        points_feat = torch.cat(points_feat, dim=0)
        return points_feat


@HEADS.register_module()
class PointsOffsetHead(BaseModule):

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 points_sampler=dict(
                     type='PointCircleSampler',
                     radius=8,
                     start_angle=0,
                     base_num_point=8,
                     same_num_all_radius=True,
                     append_center=True
                 ),
                 points_feat_extractor=dict(
                     type='PointFeatExtractor',
                     strides=(16,),
                     align_corners=False,
                     init_cfg=None
                 ),
                 bbox_head=None,
                 strides=(16,),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(PointsOffsetHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = 1
        self.strides = strides
        if points_feat_extractor is None:
            points_feat_extractor = dict(
                type='CirclePtFeatGenerator',
                radius=3, start_angle=0, base_num_point=21,
                same_num_all_radius=True, append_center=True, num_classes=1
            )
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        self.bbox_head = build_head(bbox_head)
        self.points_sampler = build_from_type(points_sampler)
        self.points_feat_extractor = build_from_type(points_feat_extractor)

        # proposal_feat
        self.proposal_feat = nn.Embedding(1, 256)
        # point2bbox
        self.transform_method = 'moment'
        self.moment_transfer = nn.Parameter(
            data=torch.zeros(2), requires_grad=True)
        self.moment_mul = 0.01

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _bbox_forward(self, points_feat, sample_points, gt_r_points, stage, object_feats, img_metas):
        """
        points_feat: (batch_size*num_proposals, C, H, W)
        sample_points: (B, num_chosen, 2)
        gt_r_points: (B, 1, 2)
        stage: int
        object_feats: (batch_size, num_proposals, C)
        """
        bbox_head = self.bbox_head[stage]
        points_offset, obj_feats, _ = bbox_head(points_feat, object_feats)
        max_shape = []
        for img_meta in img_metas:
            h, w, _ = img_meta['img_shape']
            max_shape.append(torch.tensor([w, h]))
        max_shape = torch.stack(max_shape, dim=0)
        refine_points = self.bbox_head[stage].refine_points(
            sample_points, points_offset, max_shape, refine_mode='add', clip_border=False
        )
        # all_points = self.concate_points(refine_points, gt_r_points.unsqueeze(1))
        # all_points_set = self.get_points_set(all_points)
        # bbox_pred = self.points2bbox(all_points_set)

        all_points_set = self.get_points_set(refine_points)
        bbox_pred = self.points2bbox(all_points_set)

        result = {
            'all_points': refine_points,
            'object_feat': obj_feats,
            'bbox_pred': bbox_pred
        }
        return result

    def concate_points(self, sample_points, gt_r_points):
        """
            sample_points: (B, num_chosen, 2)
            gt_r_points: (B, 1, 2)
        Return:
            all_points: (B, num_chosen + 1, 2)
        """
        return torch.cat((sample_points, gt_r_points), dim=1)

    def get_points_set(self, all_points):
        """
            all_points: (B, num_chosen + 1, 2)
        Return:
            Tensor: (B, 2 * (num_chosen + 1))
        """
        B, _, _ = all_points.shape
        return all_points.reshape(B, -1)

    def tensor2list(self, tensor):
        """
            tensor: (B, ...)
        Return:
            tensor: list[tensor] (B, (1, ...))
        """
        B = tensor.shape[0]
        result = []
        for i in range(B):
            result.append(tensor[i, ...].unsqueeze(0))
        return result

    def forward_train(self,
                      feats,
                      gt_r_points,
                      img_metas,
                      gt_bboxes,
                      imgs_whwh=None):
        """
        Args:
            feats: list[Tensor] (B, (1, C, H, W))
            gt_r_points: (B, 2)
        """
        gt_r_points = gt_r_points.type(feats[0].type())
        imgs_whwh = imgs_whwh.repeat(1, 1, 1)
        all_stage_bbox_results = []
        all_stage_loss = {}
        all_points = self.points_sampler(gt_r_points, self.strides, img_metas)
        if self.train_cfg.vis:
            from .vis import Visualize
            visual = Visualize()
            vis_all_points = np.array(all_points.cpu()).astype(np.int32)
            pre_fix = 'exp/64points/debug/train'
            visual(img_metas, vis_all_points, pre_fix)
        # convert gt_r_points (B, 2) => list[Tensor] (B, (1, 1, 2))
        gt_r_points_ = self.tensor2list(gt_r_points)
        for i in range(len(gt_r_points_)):
            gt_r_points_[i] = gt_r_points_[i].unsqueeze(0)
        all_points_ = self.tensor2list(all_points)
        # gt_r_points_ = []
        # for i in range(len(gt_r_points.shape[0])):
        #     gt_r_points_.append(gt_r_points[i].unsqueeze(0).unsqueeze(0))

        object_feats = self.points_feat_extractor(feats, gt_r_points_, self.strides).squeeze(-1).permute(0, 2, 1)
        # object_feats = self.proposal_feat.weight.clone()
        # object_feats = object_feats.repeat(4, 1).unsqueeze(1)
        for stage in range(self.num_stages):
            points_feat = self.points_feat_extractor(feats, all_points_, self.strides)
            b, c, _, num = points_feat.shape
            points_feat = points_feat.reshape(b, c, int(num ** 0.5), -1)
            result = self._bbox_forward(points_feat, all_points, gt_r_points, stage, object_feats,
                                        img_metas)
            all_points_ = self.tensor2list(result['all_points'])
            object_feats = result['object_feat']
            bbox_pred = result['bbox_pred']
            all_stage_bbox_results.append(result)
            single_stage_loss = self.bbox_head[stage].loss(
                bbox_pred.view(-1, 4),
                gt_bboxes,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * self.stage_loss_weights[stage]
        # calculate iou
        from mmdet.core.bbox.iou_calculators import bbox_overlaps
        pred_bboxes_stage6 = all_stage_bbox_results[self.num_stages-1]['bbox_pred']
        gt_bboxes_iou = torch.stack(gt_bboxes, dim=0).squeeze(1)
        ious = bbox_overlaps(pred_bboxes_stage6, gt_bboxes_iou, is_aligned=True)
        all_stage_loss['ious'] = ious

        return all_stage_loss, bbox_pred

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_first=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].

            pts: (B, num_points)
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                          ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                          ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        else:
            raise NotImplementedError
        return bbox
