from mmdet.models.roi_heads.sparse_roi_head import SparseRoIHead
from mmdet.models.builder import HEADS
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.dense_heads.embedding_rpn_head import EmbeddingRPNHead
from mmcv.runner import BaseModule


@HEADS.register_module()
class SimpleEmbeddingRPNHeadOLD(BaseModule):
    def __init__(self,
                 num_proposals=100,
                 proposal_feature_channel=256,
                 init_cfg=None,
                 **kwargs):
        super(SimpleEmbeddingRPNHeadOLD, self).__init__(init_cfg)
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features_bboxes = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)
        self.init_proposal_features_points = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def init_weights(self):
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of  the entire
        image.
        """
        super(SimpleEmbeddingRPNHeadOLD, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def _decode_init_proposals(self, imgs, img_metas, initial_bbox=None, initial_point=None):
        # num_imgs = len(imgs[0])
        num_imgs = imgs.shape[0]
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        if initial_bbox is not None:
            proposals = initial_bbox.unsqueeze(1).float()
            init_proposal_features = self.init_proposal_features_bboxes.weight.clone()
            init_proposal_features = init_proposal_features[None].expand(
                num_imgs, *init_proposal_features.size())
        elif initial_point is not None:
            proposals = self.init_proposal_bboxes.weight.clone()
            proposals = bbox_cxcywh_to_xyxy(proposals)
            proposals = proposals * imgs_whwh
            init_proposal_features = self.init_proposal_features_points.weight.clone()
            init_proposal_features = init_proposal_features[None].expand(
                num_imgs, *init_proposal_features.size())
        return proposals, init_proposal_features, imgs_whwh

    def forward_train(self, img, img_metas, initial_bbox=None, initial_point=None):
        """Forward function in training stage."""
        return self._decode_init_proposals(img, img_metas, initial_bbox=initial_bbox, initial_point=initial_point)

    def simple_test_rpn(self, img, img_metas, initial_bbox=None, initial_point=None):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas, initial_bbox=initial_bbox, initial_point=initial_point)


@HEADS.register_module()
class SimpleRefineRoIHeadOLD(SparseRoIHead):
    def __init__(self, *args, **kwargs):
        super(SimpleRefineRoIHeadOLD, self).__init__(*args, **kwargs)

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas, imgs_whwh, points=None):
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats, points=points)

        if points is None:
            proposal_list = self.bbox_head[stage].refine_bboxes(
                rois,
                rois.new_zeros(len(rois)),  # dummy arg
                bbox_pred.view(-1, bbox_pred.size(-1)),
                [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
                img_metas)
        else:
            proposal_list = self.bbox_head[stage].refine_bboxes_points(
                points,
                bbox_pred,
                imgs_whwh,
                img_metas)
        bbox_results = dict(
            # cls_score=cls_score,
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            # detach_cls_score_list=[
            #     cls_score[i].detach() for i in range(num_imgs)
            # ],
            detach_proposal_list=[item.detach().float() for item in proposal_list])

        return bbox_results

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      points=None,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas, imgs_whwh, points=points)
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
        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    points=None,
                    rescale=False):
        num_imgs = 1
        # proposal_boxes = proposal_boxes[0]
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        object_feats = proposal_features
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas, imgs_whwh, points=points)
            object_feats = bbox_results['object_feats']
            proposal_list = bbox_results['detach_proposal_list']
        pred_bbox = bbox_results['decode_bbox_pred']
        scale_factor = img_metas[0]['scale_factor']
        ori_pred_bbox = pred_bbox / pred_bbox.new_tensor(scale_factor)

        return dict(pred_bbox=pred_bbox, ori_pred_bbox=ori_pred_bbox)


from mmtrack.models.uti.uti1 import UTIDIIBoxHead
from mmdet.core import build_bbox_coder


@HEADS.register_module()
class SimpleDIIHeadOLD(UTIDIIBoxHead):
    def __init__(self, bbox_coder_point=None, *args, **kwargs):
        super(SimpleDIIHeadOLD, self).__init__(*args, **kwargs)
        self.bbox_coder_point = build_bbox_coder(bbox_coder_point)

    def refine_bboxes_points(self, points, bbox_pred, imgs_whwh, imgs_metas):
        bbox_pred = bbox_pred * imgs_whwh
        # points = torch.cat(points)
        # imgs_w
        max_shape = []
        for img_meta in imgs_metas:
            max_shape.append(torch.tensor(img_meta['img_shape']).unsqueeze(0))

        max_shape = torch.cat(max_shape)
        refine_bboxes = self.bbox_coder_point.decode(points, bbox_pred, max_shape)

        proposal_list = []
        for i in range(len(refine_bboxes)):
            proposal_list.append(refine_bboxes[i])
        return proposal_list

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, points=None):
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

        # bbox_delta = self.reg_1(self.fc_reg(reg_feat)).view(N, num_proposals, self.reg_dims)
        # kuiran
        if points == None:
            bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, self.reg_dims)
        else:
            bbox_delta = self.reg_1(self.fc_reg(reg_feat)).view(N, num_proposals, self.reg_dims)

        return bbox_delta, obj_feat.view(
            N, num_proposals, self.in_channels), attn_feats
        # return bbox_delta, attn_feats


from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.builder import build_head, build_roi_extractor
from mmdet.core import build_bbox_coder


@HEADS.register_module()
class SimpleFasterST2OLD(BaseModule):
    def __init__(self, bbox_roi_extractor, bbox_head, point2box_coder):
        super(SimpleFasterST2OLD, self).__init__()
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        self.point2box_coder = build_bbox_coder(point2box_coder)
        self.point_reg = nn.Sigmoid()

    def forward_train(self, x, proposal_list, gt_bboxes, img_metas, imgs_whwh, points=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        _, bbox_pred = self.bbox_head(bbox_feats)
        all_loss = {}
        if points is not None:
            loss_bbox = self.loss(
                bbox_pred,
                rois,
                gt_bboxes,
                imgs_whwh,
                img_metas,
                points=points
            )
        else:
            loss_bbox = self.loss(
                bbox_pred,
                rois,
                gt_bboxes,
                imgs_whwh,
                img_metas,
                points=None
            )
        all_loss.update(loss_bbox)
        return all_loss

    def simple_test(self, x, proposal_list, img_metas, imgs_whwh, points=None):
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        _, bbox_pred = self.bbox_head(bbox_feats)
        if points is not None:
            bbox_pred, ori_bbox = self.refine_bbox(bbox_pred, rois=rois, points=points,
                                                   imgs_whwh=imgs_whwh, img_metas=img_metas)
            bbox_pred = bbox_pred.squeeze(0)
            ori_bbox = ori_bbox.squeeze(0)
        else:
            bbox_pred, ori_bbox = self.refine_bbox(bbox_pred, rois=rois, points=points,
                                                   imgs_whwh=imgs_whwh, img_metas=img_metas)
        result = {}
        result['pred_bbox'] = bbox_pred
        result['ori_pred_bbox'] = ori_bbox
        return result

    def refine_bbox(self, bbox_pred, rois=None, points=None, imgs_whwh=None, img_metas=None):
        if points is not None:
            bbox_pred = self.point_reg(bbox_pred)
            bbox_pred = bbox_pred * imgs_whwh
            decode_bbox_pred = self.point2box_coder.decode(points, bbox_pred)
        else:
            rois = rois[:, 1:]
            decode_bbox_pred = self.bbox_head.bbox_coder.decode(rois, bbox_pred)
        scale_factor = img_metas[0]['scale_factor']
        ori_pred_bbox = decode_bbox_pred / decode_bbox_pred.new_tensor(scale_factor)
        return decode_bbox_pred, ori_pred_bbox

    @force_fp32(apply_to=('bbox_pred'))
    def loss(self,
             bbox_pred,
             rois,
             bbox_targets,
             imgs_whwh,
             imgs_metas,
             points=None,
             reduction_override=None):
        losses = dict()
        # bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
        bbox_targets = torch.cat(bbox_targets)
        rois = rois[:, 1:]
        if points is not None:
            # bbox_pred = bbox_pred * imgs_whwh
            # points = torch.cat(points)
            # imgs_w
            bbox_pred = self.point_reg(bbox_pred)
            max_shape = []
            for img_meta in imgs_metas:
                max_shape.append(torch.tensor(img_meta['img_shape']).unsqueeze(0))
            points = points.squeeze(1)
            imgs_whwh = imgs_whwh.squeeze(1)
            encode_bbox_targets = self.point2box_coder.encode(points, bbox_targets)
            losses['loss_bbox'] = self.bbox_head.loss_bbox(
                bbox_pred,
                encode_bbox_targets / imgs_whwh,
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override
            )
            bbox_pred = bbox_pred * imgs_whwh
            # points = points.unsqueeze(1)
            decode_bbox_pred = self.point2box_coder.decode(points, bbox_pred)
        else:
            encode_bbox_targets = self.bbox_head.bbox_coder.encode(rois, bbox_targets)
            losses['loss_bbox'] = self.bbox_head.loss_bbox(
                bbox_pred,
                encode_bbox_targets,
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            decode_bbox_pred = self.bbox_head.bbox_coder.decode(rois, bbox_pred)

        # calculate ious
        from mmdet.core.bbox.iou_calculators import bbox_overlaps
        pred_bboxes_stage6 = decode_bbox_pred
        gt_bboxes_iou = bbox_targets
        ious = bbox_overlaps(pred_bboxes_stage6, gt_bboxes_iou, is_aligned=True)
        losses['ious'] = ious
        return losses


from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import build_loss


@HEADS.register_module()
class SimpleFasterST2_with_pred_iouOLD(BaseModule):
    def __init__(self, bbox_roi_extractor, bbox_head, iou_head, bbox_coder, iouloss):
        super(SimpleFasterST2_with_pred_iouOLD, self).__init__()
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        self.iou_head = build_head(iou_head)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.iouloss = build_loss(iouloss)
        self.iou_pred_head = nn.Sigmoid()
        # self.point_reg = nn.Sigmoid()

    def forward(self, x, proposal_list, gt_bboxes, img_metas, imgs_whwh):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        _, bbox_pred = self.bbox_head(bbox_feats)
        all_loss = {}
        redine_bbox, loss_bbox = self.loss_bbox(
            bbox_pred,
            rois,
            gt_bboxes
        )
        refine_bbox = [redine_bbox[:36, :], redine_bbox[36:, :]]
        rois_1 = bbox2roi(refine_bbox)
        bbox_feats_1 = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois_1)
        iou_pred, _ = self.iou_head(bbox_feats_1)
        iou_pred = self.iou_pred_head(iou_pred)
        loss_iou, max_ious_index, proposal_bbox = self.loss_iou(
            redine_bbox,
            gt_bboxes,
            iou_pred
        )
        # max_pred_ious, max_ious_index = torch.max(iou_pred, dim=1)

        all_loss.update(loss_bbox)
        all_loss.update(loss_iou)
        # select max iou box
        # a = []
        # for i in range(len(gt_bboxes)):
        #     a.append(bbox_pred[])
        # refine_bbox_pred = bbox_pred[]
        return proposal_bbox, max_ious_index, all_loss

    def simple_test(self, x, proposal_list, img_metas, imgs_whwh):
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        _, bbox_pred = self.bbox_head(bbox_feats)
        refine_bbox = self.refine_bbox(bbox_pred, rois)
        rois_1 = bbox2roi(refine_bbox)
        bbox_feats_1 = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois_1
        )
        iou_pred, _ = self.iou_head(bbox_feats_1)
        iou_pred = self.iou_pred_head(iou_pred)
        max_ious_index, proposal_list = self.select_proposals(iou_pred, refine_bbox)
        return max_ious_index, proposal_list

    # def refine_bbox(self, bbox_pred, rois=None, points=None, imgs_whwh=None, img_metas=None):
    #     if points is not None:
    #         bbox_pred = self.point_reg(bbox_pred)
    #         bbox_pred = bbox_pred * imgs_whwh
    #         decode_bbox_pred = self.point2box_coder.decode(points, bbox_pred)
    #     else:
    #         rois = rois[:, 1:]
    #         decode_bbox_pred = self.bbox_head.bbox_coder.decode(rois, bbox_pred)
    #     scale_factor = img_metas[0]['scale_factor']
    #     ori_pred_bbox = decode_bbox_pred / decode_bbox_pred.new_tensor(scale_factor)
    #     return decode_bbox_pred, ori_pred_bbox
    def select_proposals(self, iou_pred, refine_bbox):
        iou_pred = iou_pred.squeeze(1)
        iou_pred_ = iou_pred.reshape(1, -1)
        max_pred_ious, max_ious_index = torch.max(iou_pred_, dim=1)
        bbox = []
        # bbox_pred = refine_bbox.reshape(1, -1, 4)
        # for i in range(bbox_pred.shape[0]):
        #     bbox.append(bbox_pred[i, max_ious_index[i], :].unsqueeze(0))
        bbox.append(refine_bbox[0][max_ious_index[0], :].unsqueeze(0))
        return max_ious_index, bbox

    def refine_bbox(self, bbox_pred, rois):
        rois = rois[:, 1:]
        decode_bbox_pred = self.bbox_head.bbox_coder.decode(rois, bbox_pred)
        bbox = [decode_bbox_pred]
        return bbox

    # @force_fp32(apply_to=('bbox_pred'))
    # def loss(self,
    #          bbox_pred,
    #          rois,
    #          bbox_targets,
    #          imgs_whwh,
    #          imgs_metas,
    #          points=None,
    #          reduction_override=None):
    #     pass
        # losses = dict()
        # # bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
        # bbox_targets = torch.cat(bbox_targets)
        # rois = rois[:, 1:]
        # if points is not None:
        #     # bbox_pred = bbox_pred * imgs_whwh
        #     # points = torch.cat(points)
        #     # imgs_w
        #     bbox_pred = self.point_reg(bbox_pred)
        #     max_shape = []
        #     for img_meta in imgs_metas:
        #         max_shape.append(torch.tensor(img_meta['img_shape']).unsqueeze(0))
        #     points = points.squeeze(1)
        #     imgs_whwh = imgs_whwh.squeeze(1)
        #     encode_bbox_targets = self.point2box_coder.encode(points, bbox_targets)
        #     losses['loss_bbox'] = self.bbox_head.loss_bbox(
        #         bbox_pred,
        #         encode_bbox_targets / imgs_whwh,
        #         avg_factor=bbox_targets.size(0),
        #         reduction_override=reduction_override
        #     )
        #     bbox_pred = bbox_pred * imgs_whwh
        #     # points = points.unsqueeze(1)
        #     decode_bbox_pred = self.point2box_coder.decode(points, bbox_pred)
        # else:
        #     encode_bbox_targets = self.bbox_head.bbox_coder.encode(rois, bbox_targets)
        #     losses['loss_bbox'] = self.bbox_head.loss_bbox(
        #         bbox_pred,
        #         encode_bbox_targets,
        #         avg_factor=bbox_targets.size(0),
        #         reduction_override=reduction_override)
        #     decode_bbox_pred = self.bbox_head.bbox_coder.decode(rois, bbox_pred)
        #
        # # calculate ious
        # from mmdet.core.bbox.iou_calculators import bbox_overlaps
        # pred_bboxes_stage6 = decode_bbox_pred
        # gt_bboxes_iou = bbox_targets
        # ious = bbox_overlaps(pred_bboxes_stage6, gt_bboxes_iou, is_aligned=True)
        # losses['ious'] = ious
        # return losses

    @force_fp32(apply_to=('bbox_pred'))
    def loss_bbox(self,
             bbox_pred,
             rois,
             gt_bboxes,
             reduction_override=None):
        bbox_targets = torch.cat(gt_bboxes)
        bbox_targets = bbox_targets.repeat_interleave(36, dim=0)
        rois = rois[:, 1:]
        losses = dict()
        encode_bbox_targets = self.bbox_head.bbox_coder.encode(rois, bbox_targets)
        losses['loss_bbox'] = self.bbox_head.loss_bbox(
            bbox_pred,
            encode_bbox_targets,
            avg_factor=bbox_targets.size(0),
            reduction_override=reduction_override
        )
        decode_bbox_pred = self.bbox_head.bbox_coder.decode(rois, bbox_pred)
        return decode_bbox_pred, losses

    @force_fp32(apply_to=('iou_pred'))
    def loss_iou(self,
                 bbox_pred,
                 gt_bboxes,
                 iou_pred):
        losses = {}
        iou_pred = iou_pred.squeeze(1)
        # calculate ious
        bbox_targets = torch.cat(gt_bboxes)
        bbox_targets = bbox_targets.repeat_interleave(36, dim=0)
        #
        gt_ious = bbox_overlaps(bbox_pred, bbox_targets, is_aligned=True)
        # gt_ious = calculate_box_ii(bbox_pred, bbox_targets)
        losses['pred_ious_loss'] = self.iouloss(iou_pred, gt_ious)

        iou_pred_ = iou_pred.reshape(2, -1)
        max_pred_ious, max_ious_index = torch.max(iou_pred_, dim=1)
        gt_ious_ = gt_ious.reshape(2, -1)
        gtiou = []
        bbox = []
        bbox_pred = bbox_pred.reshape(2, -1, 4)
        for i in range(len(gt_bboxes)):
            gtiou.append(gt_ious_[i, max_ious_index[i]])
            bbox.append(bbox_pred[i, max_ious_index[i], :].unsqueeze(0))
        gtiou = torch.stack(gtiou)
        # gt_iou_ = gt_ious_[max_ious_index[0].item()].item()
        losses['gt_iou'] = gtiou
        losses['max_pred_iou'] = max_pred_ious
        return losses, max_ious_index, bbox


# def fp16_clamp(x, min=None, max=None):
#     if not x.is_cuda and x.dtype == torch.float16:
#         # clamp for cpu float16, tensor fp16 has no clamp implementation
#         return x.float().clamp(min, max).half()
#
#     return x.clamp(min, max)
#
#
# def calculate_box_ii(bboxes1, bboxes2):
#     area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
#             bboxes1[..., 3] - bboxes1[..., 1])
#     area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
#             bboxes2[..., 3] - bboxes2[..., 1])
#     lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
#     rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]
#
#     wh = fp16_clamp(rb - lt, min=0)
#     overlap = wh[..., 0] * wh[..., 1]
#
#     i_box1 = overlap / area1
#     i_box2 = overlap / area2
#
#     return i_box1 * i_box2

