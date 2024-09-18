from ..builder import MODELS
from .stark import Stark
import random
import torch
import numpy as np
from copy import deepcopy
from addict import Dict
import torch.nn.functional as F

from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.models.builder import build_backbone, build_head, build_neck, build_roi_extractor
from torchvision.transforms.functional import normalize
import math


@MODELS.register_module()
class Base_refine(Stark):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 head=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Base_refine, self).__init__(backbone,
                                          neck,
                                          head,
                                          init_cfg,
                                          frozen_modules,
                                          train_cfg,
                                          test_cfg)
        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)
        if roi_head is not None:
            self.roi_head = build_head(roi_head)

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

            a = w / 5
            b = h / 5

            if a == 0:
                random_point_x = 0.
                random_point_y = 0.
            else:
                _random_x = random.uniform(0, a)
                t_b_x = random.randint(0, 1)
                t_b_y = random.randint(0, 1)
                _random_y = math.sqrt(b**2 - ((b**2) * (_random_x**2)) / a**2)

                if t_b_y == 0:
                    random_y = -_random_y
                elif t_b_y == 1:
                    random_y = _random_y

                if t_b_x == 0:
                    random_x = -_random_x
                elif t_b_x == 1:
                    random_x = _random_x

                random_point_x = round((cx + random_x), 2)
                random_point_y = round((cy + random_y), 2)
            random_points.append(torch.tensor([[random_point_x, random_point_y]]))
        random_points = torch.cat(random_points).unsqueeze(1)
        return random_points

    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   points,
    #                   **kwargs):
    #     """
    #         Args:
    #     """
    #     x = self.extract_feat(img)
    #     gt_bboxes = [bbox.float() for bbox in gt_bboxes]
    #
    #     gt_points = [_ for _ in points.float().squeeze(1)]
    #     # decide point or box
    #     point_or_bbox = random.randint(0, 1)
    #     # print(point_or_bbox)
    #     # point_or_bbox = 0
    #     if point_or_bbox == 0:
    #         points = self.gen_points(gt_bboxes).cuda()
    #         proposal_boxes, proposal_features, imgs_whwh = self.rpn_head.forward_train(img, img_metas, initial_point=points)
    #         roi_losses = self.roi_head.forward_train(
    #             x,
    #             proposal_boxes,
    #             proposal_features,
    #             img_metas,
    #             gt_bboxes,
    #             points=points,
    #             imgs_whwh=imgs_whwh
    #         )
    #     #     proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
    #     #         self.rpn_head.forward_train(x, img_metas, points=points)
    #     #     gt_points = [_ for _ in points.float().squeeze(1)]
    #     #     input_mode = 'point'
    #     elif point_or_bbox == 1:
    #         noisy_bboxes = self.gen_noisy_bbox(gt_bboxes).cuda()
    #
    #     # proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
    #     #     self.rpn_head.forward_train(x, img_metas, bbox=noisy_bboxes)
    #         proposal_boxes, proposal_features, imgs_whwh = self.rpn_head.forward_train(img, img_metas, initial_bbox=noisy_bboxes)
    #         roi_losses = self.roi_head.forward_train(
    #             x,
    #             proposal_boxes,
    #             proposal_features,
    #             img_metas,
    #             gt_bboxes,
    #             points=None,
    #             imgs_whwh=imgs_whwh
    #         )
    #     # input_mode = 'bbox'
    #
    #     return roi_losses

    def get_allimg_bbox(self, img_metas):
        pass

    # forward_train for faster st2
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      **kwargs):
        """
            Args:
        """
        x = self.extract_feat(img)
        gt_bboxes = [bbox.float() for bbox in gt_bboxes]
        # decide point or box
        point_or_bbox = random.randint(0, 1)
        # print(point_or_bbox)
        # point_or_bbox = 0
        points = None
        if point_or_bbox == 0:
            points = self.gen_points(gt_bboxes).cuda()
            proposal_boxes, _, imgs_whwh = self.rpn_head.forward_train(img, img_metas, initial_point=points)
        elif point_or_bbox == 1:
            noisy_bboxes = self.gen_noisy_bbox(gt_bboxes).cuda()
            proposal_boxes, _, imgs_whwh = self.rpn_head.forward_train(img, img_metas, initial_bbox=noisy_bboxes)
        # proposal_boxes = noisy_bboxes
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_list,
            gt_bboxes,
            img_metas,
            imgs_whwh,
            points=points
        )
        # input_mode = 'bbox'

        return roi_losses

    def get_cropped_img(self, img, target_bbox, search_area_factor,
                        output_size):
        """ Crop Image
        Only used during testing
        This function mainly contains two steps:
        1. Crop `img` based on target_bbox and search_area_factor. If the
        cropped image/mask is out of boundary of `img`, use 0 to pad.
        2. Resize the cropped image/mask to `output_size`.

        args:
            img (Tensor): of shape (1, C, H, W)
            target_bbox (list | ndarray): in [cx, cy, w, h] format
            search_area_factor (float): Ratio of crop size to target size
            output_size (float): the size of output cropped image
                (always square).
        returns:
            img_crop_padded (Tensor): of shape (1, C, output_size, output_size)
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
            pdding_mask (Tensor): the padding mask caused by cropping. It's
                of shape (1, output_size, output_size).
        """
        cx, cy, w, h = target_bbox.split((1, 1, 1, 1), dim=-1)

        img_h, img_w = img.shape[2:]
        # 1. Crop image
        # 1.1 calculate crop size and pad size
        crop_size = math.ceil(math.sqrt(w * h) * search_area_factor)
        # if crop_size < 1:
        #     raise Exception('Too small bounding box.')

        x1 = torch.round(cx - crop_size * 0.5).long()
        x2 = x1 + crop_size
        y1 = torch.round(cy - crop_size * 0.5).long()
        y2 = y1 + crop_size

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - img_w + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - img_h + 1, 0)

        # 1.2 crop image
        img_crop = img[..., y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # 1.3 pad image
        img_crop_padded = F.pad(
            img_crop,
            pad=(x1_pad, x2_pad, y1_pad, y2_pad),
            mode='constant',
            value=0)
        # 1.4 generate padding mask
        _, _, img_h, img_w = img_crop_padded.shape
        end_x = None if x2_pad == 0 else -x2_pad
        end_y = None if y2_pad == 0 else -y2_pad
        padding_mask = torch.ones((img_h, img_w),
                                  dtype=torch.float32,
                                  device=img.device)
        padding_mask[y1_pad:end_y, x1_pad:end_x] = 0.

        # 2. Resize cropped image and padding mask
        resize_factor = output_size / crop_size
        img_crop_padded = F.interpolate(
            img_crop_padded, (output_size, output_size),
            mode='bilinear',
            align_corners=False)

        padding_mask = F.interpolate(
            padding_mask[None, None], (output_size, output_size),
            mode='bilinear',
            align_corners=False).squeeze(dim=0).type(torch.bool)

        crop_bbox = torch.tensor([x1 + x1_pad, y1 + y1_pad, x2 - x2_pad, y2 - y2_pad],
                                 dtype=torch.float32,
                                 device=img.device)
        return img_crop_padded, resize_factor, padding_mask, crop_bbox

    # def init_point_bbox(self, img, point, bbox, img_metas):
    #     self.memo = Dict()
    #     img_ = normalize(
    #         img.squeeze() / 255.,
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]).unsqueeze(0)
    #     self.z_dict_list = []
    #     with torch.no_grad():
    #         x = self.extract_feat(img_)
    #     bbox = bbox[0].squeeze(0).float()
    #     point = point[0].unsqueeze(0)
    #     if self.test_cfg.test_mode == 'point':
    #         proposal_boxes, proposal_features, imgs_whwh = \
    #             self.rpn_head.forward_train(img, img_metas, initial_point=point)
    #         result = self.roi_head.simple_test(
    #             x,
    #             proposal_boxes,
    #             proposal_features,
    #             img_metas,
    #             imgs_whwh,
    #             points=point
    #         )
    #     elif self.test_cfg.test_mode == 'bbox':
    #         proposal_boxes, proposal_features, imgs_whwh = \
    #             self.rpn_head.forward_train(img, img_metas, initial_bbox=bbox)
    #         result = self.roi_head.simple_test(
    #             x,
    #             proposal_boxes,
    #             proposal_features,
    #             img_metas,
    #             imgs_whwh,
    #         )
    #     else:
    #         raise NotImplementedError
    #
    #
    #     pred_bbox, ori_pred_bbox = result['pred_bbox'], result['ori_pred_bbox']
    #     self.memo.bbox = bbox_xyxy_to_cxcywh(ori_pred_bbox).squeeze(0)
    #     self.vis_box = pred_bbox
    #     crop_bbox = bbox_xyxy_to_cxcywh(pred_bbox)
    #     crop_img = img
    #     z_patch, _, z_mask, _ = self.get_cropped_img(crop_img, crop_bbox,
    #                                                  self.test_cfg['template_factor'],
    #                                                  self.test_cfg['template_size'])
    #     z_patch = normalize(
    #         z_patch.squeeze() / 255.,
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]).unsqueeze(0)
    #     with torch.no_grad():
    #         z_feat = self.extract_feat(z_patch)
    #     self.z_dict = dict(feat=z_feat, mask=z_mask)
    #     self.z_dict_list.append(self.z_dict)
    #     for _ in range(self.num_extra_template):
    #         self.z_dict_list.append(deepcopy(self.z_dict))

    def init_point_bbox(self, img, point, bbox, img_metas):
        self.memo = Dict()
        img_ = normalize(
            img.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).unsqueeze(0)
        self.z_dict_list = []
        with torch.no_grad():
            x = self.extract_feat(img_)
        bbox = bbox[0].squeeze(0).float()
        if self.test_cfg.test_mode == 'point':
            point = point[0].unsqueeze(0)
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.forward_train(img, img_metas, initial_point=point)
        elif self.test_cfg.test_mode == 'bbox':
            point = None
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.forward_train(img, img_metas, initial_bbox=bbox)
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        result = self.roi_head.simple_test(
            x,
            proposal_list,
            img_metas,
            imgs_whwh,
            points=point
        )

        pred_bbox, ori_pred_bbox = result['pred_bbox'], result['ori_pred_bbox']
        self.memo.bbox = bbox_xyxy_to_cxcywh(ori_pred_bbox).squeeze(0)
        self.vis_box = pred_bbox
        crop_bbox = bbox_xyxy_to_cxcywh(pred_bbox)
        crop_img = img
        z_patch, _, z_mask, _ = self.get_cropped_img(crop_img, crop_bbox,
                                                     self.test_cfg['template_factor'],
                                                     self.test_cfg['template_size'])
        z_patch = normalize(
            z_patch.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).unsqueeze(0)
        with torch.no_grad():
            z_feat = self.extract_feat(z_patch)
        self.z_dict = dict(feat=z_feat, mask=z_mask)
        self.z_dict_list.append(self.z_dict)
        for _ in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict))

    def track(self, img, bbox):
        """Track the box `bbox` of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | Tensor): The bbox in previous frame. The shape of the
                bbox is (4, ) in [x, y, w, h] format.

        Returns:
        """
        H, W = img.shape[2:]
        # get the t-th search region
        x_patch, resize_factor, x_mask, _ = self.get_cropped_img(
            img, bbox, self.test_cfg['search_factor'],
            self.test_cfg['search_size']
        )  # bbox: of shape (x1, y1, w, h), x_mask: of shape (1, h, w)
        x_patch = normalize(
            x_patch.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).unsqueeze(0)

        with torch.no_grad():
            x_feat = self.extract_feat(x_patch)
            x_dict = dict(feat=x_feat, mask=x_mask)
            head_inputs = self.z_dict_list + [x_dict]
            # run the transformer
            track_results = self.head(head_inputs)

        final_bbox = self.mapping_bbox_back(track_results['pred_bboxes'],
                                            self.memo.bbox, resize_factor)
        final_bbox = self._bbox_clip(final_bbox, H, W, margin=10)

        conf_score = -1.
        if self.head.cls_head is not None:
            # get confidence score (whether the search region is reliable)
            conf_score = track_results['pred_logits'].view(-1).sigmoid().item()
            crop_bbox = bbox_xyxy_to_cxcywh(final_bbox)
            self.update_template(img, crop_bbox, conf_score)

        return conf_score, final_bbox

    def simple_test(self, img, img_metas, gt_bboxes, points, noisy_bbox, *args, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): input image of shape (1, C, H, W).
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
            gt_bboxes (list[Tensor]): list of ground truth bboxes for each
                image with shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            dict(str : ndarray): the tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        assert len(img) == 1, 'only support batch_size=1 when testing'
        self.frame_id = frame_id

        gt_bboxes_ = []
        # for bbox in gt_bboxes:
        #     gt_bboxes_.append(bbox[0][1:].unsqueeze(0))
        #     gt_bboxes_.append(bbox[1][1:].unsqueeze(0))
        gt_bboxes_.append(gt_bboxes[0].squeeze(0).float())
        gt_bboxes_ = [bbox.float() for bbox in gt_bboxes_]
        # gt_points = [(box[:, :2] + box[:, 2:]) / 2 for box in gt_bboxes_]

        if points is not None:
            random_points = points

        if frame_id == 0:
            # bbox_pred = gt_bboxes[0][0]
            # self.memo = Dict()
            # self.memo.bbox = bbox_xyxy_to_cxcywh(bbox_pred)
            # self.init(img, gt_points, img_metas)
            # if self.test_cfg.link_mode == 'bbox':
            #     self.init(img, random_points, img_metas)
            # elif self.test_cfg.link_mode == 'feat':
            #     self.init_roi_align(img, random_points, img_metas)
            self.init_point_bbox(img, points, noisy_bbox, img_metas)
            bbox_pred = self.memo.bbox
            # vis
            if self.test_cfg.vis == True:
                import cv2
                # import numpy as np
                import copy
                import os
                prefix = self.test_cfg.vis_path
                # prefix = 'debug1'
                gt_vis = torch.stack(gt_bboxes_, dim=0).squeeze(1)
                gt_vis_1 = np.array(gt_vis.cpu().squeeze(0)).astype(np.int32)
                pred_bbox_vis = self.vis_box.squeeze(0).clone().cpu().numpy().astype(np.int32)
                ppppp = random_points[0].squeeze(0).squeeze(0).clone().cpu().numpy().astype(np.int32)
                vis_noisy_bbox = noisy_bbox[0].squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)
                file_name = img_metas[0]['filename']
                # frame_id = i_m['frame_id']
                video_name = file_name.split('/')[-3]
                h, w, _ = img_metas[0]['img_shape']
                img1 = cv2.imread(file_name)
                img1 = cv2.resize(img1, (w, h))
                igs1 = copy.deepcopy(img1)
                if not os.path.isdir(prefix):
                    os.makedirs(prefix)
                igs1 = cv2.rectangle(igs1, (pred_bbox_vis[0], pred_bbox_vis[1]),
                                     (pred_bbox_vis[2], pred_bbox_vis[3]),
                                     color=(0, 256, 0))
                if self.test_cfg.test_mode == 'point':
                    igs1 = cv2.circle(igs1, (ppppp[0], ppppp[1]), 2, (0, 0, 255), 4)
                elif self.test_cfg.test_mode == 'bbox':
                    igs1 = cv2.rectangle(igs1, (vis_noisy_bbox[0], vis_noisy_bbox[1]),
                                         (vis_noisy_bbox[2], vis_noisy_bbox[3]),
                                         color=(255, 255, 0))
                igs1 = cv2.rectangle(igs1, (gt_vis_1[0], gt_vis_1[1]),
                                     (gt_vis_1[2], gt_vis_1[3]),
                                     color=(0, 0, 0))
                cv2.imwrite('{}/{}_{}.jpg'.format(prefix, video_name, frame_id), igs1)
            best_score = -1.
        else:
            best_score, bbox_pred = self.track(img, self.memo.bbox)
            self.memo.bbox = bbox_xyxy_to_cxcywh(bbox_pred)

        results = dict()
        results['track_bboxes'] = np.concatenate(
            (bbox_pred.cpu().numpy(), np.array([best_score])))
        return results
