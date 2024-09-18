import torch

from mmtrack.models import MODELS
from mmtrack.models.uti.uti1 import UnifiedTrackersInput2
import torch.nn as nn
import clip


@MODELS.register_module()
class ClipRefine(UnifiedTrackersInput2):
    def __init__(self, backbone, *args, **kwargs):
        super(ClipRefine, self).__init__(backbone, *args, **kwargs)
        self.model, self.preprocess = clip.load(backbone, device="cuda")
        self.text_linear = nn.Linear(1024, 256)

    def extract_feat(self, img, text_info):
        image = self.preprocess(img)
        text = clip.tokensize(text_info).cuda()
        with torch.no_grad():
            image_features = tuple(self.model.encode_image(image))
            text_features = self.model.encode_text(text)
        if self.with_neck:
            x = self.neck(image_features)
        return x, text_features

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      points,
                      **kwargs):
        text_info = []
        for img_meta in img_metas:
            text_info.append(img_meta['ori_filename'].split('/')[0])
        x, text_features = self.extract_feat(img, text_info)
        text_features = self.text_linear(text_features)
        points = self.gen_points(gt_bboxes)
        proposal_boxes, imgs_whwh, point_proposal_features, bbox_proposal_features, pred_iou_proposal_features = \
            self.rpn_head.forward_train(x, img_metas, points=points)
        gt_points = [_ for _ in points.float().squeeze(1)]
        input_mode = 'point'
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
            imgs_whwh=imgs_whwh,
            text_feature=text_features
        )
