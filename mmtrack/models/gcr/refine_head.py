from mmdet.models.builder import HEADS
from .uti1 import UTIDIIIOUHead, UTIDIIBoxHead
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import build_backbone, build_head, build_neck


@HEADS.register_module()
class UTIDIIroiIOUHead(UTIDIIIOUHead):
    def __init__(self, final_head=None, *args, **kwargs):
        super(UTIDIIroiIOUHead, self).__init__(*args, **kwargs)
        if final_head is not None:
            self.final_head = build_head(final_head)

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
        b, c, h, w = roi_feat.shape
        proposal_feat_iic, filter_roi_feat = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = roi_feat + self.instance_interactive_conv_dropout(proposal_feat_iic.permute(0, 2, 1).reshape(b, c, h, w))
        ious_pred = self.fc_cls_1(self.final_head(proposal_feat)[0]).view(N, num_proposals, self.num_classes)
        # ious_pred = self.final_head(proposal_feat)[0].view(N, num_proposals, self.num_classes)
        # ious_pred = self.fc_cls_1(self.fc_cls(cls_feat)).view(N, num_proposals, self.num_classes

        return ious_pred, filter_roi_feat


@HEADS.register_module()
class UTIDIIroiBoxHead(UTIDIIBoxHead):
    def __init__(self, final_head=None, *args, **kwargs):
        super(UTIDIIroiBoxHead, self).__init__(*args, **kwargs)
        if final_head is not None:
            self.final_head = build_head(final_head)

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
        proposal_feat_iic, _ = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        # jump
        # proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
        #     proposal_feat_iic)
        # !jump
        # proposal_feat = self.instance_interactive_conv_dropout(proposal_feat_iic)
        b, c, h, w = roi_feat.shape
        proposal_feat = roi_feat + self.instance_interactive_conv_dropout(proposal_feat_iic.permute(0, 2, 1).reshape(b, c, h, w))

        bbox_delta = self.final_head(proposal_feat)[1].view(N, num_proposals, self.reg_dims)
        # obj_feat = self.instance_interactive_conv_norm(proposal_feat)
        #
        # # FFN
        # obj_feat = self.ffn_norm(self.ffn(obj_feat))
        #
        # reg_feat = obj_feat
        #
        # for reg_layer in self.reg_fcs:
        #     reg_feat = reg_layer(reg_feat)
        #
        # # bbox_delta = self.reg_1(self.fc_reg(reg_feat)).view(N, num_proposals, self.reg_dims)
        # # kuiran
        # bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, self.reg_dims)

        return bbox_delta