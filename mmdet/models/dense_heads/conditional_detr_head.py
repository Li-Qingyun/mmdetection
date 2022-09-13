# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
from mmengine.structures import InstanceData
from torch import Tensor

from ..utils import multi_apply
from mmdet.registry import MODELS
from .detr_head import DETRHead
from mmdet.models.layers.transformer import inverse_sigmoid

def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    import numpy as np
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

@MODELS.register_module()
class ConditionalDETRHead(DETRHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        TODO
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ConditionalDETRHead, self).__init__(*args, **kwargs)


    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    def forward(self, x: InstanceData) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward function.

        Args:
            x:
            TODO

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

            - all_cls_scores_list (list[Tensor]): Classification scores \
            for each scale level. Each is a 4D-tensor with shape \
            [nb_dec, bs, num_query, cls_out_channels]. Note \
            `cls_out_channels` should includes background.
            - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
            outputs for each scale level. Each is a 4D-tensor with \
            normalized coordinate format (cx, cy, w, h) and shape \
            [nb_dec, bs, num_query, 4].
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, outs_dec: InstanceData) -> Tuple[Tensor, Tensor]:
        """"Forward function for a single feature level.

        Args: TODO

        Returns:
            tuple[Tensor]:

            - all_cls_scores (Tensor): Outputs from the classification head, \
            shape [nb_dec, bs, num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
            - all_bbox_preds (Tensor): Sigmoid outputs from the regression \
            head with normalized coordinate format (cx, cy, w, h). \
            Shape [nb_dec, bs, num_query, 4].
        """
        ##################
        reference_before_sigmoid = inverse_sigmoid(outs_dec.reference_points[0])
        outputs_coords = []
        for lvl in range(outs_dec.outs_trans[0].shape[0]):
            tmp = self.fc_reg(self.activate(self.reg_ffn(outs_dec.outs_trans[0][lvl])))
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        ########################
        all_cls_scores = self.fc_cls(outs_dec.outs_trans[0])
        all_bbox_preds = outputs_coord
        k1 = all_cls_scores.sum()
        k2 = all_bbox_preds.sum()
        return all_cls_scores, all_bbox_preds