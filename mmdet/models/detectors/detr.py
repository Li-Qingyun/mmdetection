# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmengine.model import xavier_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from ..layers import (DetrTransformerDecoder, DetrTransformerEncoder,
                      SinePositionalEncoding)
from .detection_transformer import TransformerDetector


@MODELS.register_module()
class DETR(TransformerDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self, *args, **kwargs) -> None:
        super(DETR, self).__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        # initialize encoder, decoder, query_embed, positional_encoding
        self._init_transformer()
        # initialize input projection
        self._init_input_proj()  # TODO: Can it be replaced with ChannelMapper?

    def _init_transformer(self) -> None:
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = DetrTransformerDecoder(**self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims  # TODO
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def _init_input_proj(self) -> None:
        in_channels = self.backbone.feat_dim  # TODO: Is this correct stably?
        self.input_proj = Conv2d(in_channels, self.embed_dims, kernel_size=1)

    def init_weights(self) -> None:  # TODO
        super(TransformerDetector, self).init_weights()
        self._init_transformer_weights()
        self._is_init = True  # TODO

    def _init_transformer_weights(self) -> None:  # TODO
        # follow the DetrTransformer to init parameters
        for coder in [self.encoder, self.decoder]:
            for m in coder.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')

    def forward_pretransformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Dict[str, Tensor]:
        feat = img_feats[-1]
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = feat.size(0)
        if batch_data_samples is not None:
            batch_input_shape = batch_data_samples[0].batch_input_shape  # noqa
            # TODO: should there be an assert about equal batch_input_shape?
            img_shape_list = [
                sample.img_shape  # noqa
                for sample in batch_data_samples
            ]
        else:
            batch_input_shape = feat.shape
            # TODO: should the batch_data_samples be OptSampleList?
            img_shape_list = [feat.shape for _ in batch_size]

        input_img_h, input_img_w = batch_input_shape
        masks = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0

        feat = self.input_proj(feat)
        # interpolate masks to have the same spatial shape with feat
        masks = F.interpolate(
            masks.unsqueeze(1), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]

        seq_feats = dict(feat=feat, masks=masks, pos_embed=pos_embed)
        return seq_feats  # noqa

    def forward_transformer(
            self,
            feat: Tensor,
            masks: Tensor,
            pos_embed: Tensor,
            query_embed: nn.Module,
            return_memory: bool = False) -> Union[Tuple[Tensor], Any]:
        bs, c, h, w = feat.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [h*w, bs, c]
        feat = feat.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        masks = masks.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=feat, query_pos=pos_embed, query_key_padding_mask=masks)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=masks)
        out_dec = out_dec.transpose(1, 2)
        if return_memory:
            # TODO: The original output is 'out_dec, memory', I kept it here.
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
            return InstanceData(outs_trans=out_dec, memory=memory)
        return InstanceData(outs_trans=out_dec)
        # TODO: For keeping the multi_apply in DETRHead.forward
