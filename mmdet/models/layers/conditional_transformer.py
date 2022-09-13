# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Sequence
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule
from torch.nn import ModuleList
from .transformer import MLP
from mmcv.cnn.bricks.drop import build_dropout

from mmengine.utils import deprecated_api_warning
from .c_attention import CMultiheadAttention

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class ConditionalDetrTransformerDecoder(BaseModule):

    def __init__(self,
                 layer_cfg=None,
                 num_layers=None,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        if isinstance(layer_cfg, dict):
            layer_cfg = [copy.deepcopy(layer_cfg) for _ in range(num_layers)]
        else:
            assert isinstance(layer_cfg, list) and \
                   len(layer_cfg) == num_layers  # TODO
        self.layer_cfg = layer_cfg  # TODO
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()
        self.embed_dims = self.layers[0].embed_dims  # TODO
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]
        #####################################################
        d_model = 256
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        # for layer_id in range(kwargs['num_layers'] - 1):
        #     self.layers[layer_id + 1].attentions[1].ca_qpos_proj = None
        #####################################################

    def _init_layers(self):
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                ConditionalDetrTransformerDecoderLayer(**self.layer_cfg[i]))

    def forward(self, query, *args, **kwargs):
        reference_points_before_sigmoid = self.ref_point_head(kwargs['query_pos'])  # [num_queries, batch_size, 2]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            kwargs['query_sine_embed'] = query_sine_embed
            kwargs['is_first'] = (layer_id == 0)
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        if self.post_norm is not None:
            query = self.post_norm(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate), reference_points

        return query, reference_points


class ConditionalDetrTransformerDecoderLayer(BaseModule):

    def __init__(self,
                 self_attn_cfg=dict(
                     type='MultiheadAttention',
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0),
                 cross_attn_cfg=dict(
                     type='MultiheadAttention',
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0),
                 ffn_cfg=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        super(ConditionalDetrTransformerDecoderLayer, self).__init__(init_cfg)
        for attn_cfg in (self_attn_cfg, cross_attn_cfg):
            if 'batch_first' in attn_cfg:
                assert batch_first == attn_cfg['batch_first']
            else:
                attn_cfg['batch_first'] = batch_first
        self.batch_first = batch_first
        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        self.self_attn = FMultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = PMultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims  # TODO
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                self_attn_masks=None,
                cross_attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                query_sine_embed=None,
                is_first=None,
                **kwargs):

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks,
            key_padding_mask=query_key_padding_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            query_sine_embed=query_sine_embed,
            is_first=is_first,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query



class FMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(embed_dims, embed_dims)
        self.sa_qpos_proj = nn.Linear(embed_dims, embed_dims)
        self.sa_kcontent_proj = nn.Linear(embed_dims, embed_dims)
        self.sa_kpos_proj = nn.Linear(embed_dims, embed_dims)
        self.sa_v_proj = nn.Linear(embed_dims, embed_dims)
        self.self_attn = CMultiheadAttention(embed_dims, num_heads, attn_drop, vdim=embed_dims,**kwargs)

        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.batch_first = batch_first
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='CMultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        if identity is None:
            identity = query
        q_content = self.sa_qcontent_proj(query)
        q_pos = self.sa_qpos_proj(query_pos)

        k_content = self.sa_kcontent_proj(query)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(query)

        # num_queries, bs, n_model = q_content.shape
        # hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        if self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        out = self.self_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))



class PMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 keep_query_pos=False,
                 **kwargs):
        super().__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(embed_dims, embed_dims)
        self.ca_qpos_proj = nn.Linear(embed_dims, embed_dims)
        self.ca_kcontent_proj = nn.Linear(embed_dims, embed_dims)
        self.ca_kpos_proj = nn.Linear(embed_dims, embed_dims)
        self.ca_v_proj = nn.Linear(embed_dims, embed_dims)
        self.ca_qpos_sine_proj = nn.Linear(embed_dims, embed_dims)
        self.cross_attn = CMultiheadAttention(embed_dims * 2, num_heads, attn_drop, vdim=embed_dims,**kwargs)

        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.batch_first = batch_first
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.keep_query_pos = keep_query_pos
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='CMultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                query_sine_embed=None,
                is_first=None,
                **kwargs):
        if identity is None:
            identity = query
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(query)
        k_content = self.ca_kcontent_proj(key)
        v = self.ca_v_proj(key)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(key_pos)

        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.num_heads, n_model//self.num_heads)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.num_heads, n_model//self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.num_heads, n_model//self.num_heads)
        k_pos = k_pos.view(hw, bs, self.num_heads, n_model//self.num_heads)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        if self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        out = self.cross_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

