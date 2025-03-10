import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as function
from src.models.cross_models.r2former_functions import no_grad_trunc_normal_, Block


class R2Former(nn.Module):
    def __init__(self, cfg, decode_norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        # img_size=(224,224), decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        #          decoder_mlp_ratio=4., decoder_norm_layer=nn.LayerNorm, embed_dim=384, num_classes=1, num_patches=16, input_dim=256, num_corr=1):
        super(R2Former, self).__init__()
        self.input_dim = cfg.dense_dim
        self.selected_number = cfg.selected_number
        self.head_idx = cfg.head_idx
        self.num_corr = cfg.num_corr
        self.decoder_embed_dim = cfg.decoder_embed_dim
        self.decoder_num_heads = cfg.decoder_num_heads
        self.decoder_mlp_ratio = cfg.decoder_mlp_ratio
        self.decoder_depth = cfg.decoder_depth

        self.geo_dim = 9
        self.pair_head = nn.Linear(self.geo_dim, self.decoder_embed_dim, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        no_grad_trunc_normal_(self.cls_token, std=.02)

        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        no_grad_trunc_normal_(self.cls_token_2, std=.02)

        self.blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.decoder_mlp_ratio,
                  qkv_bias=True, norm_layer=decode_norm_layer) for i in range(self.decoder_depth)])

        self.blocks_2 = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.decoder_mlp_ratio,
                  qkv_bias=True, norm_layer=decode_norm_layer) for i in range(2)])
        self.decoder_norm = decode_norm_layer(self.decoder_embed_dim)

        self.pair_head = nn.Linear(self.geo_dim, self.decoder_embed_dim, bias=True)
        self.pair_head_2 = nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=True)

        self.decoder_pred = nn.Linear(self.decoder_embed_dim, 2, bias=True)  # decoder to patch
        self.sm = torch.nn.Softmax(dim=1)
        # self.num_classes = num_classes
        # self.embed_dim = embed_dim
        # self.decoder_embed_dim = decoder_embed_dim
        # self.num_patches = num_patches
        # self.img_size = img_size
        # self.patch_size = [16, 16]
        #
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)
        # self.ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # fixed sin-cos embedding
        #
        #
        # # --------------------------------------------------------------------------
        # self.decoder_pos_embed.data[:, 1:].normal_(mean=0.0, std=0.01)
        # self.initialize_weights_all()
        # self.cos = nn.CosineSimilarity(dim=1)
        #

    def initialize_weights_all(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def to_map(self, x):
        # p = self.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, x.shape[-1]))
        x = torch.permute(x, (0, 3, 1, 2))
        return x

    def forward(self, feats, geometries, similarities):
        """
        Args:
            feats (Tensor): (B, F, N, C)
            geometries (Tensor): (B, F, N, C)
            similarities (Tensor): (B, F, H, N, K)

        Returns:
            final_score: torch.Tensor (B, N)
        """
        B, F, H, N, K = similarities.shape
        _, _, N1, C = feats.shape
        _, _, N2, _ = geometries.shape
        D = 3
        assert N == N1 == N2, "the similarities number {} is different from points number {}/{}".format(N, N1, N2)

        if self.head_idx == -1:
            similarities = torch.mean(similarities, dim=2).detach()  # B, F, M, K
        else:
            similarities = similarities[:, :, self.head_idx, :, :].detach()  # B, F, M, K

        sim = similarities.view(B * F, N, K)
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)  # B,1,K
        mask = torch.zeros_like(sim)  # binary [B,N,K]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = torch.sum(sim * mask, dim=2)  # [B,N]
        # values, indices = torch.topk(sim, k=self.selected_number, dim=1)

        feats = feats.view(B * F, N, C)
        geometries = geometries[:, :, :, :D].view(B * F, N, D).detach()

        selected_sim = sim.unsqueeze(-1).view(B, F, self.selected_number, 1)
        # torch.gather(sim, dim=1, index=indices).unsqueeze(-1).view(B, F, self.selected_number, 1)
        selected_fts = feats.view(B, F, self.selected_number, C)
        # torch.gather(feats, dim=1, index=indices).view(B, F, self.selected_number, C)
        selected_geos = geometries.view(B, F, self.selected_number, D)
        # torch.gather(geometries, dim=1, index=indices).view(B, F, self.selected_number, D)
        selected_fts = function.normalize(selected_fts, p=2, dim=3)

        coord_dim = D + 1
        selected_coord = torch.concat((selected_geos, selected_sim), dim=3)
        x_coordinate = selected_coord[:, :1, :, :].repeat(1, F - 1, 1, 1).view(
            B * (F - 1), self.selected_number, coord_dim)
        y_coordinate = selected_coord[:, 1:, :, :].contiguous().view(B * (F - 1), self.selected_number, coord_dim)

        x_rerank_token = selected_fts[:, :1, :, :].repeat(1, F - 1, 1, 1).view(B * (F - 1), self.selected_number, C)
        y_rerank_token = selected_fts[:, 1:, :, :].contiguous().view(B * (F - 1), self.selected_number, C)
        correlation = torch.matmul(x_rerank_token, y_rerank_token.permute((0, 2, 1)))
        xyz_matrix = torch.cat(
            [x_coordinate.unsqueeze(2).repeat(1, 1, y_rerank_token.shape[1], 1),
             y_coordinate.unsqueeze(1).repeat(1, x_rerank_token.shape[1], 1, 1),
             correlation.unsqueeze(3)],
            dim=3)

        order_q = torch.argsort(correlation.unsqueeze(3), dim=2, descending=True).repeat(1, 1, 1, self.geo_dim)
        order_k = torch.argsort(correlation.unsqueeze(3), dim=1, descending=True).repeat(1, 1, 1, self.geo_dim)

        select_q = torch.gather(input=xyz_matrix, index=order_q[:, :, :self.num_corr, :], dim=2)
        select_k = torch.gather(input=xyz_matrix, index=order_k[:, :self.num_corr, :, :], dim=1)

        select = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        select_copy = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        B_select, N_select, _, _ = select.shape

        pair_matrix = self.pair_head(select.reshape(B_select * N_select * self.num_corr, self.geo_dim)).reshape(
            B_select * N_select, self.num_corr, self.decoder_embed_dim)
        pair_matrix += get_3d_sincos_pos_embed_from_grid(
            self.decoder_embed_dim, select_copy.reshape(B_select * N_select, self.num_corr, self.geo_dim)[:, :, 4:7])
        x = torch.cat([self.cls_token_2.repeat(B_select * N_select, 1, 1), pair_matrix], dim=1)
        for blk in self.blocks_2:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.pair_head_2(x[:, 0, :].reshape(B_select * N_select, self.decoder_embed_dim)).reshape(
            B_select, N_select, self.decoder_embed_dim)
        x = x.reshape(B_select, N_select, self.decoder_embed_dim) + get_3d_sincos_pos_embed_from_grid(
            self.decoder_embed_dim, select_copy[:, :, 0, 0:3])
        x = torch.cat([self.cls_token.repeat(B_select, 1, 1), x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        local_score = self.decoder_pred(x[:, 0])
        local_score = self.sm(local_score)[:, 1].view(B, F-1)
        return local_score


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[:, :, 0])  # (H*W, D/2)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[:, :, 1])  # (H*W, D/2)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[:, :, 2])  # (H*W, D/2)

    emb = torch.cat([emb_x, emb_y, emb_z], dim=2)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    # pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('bm,d->bmd', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=2)  # (M, D)
    return emb
