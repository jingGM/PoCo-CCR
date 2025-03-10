import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.clusters import Mlp


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, '`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(~key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(~attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        # attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(SelfAttention, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        # self.dropout = build_dropout_layer(dropout)
        self.norm_out = nn.LayerNorm(d_model)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(d_model * 2, d_model), nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(d_model)
        )

    def forward(
            self,
            input_states,
            memory_weights=None,
            memory_masks=None,
            attention_factors=None,
            attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            input_states,
            input_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        norm_states = self.norm_out(hidden_states + input_states)
        output_states = self.output(norm_states)
        return output_states, attention_scores


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(CrossAttention, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        # self.dropout = build_dropout_layer(dropout)
        self.norm_out = nn.LayerNorm(d_model)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(d_model * 2, d_model), nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(d_model)
        )

    def forward(
            self,
            input_states,
            memory_states,
            memory_weights=None,
            memory_masks=None,
            attention_factors=None,
            attention_masks=None,
    ):
        input_states, self_attention_scores = self.self_attention(
            input_states,
            input_states,
            input_states,
            key_weights=None,
            key_masks=None,
            attention_factors=None,
            attention_masks=None,
        )
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        norm_states = self.norm_out(hidden_states + input_states)
        output_states = self.output(norm_states)
        return output_states, attention_scores


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.heads = cfg.heads
        self.self_attention_1 = SelfAttention(d_model=cfg.dense_dim, num_heads=cfg.heads)
        self.cross_attention_1 = CrossAttention(d_model=cfg.dense_dim, num_heads=cfg.heads)
        self.self_attention_2 = SelfAttention(d_model=cfg.dense_dim, num_heads=cfg.heads)
        self.cross_attention_2 = CrossAttention(d_model=cfg.dense_dim, num_heads=cfg.heads)

        self.f_linear_1 = nn.Linear(cfg.dense_dim, cfg.dense_dim)
        self.c_linear_1 = nn.Linear(cfg.dense_dim, cfg.dense_dim)

        self.f_linear_2 = nn.Linear(cfg.dense_dim, cfg.dense_dim)
        self.c_linear_2 = nn.Linear(cfg.dense_dim, cfg.dense_dim)
        # self.f_linear = nn.Linear(cfg.dense_dim, cfg.head_dim)
        # self.c_linear = nn.Linear(cfg.dense_dim, cfg.head_dim)

        self.cluster_layer = Mlp(in_features=cfg.dense_dim * 2, out_features=cfg.head_dim,
                                 hidden_features=int(cfg.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.out_layer = Mlp(in_features=cfg.head_dim, out_features=1,
                             hidden_features=int(cfg.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)

    def forward(self, feats, src_masks=None, ref_masks=None):
        B, F, N, C = feats.shape
        # feats = rearrange(feats, "b f n c -> (b f) n c")  # B,N,C

        reff_fts = rearrange(feats[:, :1, :, :].repeat(1, F - 1, 1, 1), "b f m c -> (b f) m c")  # B*(F-1),Mr,C
        srcf_fts = rearrange(feats[:, 1:, :, :], "b f m c -> (b f) m c")  # B*(F-1),Ms,C

        # print(srcf_fts.shape)
        src_fts, src_fscore = self.self_attention_1(input_states=srcf_fts, memory_masks=src_masks)
        ref_fts, ref_fscore = self.self_attention_1(input_states=reff_fts, memory_masks=ref_masks)

        src_cfts, src_cscore = self.cross_attention_1(src_fts, ref_fts, memory_masks=ref_masks)  # (b,n,c)
        ref_cfts, ref_cscore = self.cross_attention_1(ref_fts, src_fts, memory_masks=src_masks)  # (b,m,c)

        src_cfts = self.f_linear_1(srcf_fts) + self.c_linear_1(src_cfts)
        ref_cfts = self.f_linear_1(reff_fts) + self.c_linear_1(ref_cfts)

        src_fts, src_fscore = self.self_attention_2(input_states=src_cfts, memory_masks=src_masks)
        ref_fts, ref_fscore = self.self_attention_2(input_states=ref_cfts, memory_masks=ref_masks)

        src_cfts_2, src_cscore = self.cross_attention_2(src_fts, ref_fts, memory_masks=ref_masks)  # (b,n,c)
        ref_cfts_2, ref_cscore = self.cross_attention_2(ref_fts, src_fts, memory_masks=src_masks)  # (b,m,c)

        src_cfts_l = self.f_linear_2(srcf_fts) + self.c_linear_2(src_cfts_2)
        ref_cfts_l = self.f_linear_2(reff_fts) + self.c_linear_2(ref_cfts_2)
        # scores = torch.einsum('bnd,bmd->bnm', ref_cfts, src_cfts) / self.d_model

        _, Mr, _ = ref_cfts_l.shape
        _, Ms, _ = src_cfts_l.shape
        ref_cluster = ref_cfts_l.unsqueeze(2).repeat(1, 1, Ms, 1)  # B(F-1)H,Mr,Ms,D
        src_cluster = src_cfts_l.unsqueeze(1).repeat(1, Mr, 1, 1)  # B(F-1)H,Mr,Ms,D
        all_clusters = self.cluster_layer(torch.concat((ref_cluster, src_cluster), dim=-1))
        # ref_cluster_norm =  # B(F-1)H,Ms,D
        src_cluster_norm = all_clusters.mean(dim=1).mean(dim=1)  # B(F-1)H,D

        local_score = self.out_layer(src_cluster_norm)
        local_score = torch.sigmoid(local_score).view(B, F - 1)
        return local_score