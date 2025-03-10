import torch
from einops import rearrange
from torch import nn

from src.models.clusters import pairwise_cos_sim
from src.models.model_utils import GroupNorm
from src.models.clusters import Mlp
from utils.functions import index_batched_points
from src.models.cross_models.cross_attention import SelfAttention


class CoCsAttn(nn.Module):
    def __init__(self, cfg):
        super(CoCsAttn, self).__init__()
        # dim, mlp_ratio = 4., use_layer_scale = True, layer_scale_init_value = 1e-5,
        self.no_correlation = cfg.no_correlation
        self.max_matches = cfg.max_matches
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim

        self.self_attention_1 = SelfAttention(d_model=cfg.dense_dim, num_heads=cfg.heads)

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.ref_layer = Mlp(in_features=cfg.points_head_dim, out_features=self.head_dim,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.src_layer = Mlp(in_features=cfg.points_head_dim, out_features=self.head_dim,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.cluster_layer = Mlp(in_features=self.head_dim * 2, out_features=self.head_dim,
                                 hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.out_layer = Mlp(in_features=self.head_dim * self.heads, out_features=1,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)

    def forward(self, feats):
        """
        Args:
            feats (Tensor): (B, F, N, C)

        Returns:
            final_score: torch.Tensor (B, N)
        """
        B, F, N, C = feats.shape
        feats = rearrange(feats, "b f n c -> (b f) n c")  # B,N,C
        fts, scores = self.self_attention_1(input_states=feats, memory_masks=None)

        fts = rearrange(fts, "(b f) m c -> b f m c", f=F)  # B,F,M,C

        ref_fts = rearrange(fts[:, :1, :, :].repeat(1, F - 1, 1, 1), "b f m c -> (b f) m c")  # B*(F-1),Mr,C
        ref_fts = rearrange(ref_fts, "b m (e c) -> (b e) m c", e=self.heads)  # B(F-1)H,Mr,D

        src_fts = rearrange(fts[:, 1:, :, :], "b f m c -> (b f) m c")  # B*(F-1),Ms,C
        src_fts = rearrange(src_fts, "b m (e c) -> (b e) m c", e=self.heads)  # B(F-1)H,Ms,D

        correlation = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(ref_fts, src_fts)  # B(F-1)H,Mr,Ms
        )

        ref_center_processed = self.ref_layer(ref_fts)  # B(F-1)H,Mr,D
        src_center_processed = self.src_layer(src_fts)  # B(F-1)H,Ms,D
        BH, Mr, Ms = correlation.shape
        values, indices = correlation.view(BH, Mr * Ms).topk(k=self.max_matches, dim=1)
        mask = torch.zeros_like(correlation).view(BH, Mr * Ms)
        mask = mask.scatter_(1, indices, 1.).view(BH, Mr, Ms)  # B(F-1)H,Mr,Ms
        correlation = correlation * mask  # B(F-1)H,Mr,Ms

        ref_cluster = correlation.unsqueeze(-1) * ref_center_processed.unsqueeze(2)  # B(F-1)H,Mr,Ms,D
        src_cluster = correlation.unsqueeze(-1) * src_center_processed.unsqueeze(1)  # B(F-1)H,Mr,Ms,D
        all_clusters = self.cluster_layer(torch.concat((ref_cluster, src_cluster), dim=-1))

        ref_cluster_norm = all_clusters.sum(dim=1) / (correlation.sum(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,Ms,D
        src_cluster_norm = ref_cluster_norm.sum(dim=1) / (
                correlation.sum(dim=2).mean(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,D

        out_cluster = rearrange(src_cluster_norm, "(b e) c -> b (e c)", e=self.heads)
        local_score = self.out_layer(out_cluster)
        local_score = torch.sigmoid(local_score).view(B, F - 1)

        return local_score