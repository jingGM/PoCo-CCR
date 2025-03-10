import torch
from torch import nn

from src.models.model_utils import GroupNorm
from utils.config import PointReducerConfig
from src.models.context_cluster import PointReducer, BasicBlocks


class CoCsTotal(nn.Module):
    def __init__(self, cfg):
        super(CoCsTotal, self).__init__()
        self.sparse_layer = nn.Linear(cfg.sparse_dim, cfg.hidden)
        self.dense_layer = nn.Linear(cfg.dense_dim, cfg.hidden)
        self.context_model = BasicBlocks(dim=cfg.cocs.embed_dims, index=0, layers=cfg.cocs.layers,
                                         mlp_ratio=cfg.cocs.mlp_ratios, act_layer=nn.GELU, norm_layer=GroupNorm,
                                         drop_rate=0, drop_path_rate=0,
                                         use_layer_scale=cfg.cocs.use_layer_scale,
                                         layer_scale_init_value=cfg.cocs.layer_scale_init_value,
                                         heads=cfg.cocs.heads, head_dim=cfg.cocs.head_dim
                                         )
        self.reducer = PointReducer(config=PointReducerConfig(cfg=cfg.reducer), norm_layer=None)
        self.output_dim = cfg.dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, cfg.dim), nn.GELU(),
            nn.Linear(cfg.dim, 1), nn.Sigmoid()
        )

    def forward(self, ref_feats, ref_geos, src_feats, src_geos):
        """
        Args:
            ref_geos (Tensor): (B, 1, K, D)
            src_geos (Tensor): (B, N, K, D)
            ref_feats (Tensor): (B, 1, K, C)
            src_feats (Tensor): (B, N, K, C)

        Returns:
            final_score: torch.Tensor (B, N)
        """
        B, N, K, D = src_geos.shape
        ref_sparse_geos = torch.mean(ref_geos, dim=2, keepdim=True).repeat(1, N, 1, 1)  # B,N,1,D
        src_sparse_geos = torch.mean(src_geos, dim=2, keepdim=True)  # B,N,1,D

        dense_geos = torch.concatenate((ref_geos.repeat(1, N, 1, 1), src_geos), dim=2).view(B * N, K * 2, D)  # B*N,K1+K2,D
        sparse_geos = torch.concatenate((ref_sparse_geos, src_sparse_geos), dim=2).view(B * N, 2, D)  # B*N,2,D

        _, _, _, C1 = ref_feats.shape
        # _, _, C2 = src_global.shape
        # sparse_fts = torch.concatenate((ref_global.repeat(1, N, 1).unsqueeze(-2), src_global.unsqueeze(-2)),
        #                                dim=-2).view(B * N, 2, C2)  # B,2,C
        dense_fts = torch.concatenate((ref_feats.repeat(1, N, 1, 1), src_feats), dim=-2).view(B * N, K * 2, C1)  # B*N,K1+K2,C

        # sparse_fts = self.sparse_layer(sparse_fts)
        dense_fts = self.dense_layer(dense_fts)
        neighbors = torch.arange(0, 2 * K, 1).unsqueeze(0).unsqueeze(0).repeat(B * N, 2, 1).to(dense_fts.device)

        features, centers, similarities = self.context_model(features=dense_fts, neighbors=neighbors)  # B*N,K1+K2,C
        features = self.reducer(dense_geos=dense_geos, sparse_geos=sparse_geos, dense_feats=features,
                                down_nei_inds=neighbors, sparse_nei_inds=None).view(B*N, self.output_dim)  # B*N,2*C

        local_score = self.mlp(features).squeeze(-1).squeeze(-1).view(B, N)  # B*N,1,1
        return local_score
