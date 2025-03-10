import torch
from einops import rearrange
from torch import nn

from src.models.clusters import pairwise_cos_sim
from src.models.model_utils import GroupNorm
from src.models.clusters import Mlp
from utils.functions import index_batched_points


class ClusterBlock(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, norm_layer=GroupNorm, heads=4, head_dim=24, mlp_ratio=4):
        super(ClusterBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.heads = heads
        self.head_dim = head_dim
        hidden_dim = heads * head_dim

        self.query = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2))
        self.query_norm = GroupNorm(num_channels=hidden_dim, num_groups=1)

        self.value = nn.Sequential(nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2))
        self.value_norm = GroupNorm(num_channels=hidden_dim, num_groups=1)

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        self.mlp = Mlp(in_features=hidden_dim, out_features=hidden_dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=0.)

    def forward(self, features, neighbors):
        query_fts = self.query(features)
        query_fts_nm = self.query_norm(query_fts)  # B,N,C
        q_fts = rearrange(query_fts_nm, "b n (e c) -> (b e) n c", e=self.heads)

        value_fts = self.value(features)
        value_fts_nm = self.value_norm(value_fts)  # B,N,C
        v_fts = rearrange(value_fts_nm, "b n (e c) -> (b e) n c", e=self.heads)

        vx_query_knn_fts = index_batched_points(points=query_fts_nm, idx=neighbors)  # B,Mq,K,C
        vx_query_fts = torch.mean(vx_query_knn_fts, dim=2)  # B,Mq,C
        vx_query_fts_heads = rearrange(vx_query_fts, "b m (e c) -> (b e) m c", e=self.heads)

        vx_value_knn_fts = index_batched_points(points=value_fts_nm, idx=neighbors)  # B,Mv,K,C
        vx_value_fts = torch.mean(vx_value_knn_fts, dim=2)  # B,Mv,C
        vx_value_fts_heads = rearrange(vx_value_fts, "b m (e c) -> (b e) m c", e=self.heads)

        sim = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(vx_query_fts_heads, q_fts)  # [B,Mq,Nq]
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)  # B,1,Nq
        mask = torch.zeros_like(sim)  # binary #[B,Mq,Nq]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask  # [B,Mq,Nq]

        value_sim = v_fts.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)  # B,Mq,Nv,C
        centers_fts = (value_sim.sum(dim=2) + vx_value_fts_heads) / (sim.sum(dim=-1, keepdim=True) + 1.0)  # B,M,C

        centers_fts_arrange = rearrange(centers_fts, "(b e) m c -> b m (e c)", e=self.heads)
        out = self.mlp(centers_fts_arrange)
        return out


class CrossCluster(nn.Module):
    def __init__(self, cfg):
        super(CrossCluster, self).__init__()
        # dim, mlp_ratio = 4., use_layer_scale = True, layer_scale_init_value = 1e-5,
        self.no_correlation = cfg.no_correlation
        self.max_matches = cfg.max_matches
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim

        self.cluster = ClusterBlock(dim=cfg.dense_dim, act_layer=nn.GELU, norm_layer=GroupNorm, mlp_ratio=cfg.mlp_ratio,
                                    heads=cfg.heads, head_dim=cfg.head_dim)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        # self.sigmoid = nn.Sigmoid()

        self.ref_layer = Mlp(in_features=self.head_dim, out_features=self.head_dim,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.src_layer = Mlp(in_features=self.head_dim, out_features=self.head_dim,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.cluster_layer = Mlp(in_features=self.head_dim * 2, out_features=self.head_dim,
                                 hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.out_layer = Mlp(in_features=self.head_dim * self.heads, out_features=1,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)

    def forward(self, feats, neighbors):
        """
        Args:
            feats (Tensor): (B, F, N, C)
            neighbors (Tensor): (B, F, M, K)

        Returns:
            final_score: torch.Tensor (B, N)
        """
        B, F, N, C = feats.shape
        feats = rearrange(feats, "b f n c -> (b f) n c")  # B,N,C
        neighbors = rearrange(neighbors, "b f m k -> (b f) m k")  # B,M,K
        centers = self.cluster(features=feats, neighbors=neighbors)  # B,M,C
        centers = rearrange(centers, "(b f) m c -> b f m c", f=F)  # B,F,M,C

        ref_centers = rearrange(centers[:, :1, :, :].repeat(1, F - 1, 1, 1), "b f m c -> (b f) m c")  # B*(F-1),Mr,C
        ref_centers = rearrange(ref_centers, "b m (e c) -> (b e) m c", e=self.heads)  # B(F-1)H,Mr,D

        src_centers = rearrange(centers[:, 1:, :, :], "b f m c -> (b f) m c")  # B*(F-1),Ms,C
        src_centers = rearrange(src_centers, "b m (e c) -> (b e) m c", e=self.heads)  # B(F-1)H,Ms,D

        ref_center_processed = self.ref_layer(ref_centers)  # B(F-1)H,Mr,D
        src_center_processed = self.src_layer(src_centers)  # B(F-1)H,Ms,D
        if self.no_correlation:
            _, Mr, _ = ref_center_processed.shape
            _, Ms, _ = src_center_processed.shape
            ref_cluster = ref_center_processed.unsqueeze(2).repeat(1, 1, Ms, 1)  # B(F-1)H,Mr,Ms,D
            src_cluster = src_center_processed.unsqueeze(1).repeat(1, Mr, 1, 1)  # B(F-1)H,Mr,Ms,D
        else:
            correlation = torch.sigmoid(
                self.sim_beta + self.sim_alpha * pairwise_cos_sim(ref_centers, src_centers)  # B(F-1)H,Mr,Ms
            )

            BH, Mr, Ms = correlation.shape
            values, indices = correlation.view(BH, Mr * Ms).topk(k=self.max_matches, dim=1)
            mask = torch.zeros_like(correlation).view(BH, Mr * Ms)
            mask = mask.scatter_(1, indices, 1.).view(BH, Mr, Ms)  # B(F-1)H,Mr,Ms
            correlation = correlation * mask  # B(F-1)H,Mr,Ms

            ref_cluster = correlation.unsqueeze(-1) * ref_center_processed.unsqueeze(2)  # B(F-1)H,Mr,Ms,D
            src_cluster = correlation.unsqueeze(-1) * src_center_processed.unsqueeze(1)  # B(F-1)H,Mr,Ms,D
        all_clusters = self.cluster_layer(torch.concat((ref_cluster, src_cluster), dim=-1))

        if self.no_correlation:
            ref_cluster_norm = all_clusters.mean(dim=1)  # B(F-1)H,Ms,D
            src_cluster_norm = ref_cluster_norm.mean(dim=1)  # B(F-1)H,D
        else:
            # ref_cluster_norm = ref_cluster.sum(dim=1) / (correlation.sum(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,Ms,D
            # correlation_sum = correlation.sum(dim=1).unsqueeze(-1)  # B(F-1)H,Ms,1
            # src_weighted = correlation_sum * src_centers  # B(F-1)H,Ms,D
            ref_cluster_norm = all_clusters.sum(dim=1) / (correlation.sum(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,Ms,D
            src_cluster_norm = ref_cluster_norm.sum(dim=1) / (
                    correlation.sum(dim=2).mean(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,D

        out_cluster = rearrange(src_cluster_norm, "(b e) c -> b (e c)", e=self.heads)
        local_score = self.out_layer(out_cluster)
        local_score = torch.sigmoid(local_score).view(B, F - 1)

        return local_score


class CrossSCC(nn.Module):
    def __init__(self, dense_dim, heads, head_dim, mlp_ratio, max_matches, last=False):
        super(CrossSCC, self).__init__()
        self.cluster = ClusterBlock(dim=dense_dim, act_layer=nn.GELU, norm_layer=GroupNorm, mlp_ratio=mlp_ratio,
                                    heads=heads, head_dim=head_dim)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        # self.sigmoid = nn.Sigmoid()
        self.max_matches = max_matches
        self.heads = heads
        self.head_dim = head_dim

        self.ref_layer = Mlp(in_features=self.head_dim, out_features=self.head_dim,
                             hidden_features=int(self.head_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.src_layer = Mlp(in_features=self.head_dim, out_features=self.head_dim,
                             hidden_features=int(self.head_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

        self.last = last
        if self.last:
            self.cluster_layer = Mlp(in_features=self.head_dim * 2, out_features=self.head_dim,
                                     hidden_features=int(self.head_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
        else:
            self.ref_out = Mlp(in_features=self.head_dim, out_features=self.head_dim,
                               hidden_features=int(self.head_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
            self.src_out = Mlp(in_features=self.head_dim, out_features=self.head_dim,
                               hidden_features=int(self.head_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

    def forward(self, ref_feats, src_feats, ref_neighbors, src_neighbors):
        # B, Fr, Nr, C = ref_feats.shape
        # B, Fs, Ns, C = src_feats.shape
        # assert Fr == Fs and Nr == Ns, "the ref and src shapes are not the same"

        # ref_feats = rearrange(ref_feats, "b f n c -> (b f) n c")  # B,N,C
        # src_feats = rearrange(src_feats, "b f n c -> (b f) n c")  # B,N,C
        # ref_neighbors = rearrange(ref_neighbors, "b f m k -> (b f) m k")  # B,M,K
        # src_neighbors = rearrange(src_neighbors, "b f m k -> (b f) m k")  # B,M,K

        ref_centers = self.cluster(features=ref_feats, neighbors=ref_neighbors)  # B,M,C
        # ref_centers = rearrange(ref_centers, "(b f) m c -> b f m c", f=Fr)  # B,F,M,C
        ref_centers = rearrange(ref_centers, "b m (e c) -> (b e) m c", e=self.heads)  # B(F-1)H,Mr,D

        src_centers = self.cluster(features=src_feats, neighbors=src_neighbors)  # B,M,C
        # src_centers = rearrange(src_centers, "(b f) m c -> b f m c", f=Fr)  # B,F,M,C
        src_centers = rearrange(src_centers, "b m (e c) -> (b e) m c", e=self.heads)  # B(F-1)H,Ms,D

        ref_center_processed = self.ref_layer(ref_centers)  # B(F-1)H,Mr,D
        src_center_processed = self.src_layer(src_centers)  # B(F-1)H,Ms,D

        correlation = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(ref_centers, src_centers)  # B(F-1)H,Mr,Ms
        )

        BH, Mr, Ms = correlation.shape
        values, indices = correlation.view(BH, Mr * Ms).topk(k=self.max_matches, dim=1)
        mask = torch.zeros_like(correlation).view(BH, Mr * Ms)
        mask = mask.scatter_(1, indices, 1.).view(BH, Mr, Ms)  # B(F-1)H,Mr,Ms
        correlation = correlation * mask  # B(F-1)H,Mr,Ms

        ref_cluster = correlation.unsqueeze(-1) * ref_center_processed.unsqueeze(2)  # B(F-1)H,Mr,Ms,D
        src_cluster = correlation.unsqueeze(-1) * src_center_processed.unsqueeze(1)  # B(F-1)H,Mr,Ms,D

        if self.last:
            all_clusters = self.cluster_layer(torch.concat((ref_cluster, src_cluster), dim=-1))
            ref_cluster_norm = all_clusters.sum(dim=1) / (correlation.sum(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,Ms,D
            src_cluster_norm = ref_cluster_norm.sum(dim=1) / (
                    correlation.sum(dim=2).mean(dim=1).unsqueeze(-1) + 1)  # B(F-1)H,D
            out_cluster = rearrange(src_cluster_norm, "(b e) c -> b (e c)", e=self.heads)
            return out_cluster, correlation
        else:
            ref_cluster = ref_cluster.mean(dim=2) + self.ref_out(ref_center_processed)
            src_cluster = src_cluster.mean(dim=1) + self.src_out(src_center_processed)
            ref_cluster = rearrange(ref_cluster, "(b e) n c -> b n (e c)", e=self.heads)
            src_cluster = rearrange(src_cluster, "(b e) n c -> b n (e c)", e=self.heads)
            return ref_cluster, src_cluster


class DeepCrossCluster(nn.Module):
    def __init__(self, cfg):
        super(DeepCrossCluster, self).__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        # self.layer1 = CrossSCC(cfg.dense_dim, self.heads, cfg.head_dim, cfg.mlp_ratio, cfg.max_matches, last=False)
        # self.layer2 = CrossSCC(cfg.head_dim*self.heads, self.heads, cfg.head_dim * 2, cfg.mlp_ratio,
        #                        int(cfg.max_matches / 2), last=True)
        self.layer2 = CrossSCC(cfg.dense_dim, self.heads, cfg.head_dim * 2, cfg.mlp_ratio, cfg.max_matches, last=True)
        self.out_layer = Mlp(in_features=self.head_dim * self.heads * 2, out_features=1,
                             hidden_features=int(self.head_dim * cfg.mlp_ratio), act_layer=nn.GELU, drop=0.)

    def forward(self, feats, neighbors):
        """
        Args:
            feats (Tensor): (B, F, N, C)
            neighbors (Tensor): (B, F, M, K)

        Returns:
            final_score: torch.Tensor (B, N)
        """
        B, F, N, C = feats.shape
        ref_feats = rearrange(feats[:, :1, :, :].repeat(1, F - 1, 1, 1), "b f m c -> (b f) m c")  # B*(F-1),Mr,C
        src_feats = rearrange(feats[:, 1:, :, :], "b f m c -> (b f) m c")  # B*(F-1),Ms,C
        ref_neighbors = rearrange(neighbors[0][:, :1, :, :].repeat(1, F - 1, 1, 1),
                                  "b f m c -> (b f) m c")  # B*(F-1),Mr,C
        src_neighbors = rearrange(neighbors[0][:, 1:, :, :], "b f m c -> (b f) m c")  # B*(F-1),Ms,C
        # ref_cluster, src_cluster = self.layer1(ref_feats, src_feats, ref_neighbors, src_neighbors)
        # ref_neighbors = rearrange(neighbors[1][:, :1, :, :].repeat(1, F - 1, 1, 1),
        #                           "b f m c -> (b f) m c")  # B*(F-1),Mr,C
        # src_neighbors = rearrange(neighbors[1][:, 1:, :, :], "b f m c -> (b f) m c")  # B*(F-1),Ms,C
        # out_cluster, correlation = self.layer2(ref_feats=ref_cluster, src_feats=src_cluster,
        #                                         ref_neighbors=ref_neighbors, src_neighbors=src_neighbors)
        out_cluster, correlation = self.layer2(ref_feats, src_feats, ref_neighbors, src_neighbors)
        local_score = self.out_layer(out_cluster)
        local_score = torch.sigmoid(local_score).view(B, F - 1)
        return local_score
