import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.model_utils import trunc_normal_, GroupNorm, DropPath
from utils.functions import index_batched_points


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [N, C]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features), act_layer(),
            nn.Linear(hidden_features, out_features), nn.Dropout(drop)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, heads=4, head_dim=24):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super(Cluster, self).__init__()
        self.heads = heads
        self.head_dim = head_dim

        self.query = nn.Sequential(nn.Linear(dim, heads * head_dim), nn.LeakyReLU(0.2))
        self.query_norm = GroupNorm(num_channels=heads * head_dim, num_groups=1)

        self.value = nn.Sequential(nn.Linear(dim, heads * head_dim), nn.LeakyReLU(0.2))
        self.value_norm = GroupNorm(num_channels=heads * head_dim, num_groups=1)

        self.proj = nn.Linear(heads * head_dim, out_dim)  # for projecting channel number
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

    def _calculate_block_fts(self, q_fts, v_fts, neighbors):
        vx_query_knn_fts = index_batched_points(points=q_fts, idx=neighbors)  # B,Mq,K,C
        vx_query_fts = torch.mean(vx_query_knn_fts, dim=2)  # B,Mq,C

        vx_value_knn_fts = index_batched_points(points=v_fts, idx=neighbors)  # B,Mv,K,C
        vx_value_fts = torch.mean(vx_value_knn_fts, dim=2)  # B,Mv,C

        vx_query_fts = rearrange(vx_query_fts, "b m (e c) -> (b e) m c", e=self.heads)  # BH,M,D
        vx_value_fts = rearrange(vx_value_fts, "b m (e c) -> (b e) m c", e=self.heads)  # BH,M,D
        q_fts = rearrange(q_fts, "b n (e c) -> (b e) n c", e=self.heads)  # BH,N,D
        v_fts = rearrange(v_fts, "b n (e c) -> (b e) n c", e=self.heads)  # BH,N,D

        sim = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(vx_query_fts, q_fts)  # [BH,Mq,Nq]
        )
        similarity = rearrange(sim.clone(), "(b e) m n -> b e m n", e=self.heads)
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)  # B,1,Nq
        mask = torch.zeros_like(sim)  # binary #[B,Mq,Nq]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask  # [B,Mq,Nq]

        value_sim = v_fts.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)  # B,Mq,Nv,C
        centers_out = (value_sim.sum(dim=2) + vx_value_fts) / (sim.sum(dim=-1, keepdim=True) + 1.0)  # B,M,C

        # dispatch step, return to each point in a cluster: BM1C * BMNq1 +-> BNC
        pts_out = (centers_out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # B,Nq,C

        pts_out = rearrange(pts_out, "(b e) n c -> b n (e c)", e=self.heads)
        points_fts = self.proj(pts_out)
        return points_fts, centers_out, similarity

    def forward(self, features, neighbors):
        # start_time = time.time()
        query_fts = self.query(features)
        query_fts_nm = self.query_norm(query_fts)  # B,N,C

        value_fts = self.value(features)
        value_fts_nm = self.value_norm(value_fts)  # B,N,C
        # print("pre: {:.4f}".format(time.time() - start_time), end=";  ")
        return self._calculate_block_fts(q_fts=query_fts_nm, v_fts=value_fts_nm, neighbors=neighbors)


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, heads=4, head_dim=24):
        super(ClusterBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Cluster(dim=dim, out_dim=dim, heads=heads, head_dim=head_dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, features, neighbors):
        out, centers, sim = self.token_mixer(features=self.norm1(features), neighbors=neighbors)
        if self.use_layer_scale:
            features = features + self.drop_path(self.layer_scale_1 * out)
            features = features + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(features)))
        else:
            features = features + self.drop_path(out)
            features = features + self.drop_path(self.mlp(self.norm2(features)))
        return features, centers, sim


class BasicBlocks(nn.Module):
    def __init__(self, dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 heads=4, head_dim=24):
        super().__init__()
        self.blocks = nn.ModuleList()
        for block_idx in range(layers[index]):
            block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
            self.blocks.append(ClusterBlock(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                heads=heads, head_dim=head_dim
            ))
    
    def forward(self, features, neighbors):
        for idx, block in enumerate(self.blocks):
            features, centers, similarities = block(features=features, neighbors=neighbors)
        return features, centers, similarities


if __name__ == "__main__":
    import random
    import pickle
    from utils.functions import SampleKNN
    import numpy as np
    from utils.config import DownSamplerConfig, DataDict, PointReducerConfig, CoCsConfig

    def select_pts(point, num=2000):
        if point.shape[0] > num:
            indices = list(range(point.shape[0]))
            random.shuffle(indices)
            selected_indices = indices[:num]
            point = point[selected_indices]
        return point

    with open("/home/ANT.AMAZON.COM/jliangmd/Documents/pr_pts/debug/test_data.pkl", "rb") as f:
        all_pts = pickle.load(f)

    data = all_pts[DataDict.pts_features]
    data = np.array([select_pts(pts) for pts in data])

    geos = data[:, :, :-3]
    pts_fts = data[:, :, -3:]
    down_sampler = SampleKNN(DownSamplerConfig)
    all_points, edges, self_edges, _, _, _ = down_sampler.downsample_knn(points=data)

    cluster = BasicBlocks(dim=3, layers=[2, 2, 6, 2], index=0)
    cluster(features=torch.from_numpy(pts_fts).to(torch.float32),
            neighbors=edges[0])
    print("test")