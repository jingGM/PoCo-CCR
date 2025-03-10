import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.clusters import BasicBlocks  # , CrossAttention, CocsCrossModel
from src.models.model_utils import GroupNorm
from utils.config import PointReducerConfig, CoCsConfig
from src.models.pointconvformer import PCFLayer


class PointReducer(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    """

    def __init__(self, config: PointReducerConfig, norm_layer=None):
        super(PointReducer, self).__init__()
        self.pcf = PCFLayer(in_channel=config.in_channel, out_channel=config.out_channel,
                            weightnet_output=config.weightnet_output, num_heads=config.num_heads,
                            guidance_feat_len=config.guidance_feat_len,
                            batch_norm=config.batch_norm, viewpoint_invariant=config.viewpoint_invariant,
                            attention_type=config.attention_type, layer_norm_guidance=config.layer_norm_guidance)
        self.pointconv_res = nn.ModuleList()
        for i in range(config.residual_blocks):
            self.pointconv_res.append(PCFLayer(in_channel=config.out_channel, out_channel=config.out_channel,
                                               weightnet_output=config.weightnet_output, num_heads=config.num_heads,
                                               guidance_feat_len=config.guidance_feat_len,
                                               batch_norm=config.batch_norm,
                                               viewpoint_invariant=config.viewpoint_invariant,
                                               attention_type=config.attention_type,
                                               layer_norm_guidance=config.layer_norm_guidance))
        self.norm = norm_layer(config.out_channel) if norm_layer else nn.Identity()

    def forward(self, dense_geos, sparse_geos, dense_feats, down_nei_inds, sparse_nei_inds):
        sparse_feat, weightNetInput = self.pcf(dense_geos=dense_geos, sparse_geos=sparse_geos,
                                               dense_feats=dense_feats, nei_inds=down_nei_inds)
        for res_block in self.pointconv_res:
            sparse_feat, vi_features = res_block(dense_geos=sparse_geos, sparse_geos=sparse_geos,
                                                 dense_feats=sparse_feat, nei_inds=sparse_nei_inds)
        return self.norm(sparse_feat)


class ContextCluster(nn.Module):
    def __init__(self, cfg: CoCsConfig):
        """
        Args:
            cfg: the configuration of the context cluster
        """
        super(ContextCluster, self).__init__()
        self.output_layer = cfg.output_layer
        self.embedding_reducer = PointReducer(config=PointReducerConfig(cfg=cfg.reducer), norm_layer=None)
        # self.center_id = cfg.center_index
        # set the main block in network
        network = []
        self.norm_layers = nn.ModuleList([GroupNorm(cfg.embed_dims[0])])
        for i in range(len(cfg.layers)):
            network.append(BasicBlocks(dim=cfg.embed_dims[i], index=i, layers=cfg.layers,
                                       mlp_ratio=cfg.mlp_ratios[i], act_layer=nn.GELU, norm_layer=GroupNorm,
                                       drop_rate=0, drop_path_rate=0,
                                       use_layer_scale=cfg.use_layer_scale,
                                       layer_scale_init_value=cfg.layer_scale_init_value,
                                       heads=cfg.heads[i], head_dim=cfg.head_dim[i]
                                       ))
            if i < len(cfg.layers) - 1:
                reducer_config = PointReducerConfig(cfg=cfg.reducer)
                reducer_config.in_channel = cfg.embed_dims[i]
                reducer_config.out_channel = cfg.embed_dims[i + 1]
                network.append(PointReducer(config=reducer_config))
                self.norm_layers.append(GroupNorm(cfg.embed_dims[i + 1]))

        self.network = nn.ModuleList(network)

    def forward_clusters(self, geometries, features, neighbors, self_neightbors):
        outs = []
        last_similarity = None
        for idx, block in enumerate(self.network):
            start_time = time.time()
            torch.cuda.empty_cache()
            block_idx = idx // 2 + 1
            if idx % 2 == 0:  # CoCs
                features, centers, similarities = block(features=features, neighbors=neighbors[block_idx])
                features = self.norm_layers[block_idx - 1](features)
                if self.output_layer == 0:
                    outs = None
                elif self.output_layer == -1:
                    outs = [features, geometries[block_idx], neighbors[block_idx], last_similarity]
                elif self.output_layer == block_idx - 1:
                    outs = [features, geometries[block_idx], neighbors[block_idx], last_similarity]
                else:
                    pass
                last_similarity = similarities
            else:  # reducer
                features = block(dense_geos=geometries[block_idx], sparse_geos=geometries[block_idx + 1],
                                 dense_feats=features,
                                 down_nei_inds=neighbors[block_idx], sparse_nei_inds=self_neightbors[block_idx])
            # print("b{}: {:.4f}".format(idx, time.time() - start_time), end=";  ")
            torch.cuda.empty_cache()
        return outs, features

    def forward(self, geometries, features, neighbors, self_neightbors):
        # input embedding
        start_time = time.time()
        features = self.embedding_reducer(dense_geos=geometries[0],
                                          sparse_geos=geometries[1],
                                          dense_feats=features,
                                          down_nei_inds=neighbors[0],
                                          sparse_nei_inds=self_neightbors[0])
        # print("ce: {:.4f}".format(time.time() - start_time), end=";  ")
        # through backbone
        outs, features = self.forward_clusters(geometries=geometries, features=features,
                                               neighbors=neighbors, self_neightbors=self_neightbors)
        return outs, features


if __name__ == "__main__":
    import pickle
    from utils.functions import SampleKNN
    import numpy as np
    from utils.config import DownSamplerConfig, DataDict, PointReducerConfig, CoCsConfig
    import random


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

    geos = data[:, :, :6]
    pts_fts = data[:, :, 6:]
    down_sampler = SampleKNN(DownSamplerConfig)
    all_points, edges, self_edges, _, _, _ = down_sampler.downsample_knn(points=data)
    all_points = [pts.to(torch.float32) for pts in all_points]
    edges = [edge.to(torch.float32) for edge in edges]
    self_edges = [edge.to(torch.float32) for edge in self_edges]
    pcf = ContextCluster(CoCsConfig)
    output = pcf(geometries=all_points, neighbors=edges, self_neightbors=self_edges,
                 features=torch.from_numpy(pts_fts).to(torch.float32))
    print("test")
