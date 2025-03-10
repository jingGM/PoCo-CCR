import torch
from flopth import flopth
from torch import nn
import torch.nn.functional as F
import time
import pickle
import numpy as np
from einops import rearrange
from src.models.context_cluster import ContextCluster, PointReducer
from src.models.reranking import RerankModel

from utils.config import DataDict, ModelType, PointReducerConfig
from utils.functions import trunc_normal_, to_device, SampleKNN


class Model(nn.Module):
    def __init__(self, cfg, device):
        super(Model, self).__init__()
        self.device = device
        assert len(cfg.down_sampler.down_sample_num) == 1 + len(
            cfg.cocs_config.layers), "the layers of down sampler should be 1 + CoCs layers"
        self.down_sampler = SampleKNN(cfg=cfg.down_sampler)
        self.fix_global_model = cfg.fix_global_model

        self.model_type = cfg.model_type
        self.geo_dim = cfg.geo_dim
        self.feature_dim = cfg.feature_dim
        if self.model_type == ModelType.coarse_matching:
            self.model = ContextCluster(cfg=cfg.cocs_config)
            self.coarse_model = PointReducer(PointReducerConfig(cfg.coarse_reducer))
        elif self.model_type == ModelType.fine_matching:
            self.forward_fine_matching = RerankModel(cfg.rerank_config, device)
        elif self.model_type == ModelType.all_matching:
            self.model = ContextCluster(cfg=cfg.cocs_config)
            self.coarse_model = PointReducer(PointReducerConfig(cfg.coarse_reducer))
            self.forward_fine_matching = RerankModel(cfg.rerank_config, device)
            if self.fix_global_model:
                self._fix_paramters()
        else:
            raise ValueError("the model type {} is not defined".format(cfg.model_type))

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _fix_paramters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.coarse_model.parameters():
            param.requires_grad = False

    def forward_coarse_matching(self, cocs_output, dense_geos):
        """
        Args:
            cocs_output: B,M,C
            dense_geos: B,M,D
        """
        start_time = time.time()
        B, M, D = dense_geos.shape
        current_sparse_geos = torch.mean(dense_geos, dim=1, keepdim=True)  # B,1,C
        down_nei = to_device(
            torch.from_numpy(np.arange(start=0, stop=M, step=1)).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1),
            device=self.device)  # B,1,M
        all_features = self.coarse_model(dense_geos=dense_geos,
                                         sparse_geos=current_sparse_geos,
                                         dense_feats=cocs_output,
                                         down_nei_inds=down_nei,
                                         sparse_nei_inds=[0] * B)  # B,1,C
        current_embeddeds = F.normalize(all_features.squeeze(1), p=2, dim=-1)
        # print("cl: {:.4f}".format(time.time() - start_time), end=";  ")
        return current_embeddeds

    def process_data(self, input_data):
        if DataDict.binary_label in input_data.keys():
            all_labels = to_device(torch.tensor(input_data[DataDict.binary_label], dtype=torch.float32),
                                   device=self.device)
        else:
            all_labels = None

        if DataDict.pts_features in input_data.keys():
            all_features = input_data[DataDict.pts_features]
            B, N, M, C = all_features.shape
            all_features = all_features.view(B * N, M, C)

            geos = all_features[:, :, :self.geo_dim]
            features = to_device(all_features[:, :, self.feature_dim:].to(torch.float32), device=self.device)
            all_geos, edges, self_edges, _, _, _ = self.down_sampler.downsample_knn(points=geos)
            all_geos = [to_device(geos.to(torch.float32), device=self.device) for geos in all_geos]
            edges = [to_device(edg, device=self.device) for edg in edges]
            self_edges = [to_device(self_edg, device=self.device) for self_edg in self_edges]
            return all_labels, all_geos, edges, self_edges, features, B, N
        else:
            global_fts = to_device(torch.tensor(input_data[DataDict.global_descriptor], dtype=torch.float32),
                                   device=self.device)
            fts = to_device(torch.tensor(input_data[DataDict.features], dtype=torch.float32), device=self.device)
            geos = to_device(torch.tensor(input_data[DataDict.geometrics], dtype=torch.float32), device=self.device)
            neighbors = to_device(torch.tensor(input_data[DataDict.neighbors], dtype=torch.float32), device=self.device)
            similarities = to_device(torch.tensor(input_data[DataDict.similarities], dtype=torch.float32), device=self.device)
            return all_labels, [global_fts, fts, geos, neighbors, similarities]

    def _compose_fine_matching_input(self, coarse_embeddings, fine_fts, B, N):
        BN, C = coarse_embeddings.shape
        global_fts = coarse_embeddings.view(B, N, C)

        BN, K, D = fine_fts[0].shape
        fts = fine_fts[0].view(B, N, K, D)
        BN, K, G = fine_fts[1].shape
        geos = fine_fts[1].view(B, N, K, G)
        BN, M, K1 = fine_fts[2].shape
        neighbors = fine_fts[2].view(B, N, M, K1)
        BN, H, M, K = fine_fts[3].shape
        similarities = fine_fts[3].view(B, N, H, M, K)
        return [global_fts, fts, geos, neighbors, similarities]

    def predict(self, input_data):
        # dp_time = time.time()
        # print("dp: {:.4f}".format(time.time() - dp_time), end=";  ")

        coarse_embeddings, final_score, local_score, fine_input = None, None, None, None
        if self.model_type == ModelType.coarse_matching:
            labels, all_geos, edges, self_edges, features, B, N = self.process_data(input_data)
            fine_fts, cocs_output = self.model(geometries=all_geos, neighbors=edges, self_neightbors=self_edges,
                                               features=features)
            coarse_embeddings = self.forward_coarse_matching(cocs_output=cocs_output, dense_geos=all_geos[-2])
            if fine_fts is None:
                fine_input = None
            else:
                fine_input = self._compose_fine_matching_input(coarse_embeddings=coarse_embeddings, fine_fts=fine_fts,
                                                               B=B, N=N)
        elif self.model_type == ModelType.fine_matching:
            labels, fine_input = self.process_data(input_data)
            final_score, local_score = self.forward_fine_matching(fine_input=fine_input)
        elif self.model_type == ModelType.all_matching:
            labels, all_geos, edges, self_edges, features, B, N = self.process_data(input_data)
            fine_fts, cocs_output = self.model(geometries=all_geos, neighbors=edges, self_neightbors=self_edges,
                                               features=features)
            coarse_embeddings = self.forward_coarse_matching(cocs_output=cocs_output, dense_geos=all_geos[-2])

            fine_input = self._compose_fine_matching_input(coarse_embeddings=coarse_embeddings, fine_fts=fine_fts,
                                                           B=B, N=N)
            final_score, local_score = self.forward_fine_matching(fine_input=fine_input)
        else:
            raise ValueError("the model type {} is not defined".format(self.model_type))
        return coarse_embeddings, final_score, local_score, fine_input, labels

    def forward_recall_features(self, input_data):
        frame_ids = to_device(input_data[DataDict.frame_id].view(-1, 2), device=self.device)
        labels, all_geos, edges, self_edges, features, B, N = self.process_data(input_data)
        start_time = time.time()
        fine_fts, cocs_output = self.model(geometries=all_geos, features=features, neighbors=edges,
                                           self_neightbors=self_edges)
        coarse_embeddings = self.forward_coarse_matching(cocs_output=cocs_output, dense_geos=all_geos[-2])
        step_time = time.time() - start_time
        if fine_fts is None:
            fine_input = None
        else:
            fine_input = self._compose_fine_matching_input(coarse_embeddings=coarse_embeddings, fine_fts=fine_fts,
                                                           B=B, N=N)

        # with open("/home/jing/Documents/work/pr_pts/test_1.pkl", 'wb') as handle:
        #     pickle.dump([fine_input, coarse_embeddings], handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            DataDict.frame_id: frame_ids,
            DataDict.pts_coarse_embeds: coarse_embeddings,
            DataDict.fine_input: fine_input,
            DataDict.step_time: step_time
        }

    def forward(self, input_data):
        frame_ids = to_device(input_data[DataDict.frame_id].view(-1, 2), device=self.device)
        coarse_embeddings, final_score, local_score, fine_input, labels = self.predict(input_data=input_data)
        return {
            DataDict.frame_id: frame_ids,
            DataDict.pts_coarse_embeds: coarse_embeddings,
            DataDict.local_score: local_score,
            DataDict.final_score: final_score,
            DataDict.fine_input: fine_input,
            DataDict.binary_label: labels
        }


def get_model(config, device):
    return Model(cfg=config, device=device)

