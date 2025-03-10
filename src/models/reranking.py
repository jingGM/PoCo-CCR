import torch
from torch import nn
import torch.nn.functional as F

from src.models.cross_models.cocs_attn import CoCsAttn
from src.models.cross_models.cocs_total import CoCsTotal
from src.models.cross_models.cocs_cross import CrossCluster, DeepCrossCluster
from src.models.cross_models.r2former import R2Former
from src.models.cross_models.cross_attention import Transformer
from src.functions import Distance
# from src.models.cocs_cross import
from utils.config import CrossType, ReducerType
from utils.functions import single_fps_knn, to_device


# class MatchingScoreModel(nn.Module):
#     def __init__(self, output_dim, score_method=ReducerType.max):
#         super(MatchingScoreModel, self).__init__()
#         self.cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-8)
#         self.process_weights = nn.Sequential(nn.Linear(output_dim, output_dim), nn.GELU(),
#                                              nn.Linear(output_dim, output_dim))
#         self.similarity = nn.Sequential(nn.Linear(output_dim, output_dim), nn.GELU(),
#                                         nn.Linear(output_dim, output_dim))
#         self.score_output = nn.Sequential(nn.Linear(output_dim, int(output_dim / 2)), nn.GELU(),
#                                           nn.Linear(int(output_dim / 2), int(output_dim / 4)), nn.GELU(),
#                                           nn.Linear(int(output_dim / 4), 1), nn.Sigmoid())
#
#         self.score_method = score_method
#
#     def forward(self, ref_fts, src_fts, attn):
#         """
#         Args:
#             ref_fts: B,N,C
#             src_fts: B,M,C
#             attn: List:[ref, src],
#         Returns:
#             score: B,N
#         """
#         ref_feats_norm = F.normalize(ref_fts, p=2, dim=2)  # B,N,C
#         src_feats_norm = F.normalize(src_fts, p=2, dim=2)  # B,M,C
#         similarity = self.cos_similarity(ref_feats_norm, src_feats_norm)
#
#         weights = self.process_weights(ref_fts)  # BNC
#
#         similarity_matrix = torch.einsum('bnc,bmc->bnm', ref_feats_norm, src_feats_norm)
#         indices = torch.argmax(similarity_matrix, dim=2)  # BN
#         selected = torch.stack([src_fts[i][indices[i]] for i in range(len(src_fts))], dim=0)
#         selected_src = self.similarity(selected)  # BNC
#
#         score = self.score_output(weights + selected_src)
#
#         if self.score_method == ReducerType.max:
#             return torch.max(score.squeeze(-1), dim=1)
#         elif self.score_method == ReducerType.mean:
#             return torch.mean(score.squeeze(-1), dim=1)
#         else:
#             raise ValueError("the score type is not defined")


class RerankModel(nn.Module):
    def __init__(self, cfg, device):
        super(RerankModel, self).__init__()
        self.model_type = cfg.cross_type
        self.device = device
        self.ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # fixed sin-cos embedding

        if self.model_type == CrossType.cocs:
            self.model = CoCsTotal(cfg=cfg.cocs)
        elif self.model_type == CrossType.cross_cocs:
            self.model = CrossCluster(cfg=cfg.cocs_cross)
        elif self.model_type == CrossType.deep_cross_cocs:
            self.downsampled_num = cfg.deep_cross.downsampled_num
            self.knn = cfg.deep_cross.knn
            self.model = DeepCrossCluster(cfg=cfg.deep_cross)
        elif self.model_type == CrossType.r2former:
            self.model = R2Former(cfg=cfg.r2former)
        elif self.model_type == CrossType.cross_attn:
            self.model = Transformer(cfg=cfg.cocs_cross)
        elif self.model_type == CrossType.cocs_attn:
            self.model = CoCsAttn(cfg=cfg.cocs_cross)
        else:
            raise ValueError("the model type is not defined")

        self.distance = Distance(distance_type=cfg.distance_type, distance=False)

    def forward(self, fine_input):
        coarse_embeddings, fts, geos, neighbors, similarities = fine_input
        if self.model_type == CrossType.cocs:
            local_score = self.model(ref_feats=fts[:, :1, :, :], ref_geos=geos[:, :1, :, :],
                                     src_geos=geos[:, 1:, :, :], src_feats=fts[:, 1:, :, :])
        elif self.model_type == CrossType.cross_cocs:
            local_score = self.model(feats=fts, neighbors=neighbors)
        elif self.model_type == CrossType.deep_cross_cocs:
            B, F, N, C = geos.shape
            geos, knn_indices_0 = single_fps_knn(points=geos.view(B * F, N, C), fps_num=self.downsampled_num[0],
                                                 k_num=self.knn[0])
            B0, N0, C0 = knn_indices_0.shape
            geos, knn_indices_1 = single_fps_knn(points=geos, fps_num=self.downsampled_num[1], k_num=self.knn[1])
            B1, N1, C1 = knn_indices_1.shape
            local_score = self.model(feats=fts, neighbors=[to_device(knn_indices_0.view(B, F, N0, C0), device=self.device),
                                                           to_device(knn_indices_1.view(B, F, N1, C1), device=self.device)])
        elif self.model_type == CrossType.r2former:
            local_score = self.model(feats=fts, geometries=geos, similarities=similarities)
        elif self.model_type == CrossType.cross_attn:
            local_score = self.model(feats=fts)
        elif self.model_type == CrossType.cocs_attn:
            local_score = self.model(feats=fts)
        else:
            raise ValueError("the model type is not defined")

        global_query = coarse_embeddings[:, :1, :]
        global_values = coarse_embeddings[:, 1:, :].contiguous()
        B, N, C = global_values.shape
        global_score = self.distance(global_query.repeat(1, N, 1).view(B * N, C),
                                     global_values.view(B * N, C)).view(B, N)
        clamped_ratio = self.ratio.clamp(min=0.1, max=0.9)
        final_score = global_score.detach() * clamped_ratio + local_score.detach() * (1 - clamped_ratio)

        return final_score, local_score

    def predict(self, coarse_embeddings, fts, geos, neighbors, similarities):
        if self.model_type == CrossType.cocs:
            local_score = self.model(ref_feats=fts[:, :1, :, :], ref_geos=geos[:, :1, :, :],
                                     src_geos=geos[:, 1:, :, :], src_feats=fts[:, 1:, :, :])
        elif self.model_type == CrossType.cross_cocs:
            local_score = self.model(feats=fts, neighbors=neighbors)
        elif self.model_type == CrossType.deep_cross_cocs:
            B, F, N, C = geos.shape
            geos, knn_indices_0 = single_fps_knn(points=geos.view(B * F, N, C), fps_num=self.downsampled_num[0],
                                                 k_num=self.knn[0])
            B0, N0, C0 = knn_indices_0.shape
            geos, knn_indices_1 = single_fps_knn(points=geos, fps_num=self.downsampled_num[1], k_num=self.knn[1])
            B1, N1, C1 = knn_indices_1.shape
            local_score = self.model(feats=fts, neighbors=[to_device(knn_indices_0.view(B, F, N0, C0), device=self.device),
                                                           to_device(knn_indices_1.view(B, F, N1, C1), device=self.device)])
        elif self.model_type == CrossType.r2former:
            local_score = self.model(feats=fts, geometries=geos, similarities=similarities)
        elif self.model_type == CrossType.cross_attn:
            local_score = self.model(feats=fts)
        elif self.model_type == CrossType.cocs_attn:
            local_score = self.model(feats=fts)
        else:
            raise ValueError("the model type is not defined")

        global_query = coarse_embeddings[:, :1, :]
        global_values = coarse_embeddings[:, 1:, :].contiguous()
        B, N, C = global_values.shape
        global_score = self.distance(global_query.repeat(1, N, 1).view(B * N, C),
                                     global_values.view(B * N, C)).view(B, N)
        clamped_ratio = self.ratio.clamp(min=0.1, max=0.9)
        final_score = global_score.detach() * clamped_ratio + local_score.detach() * (1 - clamped_ratio)
        # if self.model_type == CrossType.cross_cocs:
        #     if self.model.no_correlation:
        #         clamped_ratio = 0.9
        #         local_score = clamped_ratio * global_score.detach() + local_score.detach() * (1.0-clamped_ratio)
        return final_score, local_score

# if __name__ == "__main__":
#     import pickle
#     from utils.config import CrossModelConfig, CoCsConfig
#
#     CrossModelConfig.cocs.dense_dim = CrossModelConfig.cocs_cross.dense_dim = CrossModelConfig.geotransformer.input_dim = \
#     CoCsConfig.embed_dims[1]
#     CrossModelConfig.cross_type = CrossType.cross_cocs
#
#     model = RerankModel(CrossModelConfig).cuda()
#
#     with open("/home/jing/Documents/work/pr_pts/test_1.pkl", "rb") as f:
#         all_data = pickle.load(f)
#
#     fine_input = all_data[0]
#     output = model(fine_input)
