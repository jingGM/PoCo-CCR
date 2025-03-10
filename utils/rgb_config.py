from os.path import join
import numpy as np
from easydict import EasyDict as edict


class InputType:
    image = "image"
    points = "points"

    image_rgb = "image_rgb"
    image_rgbd = "image_rgbd"

    poins_fused = "poins_fused"
    points_single = "points_single"


#########################################################################
# Configuration of Models
#########################################################################
class PointReducerConfig:
    def __init__(self, cfg):
        self.image = edict()
        self.image.stride = cfg.image.stride
        self.image.patch_size = cfg.image.patch_size
        self.image.padding = cfg.image.padding
        self.image.in_chans = cfg.image.in_chans
        self.image.embed_dim = cfg.image.embed_dim


class ModelType:
    coarse_matching = "coarse matching"
    fine_matching = "fine_matching"
    all_matching = "all_matching"


class CrossType:
    cocs = "cocs"
    single_cross_attention = "single_cross_attention"
    two_cross_attention = "two_cross_attention"


class CoarseMatchingOutputType:
    conv = "conv"
    pool = "pool"


CoCsConfig = edict()
CoCsConfig.center_index = 1
CoCsConfig.output_layers = -1
CoCsConfig.use_layer_scale = True
CoCsConfig.layer_scale_init_value = 1e-5
CoCsConfig.downsamples = [True, True, True, True, True]
CoCsConfig.mlp_ratios = [8, 8, 4, 4]
CoCsConfig.proposal_w = [2, 2, 2, 2]
CoCsConfig.proposal_h = [2, 2, 2, 2]
CoCsConfig.fold_w = [8, 4, 2, 1]
CoCsConfig.fold_h = [8, 4, 2, 1]
CoCsConfig.head_dim = [32, 32, 32, 32]
CoCsConfig.layers = [2, 2, 6, 2]             # [4, 4, 12, 4]
CoCsConfig.embed_dims = [64, 128, 320, 512]  # [64, 128, 320, 512]
CoCsConfig.heads = [4, 4, 8, 8]              # [6, 6, 12, 12]

CoCsConfig.embedding_reducer = edict()
CoCsConfig.embedding_reducer.image = edict()
CoCsConfig.embedding_reducer.image.stride = 4
CoCsConfig.embedding_reducer.image.patch_size = 4
CoCsConfig.embedding_reducer.image.padding = 0
CoCsConfig.embedding_reducer.image.in_chans = 5
CoCsConfig.embedding_reducer.image.embed_dim = 64

CoCsConfig.downstream_reducer = edict()
CoCsConfig.downstream_reducer.image = edict()
CoCsConfig.downstream_reducer.image.stride = 2
CoCsConfig.downstream_reducer.image.patch_size = 3
CoCsConfig.downstream_reducer.image.padding = 1
CoCsConfig.downstream_reducer.image.in_chans = 3
CoCsConfig.downstream_reducer.image.embed_dim = 64

CrossModelConfig = edict()
CrossModelConfig.cross_type = CrossType.cocs
CrossModelConfig.dim = CoCsConfig.embed_dims[-1]
CrossModelConfig.mlp_ratio = 4.0
CrossModelConfig.fold_w = 1
CrossModelConfig.fold_h = 1
CrossModelConfig.proposal_w = 2
CrossModelConfig.proposal_h = 2
CrossModelConfig.heads = 6
CrossModelConfig.head_dim = 32
CrossModelConfig.fold_n = 1
CrossModelConfig.center_num = 4
CrossModelConfig.layers = 2
CrossModelConfig.cross_selection_type = "max"

CrossModelConfig.max_pool = (15, 20)
CrossModelConfig.hidden = 128

ModelConfig = edict()
ModelConfig.model_type = ModelType.coarse_matching
ModelConfig.cocs_config = CoCsConfig
ModelConfig.last_type = CoarseMatchingOutputType.conv
ModelConfig.last_dim = 256
ModelConfig.last_kernel = (15, 20)
ModelConfig.last_stride = (1, 1)
ModelConfig.max_pool = (15, 20)  # (7,7)