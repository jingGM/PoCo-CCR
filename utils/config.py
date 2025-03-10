from os.path import join
import numpy as np
from easydict import EasyDict as edict
from utils.rgb_config import ModelConfig as RGBModelConfig

#########################################################################
# Configuration of Metrics and Loss
#########################################################################
class LossNames:
    loss = "loss"
    coarse_loss = "coarse_loss"
    fine_loss = "fine_loss"
    false_pos = "false_pos"
    false_neg = "false_neg"
    precision = "precision"
    neg_precision = "neg_precision"

    triplet_loss = "triplet_loss"
    circle_loss = "circle_loss"


class DistanceType:
    cos = "cos"
    l1 = "l1"
    l2 = "l2"


LossConfig = edict()
LossConfig.distance_type = DistanceType.cos
LossConfig.triplet_ratio = 1000.0
LossConfig.circle_ratio = 1.0
LossConfig.margin = 0.2
LossConfig.circular_gamma = 1.0
LossConfig.reduction = 'mean'

#########################################################################
# Configuration of Dataset and Data Loader
#########################################################################
class DataDict:
    pts_features = "pts_features"
    frame_id = "frame_id"  # [scene frame]
    binary_label = "binary label"  # single frame depth image
    rgbd_im = "rgbd_im"  # single frame depth image

    pts_coarse_embeds = "pts_coarse_embeds"  # the global embeddings
    local_score = "local_score"  # the global embeddings
    final_score = "final_score"  # the global embeddings
    fine_input = "fine_input"  # the Cocs features

    global_descriptor = "global_descriptor"
    features = "features"
    geometrics = "geometrics"
    neighbors = "neighbors"
    similarities = "similarities"
    step_time = "step_time"


class DatasetType:
    training = "training"  # dataset for training
    test = "testing"  # dataset for testing
    evaluation = "evaluation"  # dataset for evaluation
    sanity_check = "sanity"  # dataset for debugging, only one data


class FrameInfo:
    poses = "poses"
    posIds = "positives"
    negIds = "negatives"
    simScenes = "similar_scenes"


class RecallDataType:
    negative_key_frames = "negative_key_frames"
    positive_key_frames = "positive_key_frames"
    distance = "distance"


class PointFolder:
    fused = "fused"
    single = "single"


DatasetConfig = edict()  # Configuration of data loaders
DatasetConfig.name = ""
DatasetConfig.root = "/home/ANT.AMAZON.COM/jliangmd/Documents/data/ScanNetAll"
DatasetConfig.type = DatasetType.training
DatasetConfig.point_folder = PointFolder.fused

DatasetConfig.num_workers = 16
DatasetConfig.batch_size = 1
DatasetConfig.shuffle = False
DatasetConfig.distributed = False

DatasetConfig.use_hard_positive = True
DatasetConfig.use_color = True
DatasetConfig.use_normals = True
DatasetConfig.use_semantic = False
DatasetConfig.use_geos = True

DatasetConfig.max_points_num = 3000

DatasetConfig.num_pos_samples = 3
DatasetConfig.num_neg_same_scenes_samples = 1  # 7
DatasetConfig.num_neg_other_scenes_samples = 2

DatasetConfig.instance_size = 8
DatasetConfig.key_frame_type = 8
# DatasetConfig.recall_type = DatasetType.test
DatasetConfig.recall_database = RecallDataType.negative_key_frames

DatasetConfig.distance_type = DistanceType.cos
DatasetConfig.negative_mining_num = 20
DatasetConfig.negative_mining_num_ratio = 0.8

DatasetConfig.max_iteration = 50000

#########################################################################
# Configuration of Models
#########################################################################
class ModelType:
    coarse_matching = "coarse matching"
    fine_matching = "fine_matching"
    all_matching = "all_matching"


class KNNMethod:
    skl = "sklearn"
    fai = "faiss"


class SampleMethod:
    fps = "farthest point sampling"
    dvx = "dynamic voxelization"


class CrossType:
    cocs = "cocs"
    cross_cocs = "cross cocs"
    deep_cross_cocs = "deep cross cocs"
    r2former = "r2former"
    cross_attn = "cross_attn"
    cocs_attn = "cocs_attn"


class AttentionType:
    original = "qk"
    subtraction = "subtraction"


class ReducerType:
    max = "max"
    mean = "mean"


class CoarseMatchingOutputType:
    conv = "conv"
    pool = "pool"


class PointReducerConfig:
    def __init__(self, cfg):
        self.in_channel = cfg.in_channel
        self.out_channel = cfg.out_channel
        self.weightnet_output = cfg.weightnet_output
        self.num_heads = cfg.num_heads
        self.guidance_feat_len = cfg.guidance_feat_len
        self.batch_norm = cfg.batch_norm
        self.viewpoint_invariant = cfg.viewpoint_invariant
        self.attention_type = cfg.attention_type
        self.layer_norm_guidance = cfg.layer_norm_guidance
        self.residual_blocks = cfg.residual_blocks


DownSamplerConfig = edict()
DownSamplerConfig.use_normals = False
DownSamplerConfig.type = SampleMethod.fps
DownSamplerConfig.down_knn = [98, 50, 20, 10, 10]
DownSamplerConfig.self_knn = [98, 50, 20, 10, 10]
DownSamplerConfig.folds = 2
DownSamplerConfig.max_points_num = DatasetConfig.max_points_num
# farthest point sampling
DownSamplerConfig.down_sample_num = [800, 300, 100, 40, 20]
# voxelization configs
DownSamplerConfig.init_voxel_size = 0.05
DownSamplerConfig.voxel_gain = 2
DownSamplerConfig.knn_method = KNNMethod.skl
# normal configs
DownSamplerConfig.max_normal_number = 20
DownSamplerConfig.normals_radius = 0.1


CoCsConfig = edict()
CoCsConfig.output_layer = 1
CoCsConfig.use_layer_scale = True
CoCsConfig.layer_scale_init_value = 1e-5
CoCsConfig.mlp_ratios = [8, 8, 4, 4]
CoCsConfig.embed_dims = [64, 128, 320, 512]
CoCsConfig.head_dim = [32, 32, 32, 32]
CoCsConfig.heads = [6, 6, 12, 12]  # [6, 6, 12, 12] [4, 4, 8, 8]
CoCsConfig.layers = [4, 4, 12, 4]

CoCsConfig.reducer = edict()
CoCsConfig.reducer.in_channel = 3
CoCsConfig.reducer.out_channel = 64
CoCsConfig.reducer.weightnet_output = 16
CoCsConfig.reducer.num_heads = 4
CoCsConfig.reducer.guidance_feat_len = 32
CoCsConfig.reducer.batch_norm = True
CoCsConfig.reducer.viewpoint_invariant = True
CoCsConfig.reducer.layer_norm_guidance = False
CoCsConfig.reducer.attention_type = AttentionType.subtraction
CoCsConfig.reducer.residual_blocks = 1

CoarseReducer = edict()
CoarseReducer.in_channel = CoCsConfig.embed_dims[-1]
CoarseReducer.out_channel = 256
CoarseReducer.weightnet_output = 16
CoarseReducer.num_heads = 8
CoarseReducer.guidance_feat_len = 32
CoarseReducer.batch_norm = True
CoarseReducer.viewpoint_invariant = True
CoarseReducer.layer_norm_guidance = False
CoarseReducer.attention_type = AttentionType.subtraction
CoarseReducer.residual_blocks = 0

CoCsTotal = edict()
CoCsTotal.dim = 128
CoCsTotal.hidden = 256
CoCsTotal.sparse_dim = CoarseReducer.out_channel
CoCsTotal.dense_dim = CoCsConfig.embed_dims[CoCsConfig.output_layer]

CoCsTotal.reducer = edict()
CoCsTotal.reducer.in_channel = CoCsTotal.hidden
CoCsTotal.reducer.out_channel = CoCsTotal.dim
CoCsTotal.reducer.weightnet_output = 16
CoCsTotal.reducer.num_heads = 4
CoCsTotal.reducer.guidance_feat_len = 32
CoCsTotal.reducer.batch_norm = True
CoCsTotal.reducer.viewpoint_invariant = True
CoCsTotal.reducer.layer_norm_guidance = False
CoCsTotal.reducer.attention_type = AttentionType.subtraction
CoCsTotal.reducer.residual_blocks = 0

CoCsTotal.cocs = edict()
CoCsTotal.cocs.output_layer = -2
CoCsTotal.cocs.use_layer_scale = True
CoCsTotal.cocs.layer_scale_init_value = 1e-5
CoCsTotal.cocs.mlp_ratios = 4
CoCsTotal.cocs.embed_dims = CoCsTotal.hidden
CoCsTotal.cocs.heads = 4  # [6, 6, 12, 12]
CoCsTotal.cocs.head_dim = 32
CoCsTotal.cocs.layers = [8]

CoCsCross = edict()
CoCsCross.max_matches = 500
CoCsCross.no_correlation = False
CoCsCross.mlp_ratio = 4
CoCsCross.head_dim = 32
CoCsCross.points_head_dim = 16
CoCsCross.heads = 8  # [6, 6, 12, 12]
CoCsCross.dense_dim = CoCsConfig.embed_dims[CoCsConfig.output_layer]

R2FormerConfig = edict()
R2FormerConfig.dense_dim = CoCsConfig.embed_dims[CoCsConfig.output_layer]
R2FormerConfig.selected_number = DownSamplerConfig.down_sample_num[CoCsConfig.output_layer]
R2FormerConfig.head_idx = -1
R2FormerConfig.num_corr = 5
R2FormerConfig.decoder_embed_dim = 48
R2FormerConfig.decoder_num_heads = 4
R2FormerConfig.decoder_mlp_ratio = 4
R2FormerConfig.decoder_depth = 6

DeepCoCsCross = edict()
DeepCoCsCross.max_matches = 1000
DeepCoCsCross.no_correlation = False
DeepCoCsCross.mlp_ratio = 4
DeepCoCsCross.head_dim = 32
DeepCoCsCross.points_head_dim = 16
DeepCoCsCross.heads = 8  # [6, 6, 12, 12]
DeepCoCsCross.dense_dim = CoCsConfig.embed_dims[CoCsConfig.output_layer]

DeepCoCsCross.downsampled_num = DownSamplerConfig.down_sample_num[CoCsConfig.output_layer+1]
DeepCoCsCross.knn = DownSamplerConfig.down_knn[CoCsConfig.output_layer+1]

# CoCsCross.out_dim = 256
# CoCsCross.hidden_dim = 256
# CoCsCross.layers = 4
# CoCsCross.dense_dim = CoCsConfig.embed_dims[CoCsConfig.output_layer]
# CoCsCross.use_layer_scale = True
# CoCsCross.layer_scale_init_value = 1e-5
# CoCsCross.mlp_ratios = 4
# CoCsCross.head_dim = 32
# CoCsCross.heads = 8  # [6, 6, 12, 12]
#
# CoCsCross.reducer = edict()
# CoCsCross.reducer.in_channel = CoCsCross.hidden_dim
# CoCsCross.reducer.out_channel = CoCsCross.out_dim
# CoCsCross.reducer.weightnet_output = 16
# CoCsCross.reducer.num_heads = 4
# CoCsCross.reducer.guidance_feat_len = 32
# CoCsCross.reducer.batch_norm = True
# CoCsCross.reducer.viewpoint_invariant = True
# CoCsCross.reducer.layer_norm_guidance = False
# CoCsCross.reducer.attention_type = AttentionType.subtraction
# CoCsCross.reducer.residual_blocks = 0

# GeoFormer = edict()
# GeoFormer.input_dim = CoCsConfig.embed_dims[CoCsConfig.output_layer]
# GeoFormer.geo_dim = 256
# GeoFormer.selected_num = 1
# GeoFormer.decoder_embed_dim = 256
# GeoFormer.decoder_mlp_ratio = 4
# GeoFormer.decoder_depth = 4
# GeoFormer.num_heads = 4
# GeoFormer.sigma_d = 4.8
# GeoFormer.sigma_a = 15
# GeoFormer.angle_k = 3
# GeoFormer.reduction_a = ReducerType.max

CrossModelConfig = edict()
CrossModelConfig.distance_type = DistanceType.cos
CrossModelConfig.cross_type = CrossType.cross_cocs
# CrossModelConfig.geotransformer = GeoFormer
CrossModelConfig.cocs = CoCsTotal
CrossModelConfig.cocs_cross = CoCsCross
CrossModelConfig.deep_cross = DeepCoCsCross
CrossModelConfig.r2former = R2FormerConfig

Minkloc = edict()
Minkloc.in_channels = 1
Minkloc.feature_size = 256
Minkloc.planes = (64, 128, 64, 32)
Minkloc.layers = (1, 1, 1, 1)
Minkloc.num_top_down = 2
Minkloc.conv0_kernel_size = 5
Minkloc.pooling = "GeM"
Minkloc.normalize_embeddings = False
Minkloc.quant_step=0.01

PointNetVlad = edict()
PointNetVlad.points_num=DatasetConfig.max_points_num
PointNetVlad.feature_transform=True
PointNetVlad.max_pool=False
PointNetVlad.output_dim=256

ModelConfig = edict()
ModelConfig.fix_global_model = False
ModelConfig.down_sampler = DownSamplerConfig
ModelConfig.model_type = ModelType.coarse_matching
ModelConfig.cocs_config = CoCsConfig
ModelConfig.coarse_reducer = CoarseReducer

ModelConfig.last_type = CoarseMatchingOutputType.conv
ModelConfig.last_dim = 256
ModelConfig.last_kernel = (15, 20)
ModelConfig.last_stride = (1, 1)
ModelConfig.max_pool = (7, 7)  # (7,7)
ModelConfig.feature_dim = 0
ModelConfig.geo_dim = 0

ModelConfig.rerank_config = CrossModelConfig
ModelConfig.rgb_config = RGBModelConfig
ModelConfig.minkloc_config = Minkloc
ModelConfig.pointnetvlad_config = PointNetVlad
ModelConfig.network_type = 0


#########################################################################
# Configuration of Training
#########################################################################
class ScheduleMethods:
    step = "step"
    cosine = "cosine"


TrainingConfig = edict()
TrainingConfig.name = ""
TrainingConfig.eval = False
TrainingConfig.scheduler = ScheduleMethods.step
TrainingConfig.data_loader = DatasetConfig
TrainingConfig.only_model = False
TrainingConfig.sanity_check = False

TrainingConfig.recall_frequency = 5
TrainingConfig.evaluation_freq = 5
TrainingConfig.mining_frequency = -1
TrainingConfig.mining_start = 0
TrainingConfig.data_mining = False

TrainingConfig.gpus = edict()
TrainingConfig.gpus.channels_last = False
TrainingConfig.gpus.local_rank = 1
TrainingConfig.gpus.sync_bn = False
TrainingConfig.gpus.no_ddp_bb = False
TrainingConfig.gpus.device = "cuda:0"

TrainingConfig.output_dir = "./results"
TrainingConfig.snapshot = "./pretrained.pth.tar"
TrainingConfig.logger = edict()
TrainingConfig.logger.log_steps = 5
TrainingConfig.logger.verbose = False
TrainingConfig.logger.reset_num_timesteps = True
TrainingConfig.logger.log_root = TrainingConfig.output_dir + "/loginfo/"

TrainingConfig.max_epoch = 150
TrainingConfig.max_iteration_per_epoch = 50000
TrainingConfig.max_evaluation_iteration_per_epoch = 5000
TrainingConfig.max_recall_iteration_per_epoch = 10000
TrainingConfig.lr = 1e-4
TrainingConfig.weight_decay = 0
# for step scheduler
TrainingConfig.lr_decay = 0.8
TrainingConfig.lr_decay_steps = 5
# for cosine scheduler
TrainingConfig.lr_t0 = 1
TrainingConfig.lr_tm = 5
TrainingConfig.lr_min = 1e-7

TrainingConfig.loss = LossConfig
TrainingConfig.model = ModelConfig
TrainingConfig.recall_num = 5
TrainingConfig.rerank_num = 20

EvaluationConfig = edict()
EvaluationConfig.recall_num = 50
EvaluationConfig.rerank_num = 20
EvaluationConfig.strick_load = True
EvaluationConfig.generate_data = False
EvaluationConfig.verbose = False
EvaluationConfig.use_geometric = False
EvaluationConfig.model = ModelConfig
EvaluationConfig.data_loader = DatasetConfig
EvaluationConfig.snapshot = ""
EvaluationConfig.name = ""


class GeometricType:
    RANSAC = "ransac"
    TEASER = "teaser"