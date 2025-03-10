import argparse
import os

import torch

from utils.config import TrainingConfig, ScheduleMethods, FrameInfo, RecallDataType, ModelType, PointFolder, \
    CoarseMatchingOutputType, CrossType, DistanceType


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # data args:
    parser.add_argument('--training_type', type=int, default=2, help="0: 100 epochs; 1: 30 epochs, 2: 30 epochs single cos")
    parser.add_argument('--data_root', type=str, help='root of the dataset',
                        default="/home/jing/Documents/work/data/ScanNetSim")
    parser.add_argument('--batch_size', type=int, default=1, help="the negative number in the same frame")
    parser.add_argument('--workers', type=int, default=16, help="the worker number in the dataloader")
    parser.add_argument('--point_type', type=int, default=0, help="0: fused; 1: single frame")
    parser.add_argument('--max_iteration', type=int, default=50000, help="0: fused; 1: single frame")

    parser.add_argument('--no_color', action='store_true', default=False, help='if using color features')
    parser.add_argument('--no_geos', action='store_true', default=False, help='if using geometric features')
    parser.add_argument('--no_normals', action='store_true', default=False, help='if using normal information')
    parser.add_argument('--no_semantic', action='store_true', default=False, help='if using semantic information')

    parser.add_argument('--neg_num', type=int, default=3, help="the negative number in the same frame")
    parser.add_argument('--pos_num', type=int, default=3, help="the positive number in the same frame")
    parser.add_argument('--instance_size', type=int, default=8, help="instance size of each batch for recall dataset")

    # training args
    parser.add_argument('--name', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--only_load_model', action='store_true', default=False,
                        help='if only load model parameters from the snapshot')
    parser.add_argument('--snapshot', type=str, help='path of the snapshot', default="")
    parser.add_argument("--scheduler", default=0, type=int, help="0: cosine; 1: step")
    parser.add_argument('--mining_frequency', type=int, default=-1, help="how many epochs data mining is applied")
    parser.add_argument('--mining_start', type=int, default=5, help="how many epochs mining starts")
    parser.add_argument('--mining_ratio', type=float, default=0.66, help="how much data is chosen by hard cases")

    parser.add_argument('--recall_num', type=int, default=20, help="the number of recall values")
    parser.add_argument('--rerank_num', type=int, default=5, help="the number of rerank values")
    parser.add_argument('--recall_frequency', type=int, default=-1, help="how many epochs recall is applied")
    parser.add_argument('--evaluation_freq', type=int, default=-1, help="how many epochs evaluation is applied")
    parser.add_argument('--recall_database', type=int, default=0,
                        help="the key frames are chosen by negative(0) or positive(1) pairs")

    parser.add_argument('--sanity_check', action='store_true', default=False, help='verbose the training')

    parser.add_argument('--network_type', type=int, default=0, help='0:prpts, 1:image, 2:pointnetvlad, 3:minkloc')

    # model args
    ''' stage 1 '''
    parser.add_argument('--coarse_out_type', type=int, default=0, help="0: conv, 1: pool")
    parser.add_argument('--model_type', type=int, default=0, help="0: coarse matching, 1: all, 2: reranking")
    parser.add_argument('--small_model', action='store_true', default=False, help='small cocs model or large')

    ''' stage 2 '''
    parser.add_argument('--no_correlation', action='store_true', default=False)
    parser.add_argument('--generate_data', action='store_true', default=False)
    parser.add_argument('--fix_global_model', action='store_true', default=False,
                        help='fix the parameters of the model in stage 1')
    parser.add_argument('--cross_type', type=int, default=0,
                        help="0:cocs, 1: cross cocs, 2: mddified 3D r2former, 3:cross_attn, 4:cocs_attn, 5: deep cross")
    # parser.add_argument('--cross_selection_type', type=str, default="max", help="the method to choose centers")
    parser.add_argument('--fine_layer', type=int, default=1,
                        help="The output layer index of the cocs model for stage 2")

    # Loss args:
    parser.add_argument('--distance_type', type=int, default=0, help="0: cos, 1: l1, 2: l2")
    parser.add_argument('--triplet_ratio', type=float, default=100.0, help="the ratio of the triplet loss")
    parser.add_argument('--circle_ratio', type=float, default=0.01, help="the ratio of the circle loss")

    # GPUs
    parser.add_argument('--channels-last', action='store_true', default=False, help='Use channels_last memory layout')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--split-bn', action='store_true', help='Enable separate BN layers per augmentation split.')
    parser.add_argument('--device', type=int, default=-1, help="the gpu id")
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    return parser.parse_args()


def get_configuration():
    args = get_args()
    cfg = TrainingConfig

    #########################################
    # training configurations
    #########################################
    cfg.name = args.name
    cfg.sanity_check = args.sanity_check
    cfg.only_model = args.only_load_model
    cfg.snapshot = args.snapshot
    cfg.recall_num = args.recall_num
    cfg.rerank_num = args.rerank_num
    cfg.max_iteration_per_epoch = args.max_iteration

    cfg.mining_start = args.mining_start
    cfg.mining_frequency = args.mining_frequency
    cfg.data_loader.negative_mining_num_ratio = args.mining_ratio

    cfg.recall_frequency = args.recall_frequency
    cfg.evaluation_freq = args.evaluation_freq

    if args.scheduler == 0:
        cfg.scheduler = ScheduleMethods.cosine
    elif args.scheduler == 1:
        cfg.scheduler = ScheduleMethods.step
    else:
        raise ValueError("the schedule method is not defined")

    if args.training_type == 0:
        cfg.max_epoch = 100
        if cfg.scheduler == ScheduleMethods.cosine:
            cfg.lr_tm = 10
        elif cfg.scheduler == ScheduleMethods.step:
            cfg.lr_decay = 0.7
            cfg.lr_decay_steps = 2
    elif args.training_type == 1:
        cfg.max_epoch = 30
        if cfg.scheduler == ScheduleMethods.cosine:
            cfg.lr = 2e-5
            cfg.lr_tm = 30
            cfg.lr_min = 1e-8
    elif args.training_type == 2:
        cfg.max_epoch = 30
        if cfg.scheduler == ScheduleMethods.cosine:
            cfg.lr_tm = 30
            cfg.lr_min = 1e-8
    else:
        raise ValueError("the data type is not defined")

    # Devices
    if args.device >= 0:
        cfg.gpus.device = "cuda:{}".format(args.device)
    elif args.device == -1:
        cfg.gpus.device = "cuda"
        print("------------------- CUDA: ", torch.cuda.is_available(), "-----------------------")
    else:
        cfg.gpus.device = "cpu"
    # print(" ------ local rank: ", os.environ['LOCAL_RANK'])
    cfg.gpus.channels_last = args.channels_last
    cfg.gpus.local_rank = args.local_rank
    cfg.gpus.sync_bn = args.sync_bn
    cfg.gpus.split_bn = args.split_bn
    cfg.gpus.no_ddp_bb = args.no_ddp_bb

    if args.model_type == 0:
        cfg.model.model_type = ModelType.coarse_matching
        cfg.model.cocs_config.output_layer = 0
    elif args.model_type == 1:
        cfg.model.fix_global_model = args.fix_global_model
        cfg.model.model_type = ModelType.all_matching
        cfg.model.cocs_config.output_layer = args.fine_layer
        cfg.model.rerank_config.cocs.dense_dim = cfg.model.rerank_config.r2former.dense_dim = cfg.model.rerank_config.cocs_cross.dense_dim =  \
        cfg.model.cocs_config.embed_dims[cfg.model.cocs_config.output_layer]
        cfg.model.rerank_config.r2former.selected_number = cfg.model.down_sampler.down_sample_num[cfg.model.cocs_config.output_layer]
        cfg.model.rerank_config.deep_cross.downsampled_num = cfg.model.down_sampler.down_sample_num[cfg.model.cocs_config.output_layer+1:cfg.model.cocs_config.output_layer + 3]
        cfg.model.rerank_config.deep_cross.knn = cfg.model.down_sampler.down_knn[cfg.model.cocs_config.output_layer+1:cfg.model.cocs_config.output_layer + 3]
    elif args.model_type == 2:
        cfg.generate_data = args.generate_data
        cfg.model.model_type = ModelType.fine_matching
        cfg.model.cocs_config.output_layer = args.fine_layer
        cfg.model.rerank_config.cocs.dense_dim = cfg.model.rerank_config.r2former.dense_dim = cfg.model.rerank_config.cocs_cross.dense_dim = \
        cfg.model.cocs_config.embed_dims[cfg.model.cocs_config.output_layer]
        cfg.model.rerank_config.r2former.selected_number = cfg.model.down_sampler.down_sample_num[cfg.model.cocs_config.output_layer]
        cfg.model.rerank_config.deep_cross.downsampled_num = cfg.model.down_sampler.down_sample_num[cfg.model.cocs_config.output_layer+1:cfg.model.cocs_config.output_layer + 3]
        cfg.model.rerank_config.deep_cross.knn = cfg.model.down_sampler.down_knn[cfg.model.cocs_config.output_layer+1:cfg.model.cocs_config.output_layer + 3]
    else:
        raise ValueError(
            "the model type {} is not defined: 0: coarse matching, 1: fine matching".format(args.model_type))

    #########################################
    # data configurations
    #########################################
    # dataloader
    cfg.data_loader.num_workers = args.workers
    cfg.data_loader.batch_size = args.batch_size

    # dataset
    if args.point_type == 0:
        cfg.data_loader.point_folder = PointFolder.fused
    elif args.point_type == 1:
        cfg.data_loader.point_folder = PointFolder.single
    else:
        raise ValueError("the points folder name is not defined: {}, 0: fused; 1: single".format(args.point_type))

    cfg.data_loader.root = args.data_root
    cfg.data_loader.num_pos_samples = args.pos_num
    cfg.data_loader.num_neg_same_scenes_samples = args.neg_num - cfg.data_loader.num_neg_other_scenes_samples
    cfg.data_loader.instance_size = args.instance_size

    if args.recall_database == 0:
        cfg.data_loader.recall_database = RecallDataType.negative_key_frames
    elif args.recall_database == 1:
        cfg.data_loader.recall_database = RecallDataType.positive_key_frames
    elif args.recall_database == 2:
        cfg.data_loader.recall_database = RecallDataType.distance
    else:
        raise ValueError("the recall database type {} is not defined".format(args.recall_database))

    cfg.data_loader.use_color = not args.no_color
    cfg.data_loader.use_normals = cfg.model.cocs_config.reducer.viewpoint_invariant = cfg.model.coarse_reducer.viewpoint_invariant = not args.no_normals
    # cfg.data_loader.use_semantic = False
    cfg.data_loader.use_geos = not args.no_geos

    cfg.model.network_type = cfg.data_loader.network_type = args.network_type

    #########################################
    # model configurations
    #########################################
    if args.no_geos:
        cfg.model.cocs_config.reducer.in_channel = 0
        if args.no_normals:
            cfg.model.feature_dim = 3
            cfg.model.geo_dim = 3
        else:
            cfg.model.feature_dim = 6
            cfg.model.geo_dim = 6
    else:
        cfg.model.feature_dim = 0
        if args.no_normals:
            cfg.model.cocs_config.reducer.in_channel = 3
            cfg.model.geo_dim = 3
        else:
            cfg.model.cocs_config.reducer.in_channel = 6
            cfg.model.geo_dim = 6

    if args.no_color:
        if cfg.model.cocs_config.reducer.in_channel == 0:
            cfg.model.cocs_config.reducer.in_channel = 1
    else:
        cfg.model.cocs_config.reducer.in_channel += 3

    # first stage:
    if args.coarse_out_type == 0:
        cfg.model.last_type = CoarseMatchingOutputType.conv
        cfg.model.max_pool = (7, 7)
    elif args.coarse_out_type == 1:
        cfg.model.last_type = CoarseMatchingOutputType.pool
        cfg.model.max_pool = (15, 20)
    else:
        raise ValueError("coarse matching, output type is only 0, 1; got {}".format(args.coarse_out_type))

    if args.small_model:
        cfg.model.cocs_config.heads = [4, 4, 8, 8]
        cfg.model.cocs_config.layers = [2, 2, 6, 2]
    else:
        cfg.model.cocs_config.heads = [6, 6, 12, 12]
        cfg.model.cocs_config.layers = [4, 4, 12, 4]

    # second stage:
    cfg.model.rerank_config.cocs_cross.no_correlation = args.no_correlation
    if args.cross_type == 0:
        cfg.model.rerank_config.cross_type = CrossType.cocs
    elif args.cross_type == 1:
        cfg.model.rerank_config.cross_type = CrossType.cross_cocs
    elif args.cross_type == 2:
        cfg.model.rerank_config.cross_type = CrossType.r2former
    elif args.cross_type == 3:
        cfg.model.rerank_config.cross_type = CrossType.cross_attn
    elif args.cross_type == 4:
        cfg.model.rerank_config.cross_type = CrossType.cocs_attn
    elif args.cross_type == 5:
        cfg.model.rerank_config.cross_type = CrossType.deep_cross_cocs
    else:
        raise ValueError("the cross model type {} is not defined".format(args.cross_type))

    # assert args.cross_selection_type == "mean" or args.cross_selection_type == "max", \
    #     "the centers' selection method {} is not defined in the fine matching model, should be either mean or max".format(
    #         args.cross_selection_type)

    #########################################
    # Loss configurations
    #########################################
    if args.distance_type == 0:
        cfg.loss.distance_type = cfg.data_loader.distance_type = cfg.model.rerank_config.distance_type = DistanceType.cos
    elif args.distance_type == 1:
        cfg.loss.distance_type = cfg.data_loader.distance_type = cfg.model.rerank_config.distance_type = DistanceType.l1
    elif args.distance_type == 2:
        cfg.loss.distance_type = cfg.data_loader.distance_type = cfg.model.rerank_config.distance_type = DistanceType.l2
    else:
        raise ValueError("the distance type {} is not defined in the loss configuration".format(args.distance_type))

    cfg.loss.triplet_ratio = args.triplet_ratio
    cfg.loss.circle_ratio = args.circle_ratio

    return cfg
