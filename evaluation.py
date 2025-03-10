import copy
import os.path
import pickle
import shutil
import time
from os.path import exists, join

from flopth import flopth
# from pthflops import count_ops
# from torchprofile import profile_macs
# from thop import profile

import numpy as np
# import cv2
import torch
import argparse

# from comparisons.ransac import calculate_ransac, RANSAC
from src.data_loader.data_loader import registration_collate_fn_stack_mode
from src.model import get_model
from src.data_loader.recall_dataset import RecallData, RerankRecallDataset
from src.rerank_data_generation import RerankDataGenerator
from utils.config import DatasetType, ModelType, DistanceType, PointFolder, EvaluationConfig, RecallDataType, CrossType, \
    DataDict
from src.loss import RecallCalculation
from utils.functions import to_device


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose the evaluation')

    parser.add_argument('--name', type=str, help='root of the dataset', default="result")
    parser.add_argument('--data_root', type=str, help='root of the dataset', default="")
    parser.add_argument('--recall_data_type', type=int, default=2,
                        help="0: training, 1: evaluation, 2: test, 3:sanity check")
    parser.add_argument('--recall_database', type=int, default=0,
                        help="the key frames are chosen by negative(0) or positive(1) pairs")
    parser.add_argument('--bat', type=int, default=1, help="0: fused; 1: single frame")
    parser.add_argument('--idx', type=int, default=0, help="0: fused; 1: single frame")
    parser.add_argument('--point_type', type=int, default=0, help="0: fused; 1: single frame")
    parser.add_argument('--instance_size', type=int, default=1, help="frames loading each step")
    parser.add_argument('--generate_data', action='store_true', default=False, help='if using color features')
    parser.add_argument('--no_color', action='store_true', default=False, help='if using color features')
    parser.add_argument('--no_geos', action='store_true', default=False, help='if using geometric features')
    parser.add_argument('--no_normals', action='store_true', default=False, help='if using normal information')

    parser.add_argument('--network_type', type=int, default=0, help='0:prpts, 1:image, 2:pointnetvlad, 3:minkloc')

    parser.add_argument('--no_correlation', action='store_true', default=False)
    parser.add_argument('--save_stage1', action='store_true', default=False)
    parser.add_argument('--geometric', type=int, default=0, help="1: ransac, 2: teaser")
    parser.add_argument('--model_type', type=int, default=0, help="0: coarse matching, 1: all matching")
    parser.add_argument('--snapshot', type=str, help='path of the snapshot', default="")

    parser.add_argument('--output', default="rerank", type=str,
                        help='root of the dataset')
    parser.add_argument('--recall_num', type=int, default=5, help="max number of recall")
    parser.add_argument('--rerank_num', type=int, default=20, help="max number of recall")
    parser.add_argument('--distance_type', type=int, default=0, help="0: cos, 1: l1, 2: l2")
    return parser.parse_args()


def setup_config(args):
    cfg = EvaluationConfig
    cfg.save_stage1 = args.save_stage1
    cfg.recall_num = cfg.data_loader.recall_num = args.recall_num
    cfg.rerank_num = args.rerank_num
    cfg.generate_data = args.generate_data
    cfg.verbose = args.verbose
    cfg.output = args.output
    cfg.distance_type = args.distance_type
    cfg.snapshot = args.snapshot
    cfg.geometric = args.geometric
    cfg.name = args.name

    cfg.batch = args.bat
    cfg.idx = args.idx

    cfg.model.network_type = cfg.data_loader.network_type = args.network_type

    # dataset:
    cfg.data_loader.max_iteration = 5000000
    cfg.data_loader.root = args.data_root
    cfg.data_loader.instance_size = args.instance_size
    if args.point_type == 0:
        cfg.data_loader.point_folder = PointFolder.fused
    elif args.point_type == 1:
        cfg.data_loader.point_folder = PointFolder.single
    else:
        raise ValueError("the points folder name is not defined: {}, 0: fused; 1: single".format(args.point_type))

    if args.recall_data_type == 0:
        cfg.data_loader.type = DatasetType.training
    elif args.recall_data_type == 1:
        cfg.data_loader.type = DatasetType.evaluation
    elif args.recall_data_type == 2:
        cfg.data_loader.type = DatasetType.test
    elif args.recall_data_type == 3:
        cfg.data_loader.type = DatasetType.sanity_check
    else:
        raise ValueError("the recall datatype {} is not defined, only 0-2".format(args.image_type))

    if args.recall_database == 0:
        cfg.data_loader.recall_database = RecallDataType.negative_key_frames
    elif args.recall_database == 1:
        cfg.data_loader.recall_database = RecallDataType.positive_key_frames
    elif args.recall_database == 2:
        cfg.data_loader.recall_database = RecallDataType.distance
    else:
        raise ValueError("the recall database type {} is not defined".format(args.recall_database))
    # recall settings:
    if args.distance_type == 0:
        cfg.distance_type = DistanceType.cos
    elif args.distance_type == 1:
        cfg.distance_type = DistanceType.l1
    elif args.distance_type == 2:
        cfg.distance_type = DistanceType.l2
    else:
        raise ValueError("the distance type {} is not defined in the loss configuration".format(args.distance_type))

    # model settings:
    if args.model_type == 0:
        cfg.model.model_type = ModelType.coarse_matching
    elif args.model_type == 1:
        cfg.model.model_type = ModelType.all_matching
    elif args.model_type == 2:
        cfg.model.model_type = ModelType.fine_matching
    else:
        raise ValueError(
            "the model type {} is not defined: 0: coarse matching, 1: fine matching".format(args.model_type))

    cfg.model.rerank_config.cocs_cross.no_correlation = args.no_correlation
    cfg.model.rerank_config.cross_type = CrossType.cross_cocs

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

    cfg.data_loader.use_color = not args.no_color
    if args.no_color:
        if cfg.model.cocs_config.reducer.in_channel == 0:
            cfg.model.cocs_config.reducer.in_channel = 1
    else:
        cfg.model.cocs_config.reducer.in_channel += 3
    return cfg


def evaluate_recall(cfg):
    output_dir = join(cfg.output, cfg.name)

    if cfg.generate_data:
        ori_model_type = copy.deepcopy(cfg.model.model_type)
        cfg.model.model_type = ModelType.coarse_matching
        generator = RerankDataGenerator(device="cuda", model_cfg=cfg.model, snapshot=cfg.snapshot,
                                        distance_type=cfg.distance_type, sanity_check=False,
                                        data_loader_cfg=cfg.data_loader, rerank_num=cfg.rerank_num)
        generator.generate_data_type(cfg.data_loader.type, save_files=True)
        cfg.model.model_type = ori_model_type

    recallcalculator = RecallCalculation(distance_type=cfg.distance_type, verbose=cfg.verbose,
                                         max_recall_num=cfg.data_loader.recall_num,
                                         model_type=cfg.model.model_type, save_stage1=cfg.save_stage1,
                                         instance_num=cfg.data_loader.instance_size,
                                         rerank_frame_num=cfg.rerank_num,
                                         output_dir=output_dir, geometric=cfg.geometric)
    if cfg.model.model_type == ModelType.fine_matching:
        evaluation_dataset = RerankRecallDataset(cfg=cfg.data_loader)
    else:
        evaluation_dataset = RecallData(cfg.data_loader)
    model = get_model(cfg.model, device="cuda")
    if cfg.snapshot:
        state_dict = torch.load(cfg.snapshot, map_location=torch.device('cpu'))
        model_dict = state_dict['state_dict']
        # print(model_dict.keys())
        # print(model.state_dict().keys())
        model.load_state_dict(model_dict, strict=cfg.strick_load)
        torch.cuda.empty_cache()
    model.cuda()
    model.eval()
    coarse_match, fine_match = recallcalculator.evaluate_recall(model=model, dataset=evaluation_dataset, device="cuda")
    print("stage 1 reacall: ", coarse_match)
    print("stage 2 reacall: ", fine_match)


if __name__ == "__main__":
    parsed_args = get_args()
    cfgs = setup_config(parsed_args)

    cfgs.strick_load = True
    evaluate_recall(cfg=cfgs)
