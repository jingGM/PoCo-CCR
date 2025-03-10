import copy
import random
from functools import partial
from typing import List
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from utils.config import DataDict, DatasetType, ModelType
from src.data_loader.coarse_dataset import CoarseDataset
from src.data_loader.rerank_dataset import RerankDataset


def reset_seed_worker_init_fn(worker_id):
    """Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def registration_collate_fn_stack_mode(data_dicts):
    """Collate function for registration in stack mode.
    Args:
        data_dicts (List[Dict])
    Returns:
        collated_dict (Dict)
    """
    output_dict = {}
    if DataDict.binary_label in data_dicts[0].keys():
        all_labels = np.stack([single_dict[DataDict.binary_label] for single_dict in data_dicts])
        output_dict.update({DataDict.binary_label: torch.from_numpy(all_labels)})
    if DataDict.frame_id in data_dicts[0].keys():
        all_frame_ids = np.stack([single_dict[DataDict.frame_id] for single_dict in data_dicts])
        output_dict.update({DataDict.frame_id: torch.from_numpy(all_frame_ids)})

    if DataDict.pts_features in data_dicts[0].keys():
        all_pts = []
        for single_dict in data_dicts:
            single_batch = np.stack(single_dict[DataDict.pts_features])
            all_pts.append(single_batch)
        output_dict.update({DataDict.pts_features: torch.from_numpy(np.stack(all_pts))})
    else:
        output_dict.update({
            DataDict.global_descriptor: torch.from_numpy(np.stack([np.stack(
                single_dict[DataDict.global_descriptor]) for single_dict in data_dicts])),
            DataDict.features: torch.from_numpy(np.stack([np.stack(
                single_dict[DataDict.features]) for single_dict in data_dicts])),
            DataDict.geometrics: torch.from_numpy(np.stack([np.stack(
                single_dict[DataDict.geometrics]) for single_dict in data_dicts])),
            DataDict.neighbors: torch.from_numpy(np.stack([np.stack(
                single_dict[DataDict.neighbors]) for single_dict in data_dicts])),
            DataDict.similarities: torch.from_numpy(np.stack([np.stack(
                single_dict[DataDict.similarities]) for single_dict in data_dicts]))
            })
    return output_dict


def get_data_loader(cfg, model_type):
    if model_type == ModelType.fine_matching:
        dataset = RerankDataset(cfg=cfg)
    else:
        dataset = CoarseDataset(cfg)
    sampler = DistributedSampler(dataset) if cfg.distributed else None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=sampler,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )
    return data_loader


def train_data_loader(cfg, model_type):
    """
    This function is to create a training dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    cfgs.type = DatasetType.training
    return get_data_loader(cfg=cfgs, model_type=model_type)


def evaluation_data_loader(cfg, model_type):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    cfgs.type = DatasetType.evaluation
    return get_data_loader(cfg=cfgs, model_type=model_type)


def test_data_loader(cfg, model_type):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    cfgs.type = DatasetType.test
    return get_data_loader(cfg=cfgs, model_type=model_type)


def sanity_check_data_loader(cfg, model_type):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    cfgs.type = DatasetType.sanity_check
    return get_data_loader(cfg=cfgs, model_type=model_type)
