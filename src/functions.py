import math
import pickle
import warnings
from os.path import join
from typing import Optional, Union, Dict

from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils.config import DistanceType, DataDict, LossNames, ModelType


class Distance(nn.Module):
    def __init__(self, distance_type, distance=False, dim=1, eps=1e-6):
        super().__init__()
        self.distance_type = distance_type
        self.distance = distance

        if self.distance_type == DistanceType.cos:
            self.model = nn.CosineSimilarity(dim=dim, eps=eps)
        elif self.distance_type == DistanceType.l1:
            self.model = nn.PairwiseDistance(p=1)
        elif self.distance_type == DistanceType.l2:
            self.model = nn.PairwiseDistance(p=2)
        else:
            raise ValueError("the distance type is {} not defined in loss function".format(distance_type))

    def forward(self, x, y):
        if self.distance_type == DistanceType.cos:
            if self.distance:
                return 1.0 - self.model(x, y)
            else:
                return self.model(x, y)
        elif self.distance_type == DistanceType.l1 or self.distance_type == DistanceType.l2:
            if self.distance:
                return self.model(x, y)
            else:
                return 1.0 - self.model(x, y)


def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
    r"""Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    else:
        return nn.Dropout(p=p, **kwargs)


def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances
