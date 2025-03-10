import random

import numpy as np
from typing import Union, Optional, Tuple, List
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.neighbors import KDTree
# import faiss
# import open3d as o3d
from scipy.spatial.transform import Rotation
import math
import warnings
from mmcv.ops.scatter_points import dynamic_scatter
from mmcv.ops.voxelize import _Voxelization
from mmcv.ops import knn, furthest_point_sample
from utils.config import KNNMethod, SampleMethod


def VI_coordinate_transform(localized_xyz, gathered_norm, sparse_xyz_norm, K):
    """
    Compute the viewpoint-invariance aware relative position encoding in VI_PointConv
    From: X. Li et al. Improving the Robustness of Point Convolution on k-Nearest Neighbor Neighborhoods with a Viewpoint-Invariant Coordinate Transform. WACV 2023
    Code copyright 2020 Xingyi Li (MIT License)
    Input:
        localized_xyz: 3D coordinates (note VI only works on 3D) B,M,K,3
        gathered_norm: surface normals for each point B,M,K,3
        sparse_xyz_norm: surface normals for each point in the lower resolution (normally
                the same as dense_xyz_norm, except when downsampling)  B,M,3
    Return:
        VI-transformed point coordinates: a concatenation of rotation+scale invariant dimensions, scale-invariant dimensions and non-invariant dimensions
    """
    r_hat = F.normalize(localized_xyz, dim=3)  # B,M,K,3
    v_miu = sparse_xyz_norm.unsqueeze(dim=2) - torch.matmul(sparse_xyz_norm.unsqueeze(dim=2),
                                                            r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
    v_miu = F.normalize(v_miu, dim=3)
    w_miu = torch.cross(r_hat, v_miu, dim=3)
    w_miu = F.normalize(w_miu, dim=3)
    theta1 = torch.matmul(gathered_norm, sparse_xyz_norm.unsqueeze(dim=3))
    theta2 = torch.matmul(r_hat, sparse_xyz_norm.unsqueeze(dim=3))
    theta3 = torch.sum(r_hat * gathered_norm, dim=3, keepdim=True)
    theta4 = torch.matmul(localized_xyz, sparse_xyz_norm.unsqueeze(dim=3))
    theta5 = torch.sum(gathered_norm * r_hat, dim=3, keepdim=True)
    theta6 = torch.sum(gathered_norm * v_miu, dim=3, keepdim=True)
    theta7 = torch.sum(gathered_norm * w_miu, dim=3, keepdim=True)
    theta8 = torch.sum(
        localized_xyz * torch.cross(gathered_norm, sparse_xyz_norm.unsqueeze(dim=2).repeat(1, 1, K, 1), dim=3), dim=3,
        keepdim=True)
    theta9 = torch.norm(localized_xyz, dim=3, keepdim=True)
    return torch.cat([theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz],
                     dim=3).contiguous()


def index_points(points, idx):
    """
    Input:
        points: input points data, [N, C]
        idx: sample index data, [S, [K]]
    Return:
        new_points:, indexed points data, [S, [K], C]
    """
    d = points.size(-1)
    raw_size = idx.size()
    idx = idx.view(-1)
    res = points[idx]
    output = res.view(*raw_size, d)
    return output.contiguous()


def index_batched_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx.to(torch.int64)[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class SampleKNN:
    def __init__(self, cfg):
        self.sample_type = cfg.type
        self.folds = cfg.folds
        self.down_knn = cfg.down_knn
        self.self_knn = cfg.self_knn
        assert len(self.self_knn) == len(
            self.down_knn), "downsampling knn has different number {} than self knn {}".format(
            self.down_knn, self.self_knn)
        self.stages = len(self.down_knn)
        self.max_points_num = cfg.max_points_num

        # normals
        self.normals_radius = cfg.normals_radius
        self.max_normal_number = cfg.max_normal_number

        # voxelization
        self.init_voxel_size = cfg.init_voxel_size
        self.voxel_gain = cfg.voxel_gain
        self.knn_method = cfg.knn_method

        # farthest point sampling
        self.fps_nums = cfg.down_sample_num
        self.min_value = None
        self.max_value = None
        self.separation = None

    def process_single_points(self, point):
        if point.shape[0] > self.max_points_num:
            indices = list(range(point.shape[0]))
            random.shuffle(indices)
            selected_indices = indices[:self.max_points_num]
            point = point[selected_indices]
        return point

    def get_centers(self, lengths, subsample_coordinates, voxel_coordinates):
        """
        points_indices: [stages]; each stage [batches]; each batch [blocks]; each block points indices in the current point cloud
        voxels_indices: [stages]; each stage [batches]; each batch [blocks]; each block points indices in the current voxel cloud
        """
        points_indices = []
        voxels_indices = []
        for idx in range(0, len(self.down_knn), 1):
            points_indices.append([])
            voxels_indices.append([])
            for lidx in range(len(lengths[idx])):
                points_indices[idx].append([])
                voxels_indices[idx].append([])
                pts_sub_indices = subsample_coordinates[idx][lidx]
                vxs_sub_indices = voxel_coordinates[idx][lidx]

                max_coors = np.max(pts_sub_indices, axis=0) + 1  # indices are from 0 to n, which has n+1 items
                coors_size = np.around(max_coors / self.folds)
                indices_ranges = []
                for x in range(self.folds):
                    for y in range(self.folds):
                        for z in range(self.folds):
                            if x == self.folds - 1:
                                x_last = max(x * coors_size[0] + coors_size[0], max_coors[0])
                            else:
                                x_last = x * coors_size[0] + coors_size[0]

                            if y == self.folds - 1:
                                y_last = max(y * coors_size[1] + coors_size[1], max_coors[1])
                            else:
                                y_last = y * coors_size[1] + coors_size[1]

                            if z == self.folds - 1:
                                z_last = max(z * coors_size[2] + coors_size[2], max_coors[2])
                            else:
                                z_last = z * coors_size[2] + coors_size[2]
                            indices_ranges.append([x * coors_size[0], x_last,
                                                   y * coors_size[1], y_last,
                                                   z * coors_size[2], z_last])
                for indices_range in indices_ranges:
                    points_coors_indices = (set(np.where(pts_sub_indices[:, 0] >= indices_range[0])[0]) &
                                            set(np.where(pts_sub_indices[:, 0] < indices_range[1])[0]) &
                                            set(np.where(pts_sub_indices[:, 1] >= indices_range[2])[0]) &
                                            set(np.where(pts_sub_indices[:, 1] < indices_range[3])[0]) &
                                            set(np.where(pts_sub_indices[:, 2] >= indices_range[4])[0]) &
                                            set(np.where(pts_sub_indices[:, 2] < indices_range[5])[0]))
                    if len(points_coors_indices) != 0:
                        points_indices[idx][lidx].append(list(points_coors_indices))

                    voxels_coors_indices = (set(np.where(vxs_sub_indices[:, 0] >= indices_range[0])[0]) &
                                            set(np.where(vxs_sub_indices[:, 0] < indices_range[1])[0]) &
                                            set(np.where(vxs_sub_indices[:, 1] >= indices_range[2])[0]) &
                                            set(np.where(vxs_sub_indices[:, 1] < indices_range[3])[0]) &
                                            set(np.where(vxs_sub_indices[:, 2] >= indices_range[4])[0]) &
                                            set(np.where(vxs_sub_indices[:, 2] < indices_range[5])[0]))
                    if len(voxels_coors_indices) != 0:
                        voxels_indices[idx][lidx].append(list(voxels_coors_indices))
        return points_indices, voxels_indices

    # def voxelize(self, points: Union[List[np.ndarray], np.ndarray]):
    #     """
    #     output:
    #     all_points: list of all points: stage + 1; each: (N,3) xyz / (N,6) with normals
    #     lengths: list of all lengths of points: stage + 1; each: list[size0, size1...]
    #     edges: list of knn points: stage; each: indices (M,K)
    #     self_edges: list of voxel knn: stage; indices (M,K)
    #     voxel_coordinates: list of voxel coordinates (0-Max number of voxels in xyz): stage; each: [[0,1,0],[0,1,2]] M,3
    #     subsample_coordinates: list of voxel coordinates for each points: stage; each: N,3
    #     """
    #     if self.sample_type == SampleMethod.dvx:
    #         return voxelize_knn(pts=points, init_voxel_size=self.init_voxel_size, voxel_gain=self.voxel_gain,
    #                             knn=self.down_knn, self_knn=self.self_knn, knn_method=self.knn_method)
    #     elif self.sample_type == SampleMethod.fps:
    #         return fps_knn(pts=points, fps_nums=self.fps_nums, knn_nums=self.down_knn, self_knn_nums=self.self_knn)
    #     else:
    #         raise ValueError("the sample type is not defined")

    # def fps_folding(self, points: Union[np.ndarray, torch.Tensor]):
    #     if not torch.is_tensor(points):
    #         pts = torch.from_numpy(points)
    #     assert len(points.shape) == 3 and points.shape[-1] >= 3, "the points dimension should be NxKxC where C>=3"
    #     # points: B,N,C
    #     self.min_value = torch.min(points, dim=1)[0]
    #     self.max_value = torch.max(points, dim=1)[0]
    #     difference = self.max_value - self.min_value
    #     self.separation = difference / self.folds
    #     all_points = []
    #     for i in range(len(points.shape[0])):
    #         all_points.append()

    def downsample_knn(self, points):
        if self.sample_type == SampleMethod.dvx:
            all_geos, lengths, edges, self_edges, voxel_coordinates, subsample_coordinates = voxelize_knn(
                pts=points, init_voxel_size=self.init_voxel_size, voxel_gain=self.voxel_gain,
                knn=self.down_knn, self_knn=self.self_knn, knn_method=self.knn_method)
            points_indices, voxels_indices = self.get_centers(lengths=lengths, voxel_coordinates=voxel_coordinates,
                                                              subsample_coordinates=subsample_coordinates)
            return all_geos, edges, self_edges, lengths, points_indices, voxels_indices
        elif self.sample_type == SampleMethod.fps:
            if not torch.is_tensor(points):
                points = torch.from_numpy(points)
            if len(points.shape) == 2:
                points = points.unsqueeze(0)
            all_points, edges, self_edges = fps_knn(pts=points, fps_nums=self.fps_nums, knn_nums=self.down_knn,
                                                    self_knn_nums=self.self_knn)
            return all_points, edges, self_edges, None, None, None


def fps_knn(pts: Union[np.ndarray, torch.Tensor], fps_nums: List[int] = [1000, 500], knn_nums: List[int] = [20, 20],
            self_knn_nums: List[int] = [20, 20]):
    assert len(fps_nums) == len(knn_nums) == len(
        self_knn_nums), "the sample number list, knn list and self knn list should have the same length"
    assert len(pts.shape) == 3 and pts.shape[-1] >= 3, "the points dimension should be NxKxC where C>=3"


    all_points = [pts]
    edges = []
    self_edges = []
    for stage_idx in range(len(fps_nums)):
        current_num = fps_nums[stage_idx]
        knn_num = knn_nums[stage_idx]
        self_knn_num = self_knn_nums[stage_idx]
        sampled_pts, knn_indices, self_knn_indcies = single_fps_knn(points=all_points[-1], fps_num=current_num,
                                                                    k_num=knn_num, self_k_num=self_knn_num)
        all_points.append(sampled_pts)
        edges.append(knn_indices)
        self_edges.append(self_knn_indcies)
    return all_points, edges, self_edges


def single_fps_knn(points: torch.Tensor, fps_num: int, k_num: int, self_k_num=-1):
    assert len(points.shape) == 3 and points.shape[-1] >= 3, "the points dimension should be NxKxC where C>=3"
    assert fps_num >= self_k_num, "the farthest point number {} should be larger than self KNN number {}".format(
        fps_num, self_k_num)
    assert points.shape[1] > k_num, "the points number {} should be larger than KNN number {}".format(points.shape[1],
                                                                                                      k_num)

    pts = points[:, :, :3].to(torch.float).contiguous().to("cuda")
    fps_indices = furthest_point_sample(pts, fps_num)
    sampled_pts = index_batched_points(pts, fps_indices).contiguous()
    knn_indices = knn(k_num + 1, pts, sampled_pts)

    sampled_all = index_batched_points(points=points, idx=fps_indices.detach().to(points.device))

    if self_k_num > 0:
        self_knn_indices = knn(self_k_num + 1, sampled_pts, sampled_pts)
        return (sampled_all,
                knn_indices[:, 1:, :].detach().cpu().transpose(2, 1),
                self_knn_indices[:, 1:, :].detach().cpu().transpose(2, 1))
    else:
        return sampled_all, knn_indices[:, 1:, :].detach().cpu().transpose(2, 1)


def voxelize_knn(pts, init_voxel_size=0.1, voxel_gain=2,
                 knn: List[int] = [20, 20], self_knn: List[int] = [20, 20], knn_method=KNNMethod.skl):
    lengths = [[pt.shape[0] for pt in pts]]
    all_points = [pts]
    edges = []
    self_edges = []
    voxel_coordinates = []
    subsample_coordinates = []
    for s in range(len(knn)):
        all_points.append([])
        lengths.append([])
        edges.append([])
        self_edges.append([])
        voxel_coordinates.append([])
        subsample_coordinates.append([])
        for idx in range(len(all_points[s])):
            pt = all_points[s][idx]
            voxel_size = (init_voxel_size * (voxel_gain ** s),
                          init_voxel_size * (voxel_gain ** s),
                          init_voxel_size * (voxel_gain ** s))
            voxel_pts, voxel_coors, points_coors = voxelize_points(points=torch.tensor(pt), voxel_size=voxel_size)

            voxel_pts = voxel_pts.detach().cpu().numpy()
            neighbors_idx = compute_knn(pt[:, :3], voxel_pts[:, :3], knn[s], method=knn_method) + sum(lengths[s][:idx])
            self_neighbors_idx = compute_knn(voxel_pts[:, :3], voxel_pts[:, :3], self_knn[s], method=knn_method) + sum(
                lengths[s + 1][:idx])
            all_points[s + 1].append(voxel_pts)
            lengths[s + 1].append(voxel_pts.shape[0])
            edges[s].append(neighbors_idx)
            self_edges[s].append(self_neighbors_idx)

            voxel_coordinates[s].append(voxel_coors.detach().cpu().numpy())
            subsample_coordinates[s].append(points_coors.detach().cpu().numpy())

    all_points[0] = np.concatenate(all_points[0], axis=0)
    for s in range(1, len(knn) + 1, 1):
        all_points[s] = np.concatenate(all_points[s], axis=0)
        edges[s - 1] = np.concatenate(edges[s - 1], axis=0)
        self_edges[s - 1] = np.concatenate(self_edges[s - 1], axis=0)
    lengths = np.array(lengths)
    return all_points, lengths, edges, self_edges, voxel_coordinates, subsample_coordinates


def compute_knn(ref_points, query_points, K, method=KNNMethod.skl):
    """
    Compute KNN
    Input:
        ref_points: reference points (MxD)
        query_points: query points (NxD)
        K: the amount of neighbors for each point
        (Engelmann et al. Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds. ICRA 2020)
        method: Choose between two approaches: Scikit-Learn ('sklearn') or nanoflann ('nanoflann'). In general nanoflann should be faster, but sklearn is more stable
    Output:
        neighbors_idx: for each query point, its K nearest neighbors among the reference points (N x K)
    """
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(num_ref_points, (num_query_points, K)).astype(np.int32)
        return inds

    if method == KNNMethod.skl:
        kdt = KDTree(ref_points)
        neighbors_idx = kdt.query(query_points, k=K, return_distance=False)
    # elif method == "faiss":
    #     index = faiss.index_factory(3, "Flat")
    #     index.train(ref_points)
    #     index.add(ref_points)
    #     distances, neighbors_idx = index.search(query_points.astype(np.float32), K)
    else:
        raise Exception('compute_knn: unsupported knn algorithm')

    return neighbors_idx


def voxelize_points(points: torch.Tensor, voxel_size, device="cuda", coors_range=None):
    if type(voxel_size) == float:
        voxel_size = (voxel_size, voxel_size, voxel_size)
    points = to_device(points, device=device)

    if coors_range is None:
        points_min = points.min(dim=0)[0]
        points_max = points.max(dim=0)[0]
        coors_range = (points_min[0] - voxel_size[0] / 2.0,
                       points_min[1] - voxel_size[1] / 2.0,
                       points_min[2] - voxel_size[2] / 2.0,
                       points_max[0] + voxel_size[0] / 2.0,
                       points_max[1] + voxel_size[1] / 2.0,
                       points_max[2] + voxel_size[2] / 2.0)
    points_to_voxel_indices = _Voxelization.apply(points[:, :3], voxel_size, coors_range, -1, -1)
    voxel_feats, voxel_indices = dynamic_scatter(points.contiguous(), points_to_voxel_indices.contiguous(),
                                                 "mean")
    # tsssss = torch.round((voxel_feats[:, :3].detach().cpu() - points_min) / voxel_size[0])[:, [2, 1, 0]]
    # valeus = torch.sum(tsssss - voxel_indices.detach().cpu())
    return voxel_feats, voxel_indices, points_to_voxel_indices


def get_device(device: Union[torch.device, str] = "cuda") -> torch.device:
    """
    get the device of the input string or torch.device
    Args:
        device: string or torch device

    Returns:
        the device format that can be used for torch tensors
    """
    if isinstance(device, str):
        assert device == "cuda" or device == "cuda:0" or device == "cuda:1" or device == "cpu", \
            "device should only be 'cuda' or 'cpu' "
    device = torch.device(device)
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def to_device(x, device):
    """
    put the input to a torch tensor in the given device
    Args:
        x: list, tuple of torch tensors, dict of torch tensors or torch tensors
        device: pytorch device
    Returns:
        the input data in the given device
    """
    if isinstance(x, list):
        x = [to_device(item, device) for item in x]
    elif isinstance(x, tuple):
        x = (to_device(item, device) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if device == "cuda":
            x = x.cuda()
        else:
            x = x.to(device)
    return x


def release_cuda(x):
    """
    put the torch tensors from cuda to numpy
    Args:
        x: in put torch tensor, list of, dict of or tuple of torch tensors in cuda

    Returns:
        input data in local numpy
    """
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def transform_pts(pts, transform, rotation=None, translation=None):
    if rotation is not None:
        pass
    else:
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
    points = np.matmul(pts, rotation.transpose(-1, -2)) + translation
    return points


def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points