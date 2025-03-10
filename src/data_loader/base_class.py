import copy
import pickle
from os.path import join, exists

import cv2
import torch
from tqdm import tqdm
import numpy as np
import random
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import open3d as o3d

from utils.config import FrameInfo, DatasetConfig, DataDict
from src.functions import Distance


class BaseDataClass(Dataset, ABC):
    def __init__(self, cfg: DatasetConfig):
        self.point_folder = cfg.point_folder
        self.network_type = cfg.network_type
        self.root = cfg.root
        self.scans_root = join(self.root, "scans")
        if not exists(self.scans_root):
            raise ValueError("The dataset's scans path \"{}\" cannot be found".format(self.scans_root))

        self.use_hard_positive = cfg.use_hard_positive
        self.use_color = cfg.use_color
        self.use_normals = cfg.use_normals
        self.use_geos = cfg.use_geos
        self.use_semantic = cfg.use_semantic
        self.max_points_num = cfg.max_points_num
        self.max_iteration = cfg.max_iteration

        self.num_pos_samples = cfg.num_pos_samples
        self.num_neg_other_scenes_samples = cfg.num_neg_other_scenes_samples
        self.num_neg_same_scenes_samples = cfg.num_neg_same_scenes_samples

        self.distance = Distance(distance_type=cfg.distance_type, distance=True)

        self.negative_mining_num_hard = int(cfg.negative_mining_num * cfg.negative_mining_num_ratio)
        self.negative_mining_num_other = cfg.negative_mining_num - self.negative_mining_num_hard
        self.data_mining = False

        self.data_type = None
        self.scenes = None
        self.original_frame_info = None
        self.frame_info = None
        self.update_data_type(cfg.type)
        self.all_indices = self.get_all_indices()

    def reset(self):
        pass

    def update_data_type(self, data_type):
        self.data_type = data_type
        with open(join(self.root, "{}.pkl".format(self.data_type)), 'rb') as handle:
            data = pickle.load(handle)
        self.scenes = list(data.keys())
        self.original_frame_info = data
        self.frame_info = copy.deepcopy(self.original_frame_info)

    def get_all_indices(self):
        """get all indices of the scenes, and remove the frames that don't have enough positive pairs
        Args:
            num_pos_samples: the index the scene

        Returns:
            all the available scene and frame pairs
        """
        all_indices = []
        for idx_scene in range(len(self.scenes)):
            positives = self.frame_info[self.scenes[idx_scene]][FrameInfo.posIds]
            for idx_frame in range(len(positives)):
                current_positives = positives[idx_frame]
                if len(current_positives) >= self.num_pos_samples:
                    all_indices.append([idx_scene, idx_frame])
        return np.array(all_indices)

    def load_data(self, scene_idx, frame_idx) -> np.ndarray:
        """
        Process data given the scene and frame indices. After procession a dict will be returned

        Args:
        scene_index: the index of the scenes
        frame_index: the index of the frame

        Returns:
        a diction of the frame data
        """
        if self.network_type == 1:
            points_dir = join(self.scans_root,
                              "{}/rgbd/{}.pkl".format(self.scenes[scene_idx], frame_idx))
            with open(points_dir, 'rb') as handle:
                rgbd = pickle.load(handle)
            new_rgb = cv2.resize(rgbd["rgb"], (640, 480), interpolation=cv2.INTER_AREA)
            return np.moveaxis(new_rgb, -1, 0)
        else:
            points_dir = join(self.scans_root, "{}/{}/{}.pkl".format(self.scenes[scene_idx], self.point_folder, frame_idx))
            with open(points_dir, 'rb') as handle:
                pts = pickle.load(handle)

            if pts.shape[0] > self.max_points_num:
                indices = list(range(pts.shape[0]))
                random.shuffle(indices)
                selected_indices = indices[:self.max_points_num]
                pts = pts[selected_indices]

            if self.use_normals:
                geos = pts[:, :6]
            else:
                geos = pts[:, :3]

            if self.use_color:
                colors = pts[:, 6:9]
                data = np.concatenate((geos, colors), axis=1)
            else:
                if not self.use_geos:
                    colors = np.ones((pts.shape[0], 1))
                    data = np.concatenate((geos, colors), axis=1)
                else:
                    data = geos

            if self.use_semantic and pts.shape[1] == 10:
                data = np.concatenate((data, pts[:, -1]), axis=1)
            return data

    def update_dataset(self, file_name):
        self.data_mining = True
        folder = join(self.root, file_name)
        with open(join(folder, "mining.pkl"), 'rb') as handle:
            data = pickle.load(handle)
        frame_ids = data["ids"]
        file_num = data["file_num"]
        global_ids = data["global_ids"]
        # global_descriptors = torch.from_numpy(data["descriptors"])
        for index in tqdm(range(len(frame_ids)), desc="update negative mining in dataloader"):
            scene, f_id = frame_ids[index]

            glob_f, glob_id = global_ids[index]
            with open(join(folder, "{}.pkl".format(glob_f)), 'rb') as handle:
                current_global = pickle.load(handle)[glob_id].unsqueeze(0)

            distances = []
            for f in range(file_num):
                with open(join(folder, "{}.pkl".format(f)), 'rb') as handle:
                    global_descriptors = pickle.load(handle)
                distance = self.distance(current_global.repeat(global_descriptors.shape[0], 1), global_descriptors).numpy()
                distances.append(distance)
            topk_indices = np.argsort(np.concatenate(distances, axis=0))

            hard_negatives = []
            start_index = 0
            while len(hard_negatives) < self.negative_mining_num_hard:
                other_idx = topk_indices[start_index]
                other_s, other_f = frame_ids[other_idx]
                if (other_s == scene) and (other_f in self.original_frame_info[scene][FrameInfo.negIds][f_id]):
                    hard_negatives.append([self.scenes.index(other_s), other_f])
                elif (other_s != scene) and (other_s not in self.original_frame_info[scene][FrameInfo.simScenes]):
                    hard_negatives.append([self.scenes.index(other_s), other_f])
                else:
                    pass
                start_index += 1

            other_candidates = topk_indices[start_index:].tolist()
            while len(hard_negatives) < self.negative_mining_num_hard + self.negative_mining_num_other:
                random.shuffle(other_candidates)
                other_idx = topk_indices[other_candidates.pop(0)]
                other_s, other_f = frame_ids[other_idx]

                if (other_s == scene) and (other_f in self.original_frame_info[scene][FrameInfo.negIds][f_id]):
                    hard_negatives.append([self.scenes.index(other_s), other_f])
                elif (other_s != scene) and (other_s not in self.original_frame_info[scene][FrameInfo.simScenes]):
                    hard_negatives.append([self.scenes.index(other_s), other_f])
                else:
                    pass
            self.frame_info[scene][FrameInfo.negIds][f_id] = hard_negatives


class SimpleDataClass(BaseDataClass):
    def __init__(self, cfg):
        super(SimpleDataClass, self).__init__(cfg)
        self.instance_size = cfg.instance_size

    def __len__(self):
        length = int(float(len(self.all_indices)) / float(self.instance_size))
        left = float(len(self.all_indices)) % float(self.instance_size)
        if left == 0:
            return length
        else:
            return length + 1

    def __getitem__(self, index):
        start_index = index * self.instance_size
        data_output = {}
        ids = []
        all_data = []
        for idx in range(self.instance_size):
            if idx + start_index >= len(self.all_indices):
                break
            else:
                scene_idx, frame_idx = self.all_indices[start_index + idx]
            current_data = self.load_data(scene_idx=scene_idx, frame_idx=frame_idx)
            ids.append([scene_idx, frame_idx])
            all_data.append(current_data)
        data_output[DataDict.frame_id] = np.array(ids)
        data_output[DataDict.pts_features] = all_data
        return data_output

