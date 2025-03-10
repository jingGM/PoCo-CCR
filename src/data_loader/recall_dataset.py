import copy
import os.path
import pickle
import warnings
from os.path import join, exists
import random
from typing import Tuple, List

import numpy as np
import numpy.random
from tqdm import tqdm

from src.data_loader.base_class import SimpleDataClass
from utils.config import DatasetConfig, RecallDataType, DataDict, FrameInfo
from utils.functions import transform_pts


class RecallData(SimpleDataClass):
    def __init__(self, cfg: DatasetConfig):
        """Initialize parameters of ScannetCoCs dataset.

        Args:
          cfg: The configuration of the dataset loader

        Raises:
          ValueError: If the dataset root cannot be found
        """
        self.selected_scenes = []
        self.selected_frames = []
        self.gt = {}
        self.centers = {}

        self.frame_distance = 3

        super(RecallData, self).__init__(cfg)
        self.num_pos_samples = 0
        self.recall_database = cfg.recall_database
        self.reset()

    def _load_all_selected_frames(self, sid):
        scene = self.scenes[sid]
        centers = []
        poses = self.frame_info[scene][FrameInfo.poses]
        for idx_frame in range(len(poses)):
            points = self.load_data(sid, frame_idx=idx_frame)
            pose = poses[idx_frame]
            t_pts = transform_pts(pts=points[:, :3], transform=pose)
            centers.append(np.mean(t_pts, axis=0))

        selected_frames = []
        selected_centers = []
        for i in range(len(centers)):
            if len(selected_frames) == 0:
                selected_frames.append(i)
                selected_centers.append(centers[i])
            else:
                distances = np.linalg.norm(np.stack(selected_centers) - centers[i][None, ...], axis=1)
                if np.any(distances < self.frame_distance):
                    continue
                else:
                    selected_frames.append(i)
                    selected_centers.append(centers[i])
        self.centers[scene] = centers
        return selected_frames

    def reset(self):
        self.selected_scenes = []
        self.all_indices = self.get_all_indices()

        # get all key frames
        self.selected_frames = []
        # for scene_idx in range(len(self.scenes)):
            # scene = self.scenes[scene_idx]
        for sindex in tqdm(range(len(self.selected_scenes)), desc="calculate recall frames: "):
            scene = self.selected_scenes[sindex]
            scene_idx = self.scenes.index(scene)
            if self.recall_database == RecallDataType.positive_key_frames:
                frames = self.frame_info[scene][RecallDataType.positive_key_frames]
            elif self.recall_database == RecallDataType.negative_key_frames:
                frames = self.frame_info[scene][RecallDataType.negative_key_frames]
            elif self.recall_database == RecallDataType.distance:
                frames = self._load_all_selected_frames(sid=scene_idx)
            else:
                raise ValueError("the recall database type {} is not defined".format(self.recall_database))
            for frame_idx in frames:
                self.selected_frames.append([scene_idx, int(frame_idx)])

        # calculate the ground truth from the database
        self.gt = {}
        for gt_idx in tqdm(range(len(self.all_indices)),
                           desc="calculate ground truth distance for each frame given the candidates: "):
            scene_id, frame_id = self.all_indices[gt_idx]
            positives = self.frame_info[self.scenes[scene_id]][FrameInfo.posIds][frame_id]
            negatives = self.frame_info[self.scenes[scene_id]][FrameInfo.negIds][frame_id]
            frame_gt = []
            for s, f in self.selected_frames:
                if self.recall_database == RecallDataType.positive_key_frames:
                    if (s == scene_id) and (f in positives):
                        frame_gt.append(f)
                elif self.recall_database == RecallDataType.negative_key_frames:
                    if (s == scene_id) and (f not in negatives) and (f != frame_id):
                        frame_gt.append(f)
                elif self.recall_database == RecallDataType.distance:
                    if (s == scene_id) and (f in positives):
                        current_center = self.centers[self.scenes[s]][frame_id]
                        other_center = self.centers[self.scenes[s]][f]
                        if np.linalg.norm(current_center - other_center) <= self.frame_distance:
                            frame_gt.append(f)
            self.gt[(scene_id, frame_id)] = frame_gt

    def get_all_indices(self):
        """get all indices of the scenes, and remove the frames that don't have enough positive pairs
        Args:
            num_pos_samples: the index the scene

        Returns:
            all the available scene and frame pairs
        """
        all_indices = []
        scenes = copy.deepcopy(self.scenes)
        random.shuffle(scenes)
        self.selected_scenes = []
        for scene in scenes:
            idx_scene = self.scenes.index(scene)
            if len(all_indices) < self.max_iteration:
                self.selected_scenes.append(scene)
                positives = self.frame_info[self.scenes[idx_scene]][FrameInfo.posIds]
                for idx_frame in range(len(positives)):
                    if len(all_indices) < self.max_iteration:
                        current_positives = positives[idx_frame]
                        if len(current_positives) >= self.num_pos_samples:
                            all_indices.append([idx_scene, idx_frame])
            else:
                break
        return np.array(all_indices)

    def process_frames(self, frames):
        data_output = {}
        ids = []
        all_fts = []
        for scene_ind, frame_ind in frames:
            current_data = self.load_data(scene_idx=scene_ind, frame_idx=frame_ind)
            ids.append([scene_ind, frame_ind])
            all_fts.append(current_data)
        data_output[DataDict.frame_id] = np.array(ids)
        data_output[DataDict.pts_features] = all_fts
        return data_output


class RerankRecallDataset(RecallData):
    def __init__(self, cfg: DatasetConfig):
        super(RerankRecallDataset, self).__init__(cfg)
        with open(join(self.root, "rerank", "{}.pkl".format(self.data_type)), 'rb') as handle:
            self.frame_info = pickle.load(handle)
        self.scenes = list(self.frame_info.keys())
        self.all_indices = self.get_all_indices()

    def __len__(self):
        """
        Return the length of data here

        Returns:
        The length of the dataset
        """
        return len(self.all_indices)

    def load_data(self, scene_idx, frame_idx):
        scene = self.scenes[scene_idx]
        with open(join(self.root, "rerank", str(scene), "{}.pkl".format(frame_idx)), 'rb') as handle:
            data = pickle.load(handle)

        global_descriptor, fts, geos, neighbors, similarities = data
        return global_descriptor, fts, geos, neighbors, similarities

    def __getitem__(self, index):
        scene_idx, frame_idx = self.all_indices[index]
        global_descriptor, fts, geos, neighbors, similarities = self.load_data(scene_idx, frame_idx)
        return scene_idx, frame_idx, global_descriptor, fts, geos, neighbors, similarities
