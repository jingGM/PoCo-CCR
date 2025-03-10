import copy
import pickle
from os.path import join
import random

import numpy as np
import torch
from tqdm import tqdm

from src.data_loader.base_class import BaseDataClass
from utils.config import FrameInfo, DatasetConfig, PointFolder, DataDict


class CoarseDataset(BaseDataClass):
    def __init__(self, cfg: DatasetConfig):
        super(CoarseDataset, self).__init__(cfg)
        self.instance_size = cfg.instance_size

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
        for scene in scenes:
            idx_scene = self.scenes.index(scene)
            if len(all_indices) < self.max_iteration:
                positives = self.frame_info[self.scenes[idx_scene]][FrameInfo.posIds]
                for idx_frame in range(len(positives)):
                    if len(all_indices) < self.max_iteration:
                        current_positives = positives[idx_frame]
                        if len(current_positives) >= self.num_pos_samples:
                            all_indices.append([idx_scene, idx_frame])
                    else:
                        break
            else:
                break
        return np.array(all_indices)

    def batch_length(self):
        length = int(float(len(self.all_indices)) / float(self.instance_size))
        left = float(len(self.all_indices)) % float(self.instance_size)
        if left == 0:
            return length
        else:
            return length + 1

    def get_batch(self, index):
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

    def reset(self):
        self.all_indices = self.get_all_indices()

    def __len__(self):
        """
        Return the length of data here

        Returns:
        The length of the dataset
        """
        return len(self.all_indices)

    def __getitem__(self, index):
        """
        Get each data set including the current frame, positive frames and also negative frames

        Args:
        index: the index of all the frames

        Returns:
        a list of frame data, including the current frame, positive frames and also negative frames
        """
        all_fts = []
        all_labels = []
        all_frame_ids = []

        # Current/query pcd indices
        scene_idx, frame_idx = self.all_indices[index]
        current_pts = self.load_data(scene_idx=scene_idx, frame_idx=frame_idx)
        all_labels.append(-1)
        all_fts.append(current_pts)
        all_frame_ids.append([scene_idx, frame_idx])

        # Positive frames
        positive_cases = self.frame_info[self.scenes[scene_idx]][FrameInfo.posIds][frame_idx]
        selected_positive_indices = np.random.choice(positive_cases, self.num_pos_samples, replace=False).tolist()
        for positive_index in selected_positive_indices:
            positive_data = self.load_data(scene_idx=scene_idx, frame_idx=positive_index)
            all_labels.append(1)
            all_fts.append(positive_data)
            all_frame_ids.append([scene_idx, positive_index])

        # Negative cases
        negative_cases = self.frame_info[self.scenes[scene_idx]][FrameInfo.negIds][frame_idx]
        negative_num = self.num_neg_same_scenes_samples + self.num_neg_other_scenes_samples
        if self.data_mining:
            negative_cases = self.frame_info[self.scenes[scene_idx]][FrameInfo.negIds][frame_idx]
            negative_indices = list(range(len(negative_cases)))
            selected_negative_indices = np.random.choice(negative_indices, negative_num, replace=False).tolist()
            for negative_index in selected_negative_indices:
                s_id, f_id = negative_cases[negative_index]
                negative_data = self.load_data(scene_idx=s_id, frame_idx=f_id)
                all_labels.append(0)
                all_fts.append(negative_data)
                all_frame_ids.append([s_id, f_id])
        else:
            # Negative frames in the same scene
            if self.num_neg_same_scenes_samples > 0:
                if len(negative_cases) >= self.num_neg_same_scenes_samples:
                    selected_negative_indices = np.random.choice(negative_cases, self.num_neg_same_scenes_samples,
                                                                 replace=False).tolist()
                else:
                    selected_negative_indices = negative_cases
                for negative_index in selected_negative_indices:
                    negative_data = self.load_data(scene_idx=scene_idx, frame_idx=negative_index)
                    all_labels.append(0)
                    all_fts.append(negative_data)
                    all_frame_ids.append([scene_idx, negative_index])
            else:
                selected_negative_indices = []

            # Negative frames in the different scenes
            other_scenarios = negative_num - len(selected_negative_indices)
            potential_scenarios_indices = self.all_indices[np.where(self.all_indices[:, 0] != scene_idx)]
            other_indices = []
            while len(other_indices) < other_scenarios:
                n_idx = np.random.choice(np.array(list(range(len(potential_scenarios_indices)))), 1, replace=False)[0]
                ns_id, nf_id = potential_scenarios_indices[n_idx]
                if self.scenes[ns_id] not in self.frame_info[self.scenes[scene_idx]][FrameInfo.simScenes]:
                    other_indices.append([ns_id, nf_id])

            for index in range(len(other_indices)):
                negative_data = self.load_data(scene_idx=other_indices[index][0], frame_idx=other_indices[index][1])
                all_labels.append(0)
                all_fts.append(negative_data)
                all_frame_ids.append([other_indices[index][0], other_indices[index][1]])

        return {
            DataDict.binary_label: all_labels,
            DataDict.pts_features: all_fts,
            DataDict.frame_id: np.array(all_frame_ids)
        }
