import pickle
from os.path import join

import numpy as np
from src.data_loader.base_class import BaseDataClass
from src.data_loader.recall_dataset import RecallData
from utils.config import FrameInfo, DatasetConfig, PointFolder, DataDict


class RerankDataset(BaseDataClass):
    def __init__(self, cfg: DatasetConfig):
        super(RerankDataset, self).__init__(cfg)
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
        """
        Get each data set including the current frame, positive frames and also negative frames

        Args:
        index: the index of all the frames

        Returns:
        a list of frame data, including the current frame, positive frames and also negative frames
        """
        all_labels = []
        all_frame_ids = []
        all_gds = []
        all_fts = []
        all_geos = []
        all_nbs = []
        all_sims = []

        # Current/query pcd indices
        scene_idx, frame_idx = self.all_indices[index]
        global_descriptor, fts, geos, neighbors, similarities = self.load_data(scene_idx=scene_idx, frame_idx=frame_idx)
        all_labels.append(-1)
        all_frame_ids.append([scene_idx, frame_idx])
        all_gds.append(global_descriptor)
        all_fts.append(fts)
        all_geos.append(geos)
        all_nbs.append(neighbors)
        all_sims.append(similarities)

        # Positive frames
        positive_cases = self.frame_info[self.scenes[scene_idx]][FrameInfo.posIds][frame_idx]
        selected_positive_indices = np.random.choice(positive_cases, self.num_pos_samples, replace=False).tolist()
        for positive_index in selected_positive_indices:
            global_descriptor, fts, geos, neighbors, similarities = self.load_data(scene_idx=scene_idx, frame_idx=positive_index)
            all_labels.append(1)
            all_frame_ids.append([scene_idx, positive_index])
            all_gds.append(global_descriptor)
            all_fts.append(fts)
            all_geos.append(geos)
            all_nbs.append(neighbors)
            all_sims.append(similarities)

        # Negative cases
        negative_num = self.num_neg_same_scenes_samples + self.num_neg_other_scenes_samples
        negative_cases = self.frame_info[self.scenes[scene_idx]][FrameInfo.negIds][frame_idx]
        negative_indices = list(range(len(negative_cases)))
        selected_negative_indices = np.random.choice(negative_indices, negative_num, replace=False).tolist()
        for negative_index in selected_negative_indices:
            scene, f_id = negative_cases[negative_index]
            s_id = self.scenes.index(scene)
            global_descriptor, fts, geos, neighbors, similarities = self.load_data(scene_idx=s_id, frame_idx=f_id)
            all_labels.append(0)
            all_frame_ids.append([s_id, f_id])
            all_gds.append(global_descriptor)
            all_fts.append(fts)
            all_geos.append(geos)
            all_nbs.append(neighbors)
            all_sims.append(similarities)

        return {
            DataDict.binary_label: all_labels,
            DataDict.global_descriptor: all_gds,
            DataDict.features: all_fts,
            DataDict.geometrics: all_geos,
            DataDict.neighbors: all_nbs,
            DataDict.similarities: all_sims,
            DataDict.frame_id: np.array(all_frame_ids)
        }



