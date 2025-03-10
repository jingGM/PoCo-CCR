import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess
import shutil
import numpy as np
import torch
from tqdm import tqdm
from src.functions import Distance
from utils.config import TrainingConfig, LossNames, ModelType, DataDict, DatasetType, ScheduleMethods, FrameInfo
from utils.functions import get_device, to_device, release_cuda
from utils.logger import SummaryBoard, configure_logger
from src.model import get_model
from src.data_loader.data_loader import registration_collate_fn_stack_mode
from src.data_loader.base_class import SimpleDataClass
from src.data_loader.recall_dataset import RecallData


class RerankDataGenerator:
    def __init__(self, device, model_cfg, snapshot, distance_type, sanity_check, data_loader_cfg, rerank_num):
        if device == "cuda":
            self.device = "cuda:0"
        else:
            self.device = get_device(device=device)

        self.model = get_model(config=model_cfg, device=self.device)
        self.load_snapshot(snapshot)
        self.model.to(self.device)
        self.model.eval()

        self.distance = Distance(distance_type=distance_type, distance=True)

        self.sanity = sanity_check
        self.data_loader_cfg = copy.deepcopy(data_loader_cfg)
        self.data_loader_cfg.num_pos_samples = 0
        self.root = self.data_loader_cfg.root
        self.rerank_root = join(self.root, "rerank")
        # if exists(self.rerank_root):
        #     try:
        #         shutil.rmtree(self.rerank_root)
        #     except OSError as e:
        #         print("Error: %s - %s." % (e.filename, e.strerror))

        # self.rerank_frame_num = rerank_num

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            message = 'Missing keys: {}'.format(missing_keys)
        if len(unexpected_keys) > 0:
            message = 'Unexpected keys: {}'.format(unexpected_keys)
        return state_dict

    def clean_info(self, frame_info):
        scenes = list(frame_info.keys())
        for idx_scene in range(len(scenes)):
            poses = frame_info[scenes[idx_scene]][FrameInfo.poses]
            for i in range(len(poses)):
                frame_info[scenes[idx_scene]][FrameInfo.posIds][i] = []
                frame_info[scenes[idx_scene]][FrameInfo.negIds][i] = []
        return frame_info

    def generate_data_type(self, data_type, save_files=True):
        self.data_loader_cfg.type = data_type
        data_loader = RecallData(cfg=self.data_loader_cfg)

        global_descriptors = []
        frames = []
        for iteration, data_dict in enumerate(tqdm(data_loader, desc="generate rerank data {}".format(data_type))):
            if iteration >= len(data_loader):
                break
            input_dict = registration_collate_fn_stack_mode([data_dict])
            input_dict = to_device(input_dict, device=self.device)
            results = self.model.forward_recall_features(input_data=input_dict)

            global_descriptor, fts, geos, neighbors, similarities = results[DataDict.fine_input]
            global_descriptor = global_descriptor.squeeze(0).detach().cpu().numpy()
            global_descriptors.append(global_descriptor)

            fts = fts.squeeze(0).detach().cpu().numpy()
            geos = geos.squeeze(0).detach().cpu().numpy()
            neighbors = neighbors.squeeze(0).detach().cpu().numpy()
            similarities = similarities.squeeze(0).detach().cpu().numpy()
            frame_id = results[DataDict.frame_id].detach().cpu().numpy()

            for idx in range(len(frame_id)):
                s_id, f_id = frame_id[idx]
                scene = data_loader.scenes[s_id]
                frames.append([scene, f_id])
                if save_files:
                    folder_name = join(self.rerank_root, str(scene))
                    if not exists(folder_name):
                        os.makedirs(folder_name)
                    with open(join(folder_name, "{}.pkl".format(f_id)), 'wb') as handle:
                        pickle.dump([global_descriptor[idx], fts[idx], geos[idx], neighbors[idx], similarities[idx]],
                                    handle, protocol=pickle.HIGHEST_PROTOCOL)
        global_descriptors = torch.from_numpy(np.concatenate(global_descriptors, axis=0))

        frame_info = self.clean_info(copy.deepcopy(data_loader.frame_info))
        all_frames_info = {}
        for index in tqdm(range(global_descriptors.shape[0]), desc="update frame info {}".format(data_type)):
            scene, f_id = frames[index]
            current_global = global_descriptors[index].unsqueeze(0)
            distances = self.distance(current_global.repeat(global_descriptors.shape[0], 1),
                                      global_descriptors).numpy()
            topk_indices = np.argsort(distances)

            all_indices = []
            for i in range(len(topk_indices)):
                all_indices.append(frames[topk_indices[i]])
            if scene not in all_frames_info.keys():
                all_frames_info[scene] = [[] for _ in range(len(data_loader.frame_info[scene][FrameInfo.posIds]))]
            all_frames_info[scene][f_id] = copy.deepcopy(all_indices)

            gt_positives = data_loader.frame_info[scene][FrameInfo.posIds][f_id]
            gt_negatives = data_loader.frame_info[scene][FrameInfo.negIds][f_id]

            scene_frames = min(len(data_loader.frame_info[scene][FrameInfo.posIds]) * 2 + 1, len(topk_indices))
            positives = []
            negatives = []
            for i in range(scene_frames):
                other_s, other_f = frames[topk_indices[i]]
                if scene == other_s:
                    if other_f in gt_positives:
                        positives.append(other_f)
                    elif other_f in gt_negatives:
                        negatives.append([other_s, other_f])
                elif scene != other_s:
                    negatives.append([other_s, other_f])
            frame_info[scene][FrameInfo.posIds][f_id] = copy.deepcopy(positives)
            frame_info[scene][FrameInfo.negIds][f_id] = copy.deepcopy(negatives)
        with open(join(self.rerank_root, "{}.pkl".format(data_type)), 'wb') as handle:
            pickle.dump(frame_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(join(self.rerank_root, "recall.pkl"), 'wb') as handle:
            pickle.dump(all_frames_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, types=None, save_files=True):
        if self.sanity:
            data_types = [DatasetType.sanity_check]
        else:
            if types is None:
                data_types = [DatasetType.training, DatasetType.evaluation, DatasetType.test]
            else:
                data_types = types
        for data_type in data_types:
            self.generate_data_type(data_type=data_type, save_files=save_files)