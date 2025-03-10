import copy
import os
import pickle
import shutil
from os.path import join, exists

import cv2
import imageio
# from flopth import flopth
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from src.data_loader.recall_dataset import RecallData
from utils.config import DistanceType, DataDict, LossNames, ModelType, GeometricType
from utils.functions import to_device
from src.data_loader.data_loader import registration_collate_fn_stack_mode
# from src.data_loaders.recall_dataset import RecallData
from src.functions import Distance
# from comparisons.ransac import calculate_ransac, calculate_teaser
# from pypapi import events, papi_high as high


class TripletLoss(nn.Module):
    def __init__(self, margin, reduction, distance):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

        self.dis = distance
        self.triplet = nn.TripletMarginWithDistanceLoss(margin=self.margin, reduction=self.reduction,
                                                        distance_function=self.dis)

    def forward(self, embedding, labels):
        positives = []
        negatives = []
        current_frame = embedding[0].unsqueeze(0)
        for f in range(1, len(labels), 1):
            if labels[f] < 0.5:
                negatives.append(embedding[f])
            else:
                positives.append(embedding[f])
        negatives = torch.stack(negatives, dim=0)
        positives = torch.stack(positives, dim=0)

        loss = self.triplet(current_frame.repeat(positives.shape[0], 1), positives, negatives)
        return loss


class CircleLoss(nn.Module):
    """
    According to https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py
    """

    def __init__(self, margin, reduction, similarity, circular_gamma):
        super(CircleLoss, self).__init__()
        self.gamma = circular_gamma
        self.margin = margin
        self.reduction = reduction
        self.soft_plus = nn.Softplus()
        self.similarity = similarity

    def forward(self, embedding, labels):
        distances = self.similarity(embedding[0, :].unsqueeze(0), embedding[1:, :])
        positives = []
        negatives = []
        for f in range(1, len(labels), 1):
            if labels[f] < 0.5:
                negatives.append(distances[f - 1])
            else:
                positives.append(distances[f - 1])

        delta_p = 1 - self.margin
        delta_n = self.margin

        sp, sn = torch.stack(positives, dim=0), torch.stack(negatives, dim=0)

        ap = torch.clamp_min(-sp.detach() + 1 - self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)
        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        # ap = torch.clamp_min(1 - sp.detach(), min=0.)
        # an = torch.clamp_min(sn.detach(), min=0.)
        # logit_p = - ap * (sp - delta_p) * self.gamma
        # logit_n = an * (sn - delta_n) * self.gamma
        # loss = torch.log(1 + torch.sum(torch.exp(logit_p)) * torch.sum(torch.exp(logit_n)))
        return loss


class Loss(nn.Module):
    def __init__(self, cfg, model_type):
        super(Loss, self).__init__()
        self.model_type = model_type
        if self.model_type == ModelType.coarse_matching:
            self.triplet_ratio = cfg.triplet_ratio
            self.circle_ratio = cfg.circle_ratio
            self.circle_dis = Distance(distance_type=cfg.distance_type, distance=False)
            self.circle_loss = CircleLoss(margin=cfg.margin, reduction=cfg.reduction,
                                          similarity=self.circle_dis, circular_gamma=cfg.circular_gamma)
            self.triplet_dis = Distance(distance_type=cfg.distance_type, distance=True)
            self.triplet_loss = TripletLoss(margin=cfg.margin, reduction=cfg.reduction,
                                            distance=self.triplet_dis)
        elif self.model_type == ModelType.fine_matching:
            self.loss_func = nn.CrossEntropyLoss()
        elif self.model_type == ModelType.all_matching:
            self.triplet_ratio = cfg.triplet_ratio
            self.circle_ratio = cfg.circle_ratio
            self.circle_dis = Distance(distance_type=cfg.distance_type, distance=False)
            self.circle_loss = CircleLoss(margin=cfg.margin, reduction=cfg.reduction,
                                          similarity=self.circle_dis, circular_gamma=cfg.circular_gamma)
            self.triplet_dis = Distance(distance_type=cfg.distance_type, distance=True)
            self.triplet_loss = TripletLoss(margin=cfg.margin, reduction=cfg.reduction,
                                            distance=self.triplet_dis)
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("the model type {} is not defined".format(self.model_type))

    def _forward_coarse_loss(self, embeddings, labels):
        B, K = labels.shape
        embeddings = embeddings.view(B, K, embeddings.shape[-1])
        triplet_loss = 0
        circle_loss = 0
        for i in range(B):
            triplet_loss += self.triplet_loss(embedding=embeddings[i], labels=labels[i])
            circle_loss += self.circle_loss(embedding=embeddings[i], labels=labels[i])
        loss = self.circle_ratio * circle_loss + self.triplet_ratio * triplet_loss
        loss /= float(B)
        circle_loss /= float(B)
        triplet_loss /= float(B)
        return {LossNames.coarse_loss: loss,
                LossNames.triplet_loss: triplet_loss,
                LossNames.circle_loss: circle_loss}

    def _forward_fine_loss(self, embeddings, labels):
        B, K = embeddings.shape
        loss = 0
        for i in range(B):
            loss += self.loss_func(embeddings[i].view(-1), labels[i][1:].view(-1))
        return {LossNames.fine_loss: loss}

    def forward(self, input_dict):
        coarse_embeddings = input_dict[DataDict.pts_coarse_embeds]  # (B*K, D)
        fine_embeddings = input_dict[DataDict.local_score]  # (B*K, D)
        labels = input_dict[DataDict.binary_label]
        loss = 0
        loss_info = {}
        if coarse_embeddings is not None:
            coarse_loss = self._forward_coarse_loss(embeddings=coarse_embeddings, labels=labels)
            loss += coarse_loss[LossNames.coarse_loss]
            loss_info.update(coarse_loss)
        if fine_embeddings is not None:
            fine_loss = self._forward_fine_loss(embeddings=fine_embeddings, labels=labels)
            loss += fine_loss[LossNames.fine_loss]
            loss_info.update(fine_loss)
        loss_info.update({LossNames.loss: loss})
        return loss_info

    @torch.no_grad()
    def _forward_embeddings(self, embeddings, model_type):
        similarities = []
        for b in range(embeddings.shape[0]):
            if model_type == ModelType.coarse_matching:
                similarity = self.circle_dis(embeddings[b][0].unsqueeze(0).repeat((embeddings[b].shape[0] - 1, 1)),
                                             embeddings[b][1:])
            elif model_type == ModelType.fine_matching:
                similarity = embeddings[b]
            elif model_type == ModelType.all_matching:
                similarity = embeddings[b]
            else:
                raise ValueError("the model type {} is not defined".format(model_type))
            similarities.append(similarity)
        return torch.stack(similarities, dim=0)

    @torch.no_grad()
    def _forward_single_evaluate(self, similarities, labels):
        B = similarities.shape[0]
        K = labels.shape[1]
        fp = 0
        fn = 0
        p = 0
        neg_p = 0
        for b in range(B):
            false_positives = 0.
            true_positives = 0.
            false_negatives = 0.
            true_negatives = 0.

            negative_cases = 0.
            positive_cases = 0.
            for f in range(0, K - 1):
                label = labels[b][f + 1]
                similarity = similarities[b][f]

                if label > 0.5:
                    positive_cases += 1.0
                else:
                    negative_cases += 1.0

                if similarity > 0.5:
                    if label > 0.5:
                        true_positives += 1.0
                    else:
                        false_positives += 1.0
                elif similarity < 0.5:
                    if label > 0.5:
                        false_negatives += 1.0
                    else:
                        true_negatives += 1.0

            all_detected_positives = true_positives + false_positives
            all_detected_negatives = true_negatives + false_negatives

            p += true_positives / all_detected_positives if all_detected_positives > 0 else 0
            neg_p += true_negatives / all_detected_negatives if all_detected_negatives > 0 else 0
            fn += false_negatives / positive_cases if positive_cases > 0 else 0
            fp += false_positives / negative_cases if negative_cases > 0 else 0

        return {
            LossNames.false_neg: fn / B,
            LossNames.false_pos: fp / B,
            LossNames.precision: p / B,
            LossNames.neg_precision: neg_p / B,
        }

    @torch.no_grad()
    def _update_dict(self, output_dict, model_type):
        if model_type == ModelType.coarse_matching:
            heading = "coarse_"
        elif model_type == ModelType.fine_matching:
            heading = "fine_"
        elif model_type == ModelType.all_matching:
            heading = "fine_all_"
        else:
            raise ValueError("the model type {} is not defined".format(model_type))
        new_dict = {}
        for key, val in output_dict.items():
            new_dict[heading + key] = val
        return new_dict

    @torch.no_grad()
    def evaluate(self, input_dict):
        labels = input_dict[DataDict.binary_label]
        coarse_embeddings = input_dict[DataDict.pts_coarse_embeds]  # (B*K, D)
        fine_embeddings = input_dict[DataDict.local_score]  # (B, K)

        output_dict = {}
        if coarse_embeddings is not None:
            coarse_embeddings = coarse_embeddings.view(labels.shape[0], labels.shape[1], coarse_embeddings.shape[-1])
            coarse_similarities = self._forward_embeddings(embeddings=coarse_embeddings,
                                                           model_type=ModelType.coarse_matching)
            coarse_dict = self._forward_single_evaluate(similarities=coarse_similarities, labels=labels)
            output_dict.update(self._update_dict(coarse_dict, model_type=ModelType.coarse_matching))
        if fine_embeddings is not None:
            # fine_embeddings = fine_embeddings.view(labels.shape[0], labels.shape[1], fine_embeddings.shape[-1])
            fine_similarities = self._forward_embeddings(embeddings=fine_embeddings, model_type=ModelType.fine_matching)
            fine_dict = self._forward_single_evaluate(similarities=fine_similarities, labels=labels)
            output_dict.update(self._update_dict(fine_dict, model_type=ModelType.coarse_matching))
        return output_dict


class RecallCalculation:
    def __init__(self, distance_type, max_recall_num, model_type, save_stage1=False, verbose=False, instance_num=8, rerank_frame_num=20,
                 save_results=False, saving_dir="test", output_dir="", geometric=False):
        self.fine_inputs_index = 0
        self.database_fine_index = 0
        self.global_descriptors = []
        self.frames = []
        self.output_dir = output_dir
        self.database_output = join(output_dir, "database")
        self.images_output = join(output_dir, "images")
        self.verbose = verbose
        self.save_stage1 = save_stage1

        if not exists(self.database_output):
            # try:
            #     shutil.rmtree(self.output_dir)
            # except OSError as e:
            #     print("Error: %s - %s." % (e.filename, e.strerror))
            os.makedirs(self.database_output)
            os.makedirs(self.images_output)

        self.distance = Distance(distance_type=distance_type, distance=True)
        self.max_recall_num = max_recall_num
        self.rerank_frame_num = rerank_frame_num
        self.instance_num = instance_num

        self.geometric = geometric

        # self.model_type = model_type
        # self.verbose = verbose
        # self.show_num = 50
        # self.save_results = save_results
        # self.saving_dir = saving_dir

    def _load_rgb(self, dataset, scene, f_id):
        points_dir = join(dataset.scans_root, "{}/{}/{}.pkl".format(scene, "rgbd", f_id))
        with open(points_dir, 'rb') as handle:
            rgbd = pickle.load(handle)
        return rgbd["rgb"]

    def save_images(self, saving_frames, name, dataset):
        output_folder = join(self.images_output, name)

        s_id, f_id = saving_frames[0]
        saving_scene = dataset.scenes[s_id]
        image = self._load_rgb(dataset=dataset, scene=saving_scene, f_id=f_id)
        title = "({},{})".format(saving_scene, f_id)
        separation = np.ones((image.shape[0], 10, 3)) * 255

        for idx in range(1, len(saving_frames), 1):
            s_id, f_id = saving_frames[idx]
            scene = dataset.scenes[s_id]
            rgb = self._load_rgb(dataset=dataset, scene=scene, f_id=f_id)
            image = np.concatenate((image, separation, rgb), axis=1).astype(np.uint8)
            title += "-({},{})".format(scene, f_id)

        output_path = os.path.join(output_folder, "{}".format(saving_scene))
        if not exists(output_path):
            os.makedirs(output_path)
        imageio.imwrite(os.path.join(output_path, title + '.jpg'), image)

    def evaluate_rerank_recall(self, model, dataset, device):
        with open(join(self.database_output, "database_features_0.pkl"), "rb") as handle:
            frames, database_frames, global_descriptors = pickle.load(handle)

        global_descriptors = np.stack(global_descriptors)
        total_rerank_num = min(len(database_frames), self.rerank_frame_num)
        stage1_results = np.zeros((self.max_recall_num,))
        stage2_results = np.zeros((total_rerank_num,))
        total_num = 0

        stage1_frame_results = {}
        stage2_frame_results = {}

        total_times = []
        for idx in tqdm(range(len(frames)), desc="calculating recall of each frame"):
            scene_id, frame_id = frames[idx]
            with open(join(self.database_output, "{}.pkl".format(idx)), "rb") as handle:
                current_fine_inputs = pickle.load(handle)
            current_global = current_fine_inputs[DataDict.global_descriptor]
            current_features = torch.from_numpy(current_fine_inputs[DataDict.features]).unsqueeze(0)
            current_geometrics = torch.from_numpy(current_fine_inputs[DataDict.geometrics]).unsqueeze(0)
            current_neighbors = torch.from_numpy(current_fine_inputs[DataDict.neighbors]).unsqueeze(0)
            current_similarities = torch.from_numpy(current_fine_inputs[DataDict.similarities]).unsqueeze(0)
            gt = dataset.gt[(scene_id, frame_id)]
            if (frame_id in gt) or len(gt) == 0:
                continue

            # global retrieval
            distances = self.distance(torch.from_numpy(np.tile(current_global, (len(global_descriptors), 1))),
                                      torch.from_numpy(global_descriptors)).numpy()
            topk_global_indices = np.argsort(distances)
            current_results = np.zeros((self.max_recall_num,))

            other_sorted_frames = [frames[database_frames[topk_global_indices[k]]].tolist() for k in
                                   range(self.max_recall_num)]
            stage1_frame_results[(dataset.scenes[scene_id], frame_id)] = [[dataset.scenes[f[0]], f[1]] for f in
                                                                          other_sorted_frames]
            for k in range(self.max_recall_num):
                scene_other, frame_other = frames[database_frames[topk_global_indices[k]]]
                if (scene_other == scene_id) and (frame_other in gt):
                    current_results[k:] = 1
                if scene_other == scene_id:
                    if frame_other in gt:
                        current_results[k:] = 1
                        break
            stage1_results += current_results
            total_num += 1

            # rerank
            current_global = torch.from_numpy(current_global).unsqueeze(0)
            frames_num = self.instance_num - 1
            other_idx = 0
            all_score = []
            times = []
            while other_idx < total_rerank_num:
                selected_frame_indices = [database_frames[topk_global_indices[ind]] for ind in range(
                    other_idx, other_idx + min(frames_num, total_rerank_num - other_idx), 1)]
                all_global = [current_global]
                all_fts = [current_features]
                all_geos = [current_geometrics]
                all_nbs = [current_neighbors]
                all_sms = [current_similarities]
                for selected_index in selected_frame_indices:
                    with open(join(self.database_output, "{}.pkl".format(selected_index)), "rb") as handle:
                        other_fine_inputs = pickle.load(handle)
                    other_features = torch.from_numpy(other_fine_inputs[DataDict.features]).unsqueeze(0)
                    other_geometrics = torch.from_numpy(other_fine_inputs[DataDict.geometrics]).unsqueeze(0)
                    other_neighbors = torch.from_numpy(other_fine_inputs[DataDict.neighbors]).unsqueeze(0)
                    other_similarities = torch.from_numpy(other_fine_inputs[DataDict.similarities]).unsqueeze(0)
                    other_global = torch.from_numpy(other_fine_inputs[DataDict.global_descriptor]).unsqueeze(0)

                    all_global.append(other_global)
                    all_fts.append(other_features)
                    all_geos.append(other_geometrics)
                    all_nbs.append(other_neighbors)
                    all_sms.append(other_similarities)
                fine_input = [
                    to_device(torch.concat(all_global, dim=0).unsqueeze(0), device=device),
                    to_device(torch.concat(all_fts, dim=0).unsqueeze(0), device=device),
                    to_device(torch.concat(all_geos, dim=0).unsqueeze(0), device=device),
                    to_device(torch.concat(all_nbs, dim=0).unsqueeze(0), device=device),
                    to_device(torch.concat(all_sms, dim=0).unsqueeze(0), device=device)]

                # high.start_counters([events.PAPI_FP_OPS, ])
                start_time = time.time()
                final_score, local_score = model.forward_fine_matching(fine_input=fine_input)
                # flop = high.stop_counters()
                times.append(time.time() - start_time)
                # flops.append(flop)
                all_score.append(local_score.flatten().detach().cpu().numpy())
                torch.cuda.empty_cache()
                other_idx += frames_num

            # current_flop = sum(flops) / total_rerank_num
            current_time = sum(times) / total_rerank_num
            # total_flops.append(current_flop)
            total_times.append(current_time)

            other_distance = 1 - np.concatenate(all_score)
            topk_local_indices = np.argsort(other_distance)

            s2_selected_frames = [frames[database_frames[topk_global_indices[topk_local_indices[k]]]].tolist() for k
                                  in range(total_rerank_num)]
            stage2_frame_results[(dataset.scenes[scene_id], frame_id)] = [[dataset.scenes[f[0]], f[1]] for f in
                                                                          s2_selected_frames]
            fine_results = np.zeros((total_rerank_num,))
            for k in range(total_rerank_num):
                selected_global_idx = topk_local_indices[k]
                scene_other, frame_other = frames[database_frames[topk_global_indices[selected_global_idx]]]
                if (scene_other == scene_id) and (frame_other in gt):
                    fine_results[k:] = 1
            stage2_results += fine_results

            print(stage1_results / float(total_num), stage2_results / float(total_num))
        return stage1_results / float(total_num), stage2_results / float(total_num)

    def multiple_threads(self, model, dataset, batch, idx):
        frames = []
        database_frames = []
        global_descriptors = []

        frames = copy.deepcopy(dataset.all_indices)
        each_batch = int(len(frames) / batch)

        if idx == 0:
            start_id = 0
            end_id = each_batch
        elif idx == batch - 1:
            start_id = (batch - 1) * each_batch
            end_id = len(frames)
        else:
            start_id = each_batch * idx
            end_id = each_batch * (1 + idx)

        for fi in tqdm(range(start_id, end_id, 1), desc="calculating recall frames features"):
            s_id, f_id, global_descriptor, fts, geos, neighbors, similarities = dataset[fi]
            if [s_id, f_id] in dataset.selected_frames:
                database_frames.append(fi)
                global_descriptors.append(global_descriptor)
            fine_inputs = {
                DataDict.global_descriptor: global_descriptor,
                DataDict.features: fts,
                DataDict.geometrics: geos,
                DataDict.neighbors: neighbors,
                DataDict.similarities: similarities,
            }
            with open(join(self.database_output, "{}.pkl".format(fi)), 'wb') as handle:
                pickle.dump(fine_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(join(self.database_output, "database_features_{}.pkl".format(idx)), 'wb') as handle:
                pickle.dump([frames, database_frames, global_descriptors], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate_recall(self, model, dataset, device):
        frames = []
        database_frames = []
        global_descriptors = []
        fine_inputs_index = 0
        only_global = False
        times = []
        if model.model_type == ModelType.fine_matching:
            frames = copy.deepcopy(dataset.all_indices)
            for fi in tqdm(range(len(frames)), desc="calculating recall frames features"):
                s_id, f_id, global_descriptor, fts, geos, neighbors, similarities = dataset[fi]
                if [s_id, f_id] in dataset.selected_frames:
                    database_frames.append(fine_inputs_index)
                    global_descriptors.append(global_descriptor)
                fine_inputs = {
                    DataDict.global_descriptor: global_descriptor,
                    DataDict.features: fts,
                    DataDict.geometrics: geos,
                    DataDict.neighbors: neighbors,
                    DataDict.similarities: similarities,
                }
                with open(join(self.database_output, "{}.pkl".format(fine_inputs_index)), 'wb') as handle:
                    pickle.dump(fine_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                fine_inputs_index += 1
            with open(join(self.database_output, "database_features.pkl"), 'wb') as handle:
                pickle.dump([frames, database_frames, global_descriptors], handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            for iteration, data_dict in enumerate(tqdm(dataset, desc="calculating recall frames features")):
                if iteration >= len(dataset):
                    break
                input_dict = registration_collate_fn_stack_mode([data_dict])
                input_dict = to_device(input_dict, device=device)

                results = model.forward_recall_features(input_data=input_dict)
                times.append(results[DataDict.step_time])
                # print(np.mean(times), results[DataDict.step_time])

                frame_ids = results[DataDict.frame_id].detach().cpu().numpy()
                for fid in frame_ids:
                    frames.append(fid)

                if results[DataDict.fine_input] is not None and (model.model_type != ModelType.coarse_matching):
                    global_descriptor, fts, geos, neighbors, similarities = results[DataDict.fine_input]
                    global_descriptor = global_descriptor.squeeze(0).detach().cpu().numpy()
                    fts = fts.squeeze(0).detach().cpu().numpy()
                    geos = geos.squeeze(0).detach().cpu().numpy()
                    neighbors = neighbors.squeeze(0).detach().cpu().numpy()
                    similarities = similarities.squeeze(0).detach().cpu().numpy()
                    for i in range(global_descriptor.shape[0]):
                        s_id, f_id = frame_ids[i]
                        if [s_id, f_id] in dataset.selected_frames:
                            database_frames.append(fine_inputs_index)
                            global_descriptors.append(global_descriptor[i])
                        fine_inputs = {
                            DataDict.global_descriptor: global_descriptor[i],
                            DataDict.features: fts[i],
                            DataDict.geometrics: geos[i],
                            DataDict.neighbors: neighbors[i],
                            DataDict.similarities: similarities[i],
                        }
                        with open(join(self.database_output, "{}.pkl".format(fine_inputs_index)), 'wb') as handle:
                            pickle.dump(fine_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        fine_inputs_index += 1
                else:
                    only_global = True
                    global_descriptor = results[DataDict.pts_coarse_embeds].detach().cpu().numpy()
                    for i in range(global_descriptor.shape[0]):
                        s_id, f_id = frame_ids[i]
                        if [s_id, f_id] in dataset.selected_frames:
                            database_frames.append(fine_inputs_index)
                            global_descriptors.append(global_descriptor[i])
                        with open(join(self.database_output, "{}.pkl".format(fine_inputs_index)), 'wb') as handle:
                            pickle.dump(global_descriptor[i], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        fine_inputs_index += 1

        global_descriptors = np.stack(global_descriptors)
        total_rerank_num = min(len(database_frames), self.rerank_frame_num)
        stage1_results = np.zeros((self.max_recall_num,))
        stage2_results = np.zeros((total_rerank_num,))
        total_num = 0

        stage1_frame_results = {}
        stage2_frame_results = {}

        total_flops = []
        total_times = []
        first_stage_dataset = {}
        for idx in tqdm(range(len(frames)), desc="calculating recall of each frame"):
            scene_id, frame_id = frames[idx]
            with open(join(self.database_output, "{}.pkl".format(idx)), "rb") as handle:
                current_fine_inputs = pickle.load(handle)
            if only_global:
                current_global = current_fine_inputs
            else:
                current_global = current_fine_inputs[DataDict.global_descriptor]
                current_features = torch.from_numpy(current_fine_inputs[DataDict.features]).unsqueeze(0)
                current_geometrics = torch.from_numpy(current_fine_inputs[DataDict.geometrics]).unsqueeze(0)
                current_neighbors = torch.from_numpy(current_fine_inputs[DataDict.neighbors]).unsqueeze(0)
                current_similarities = torch.from_numpy(current_fine_inputs[DataDict.similarities]).unsqueeze(0)
            gt = dataset.gt[(scene_id, frame_id)]
            if (frame_id in gt) or len(gt) == 0:
                continue

            distances = self.distance(torch.from_numpy(np.tile(current_global, (len(global_descriptors), 1))),
                                      torch.from_numpy(global_descriptors)).numpy()
            topk_global_indices = np.argsort(distances)
            current_results = np.zeros((self.max_recall_num,))
            if self.verbose:
                saving_frames = [[scene_id, frame_id]] + [frames[database_frames[topk_global_indices[k]]].tolist() for k in range(3)]
                self.save_images(saving_frames=saving_frames, dataset=dataset, name="stage1")

            if self.save_stage1:
                other_sorted_frames = [frames[database_frames[topk_global_indices[k]]].tolist() for k in range(min(100, len(topk_global_indices)))]
                stage1_frame_results[(dataset.scenes[scene_id], frame_id)] = {
                    "gt": gt, "frames": [[dataset.scenes[f[0]], f[1]] for f in other_sorted_frames]
                }

            for k in range(self.max_recall_num):
                scene_other, frame_other = frames[database_frames[topk_global_indices[k]]]
                # if (scene_other == scene_id) and (frame_other in gt):
                #     current_results[k:] = 1
                #     break
                if scene_other == scene_id:
                    if frame_other in gt:
                        current_results[k:] = 1
                        break
            stage1_results += current_results
            total_num += 1

            if not only_global:
                current_global = torch.from_numpy(current_global).unsqueeze(0)
                frames_num = self.instance_num - 1
                other_idx = 0
                all_score = []
                flops = []
                times = []
                while other_idx < total_rerank_num:
                    selected_frame_indices = [database_frames[topk_global_indices[ind]] for ind in range(
                        other_idx, other_idx + min(frames_num, total_rerank_num - other_idx), 1)]
                    all_global = [current_global]
                    all_fts = [current_features]
                    all_geos = [current_geometrics]
                    all_nbs = [current_neighbors]
                    all_sms = [current_similarities]
                    for selected_index in selected_frame_indices:
                        with open(join(self.database_output, "{}.pkl".format(selected_index)), "rb") as handle:
                            other_fine_inputs = pickle.load(handle)
                        other_features = torch.from_numpy(other_fine_inputs[DataDict.features]).unsqueeze(0)
                        other_geometrics = torch.from_numpy(other_fine_inputs[DataDict.geometrics]).unsqueeze(0)
                        other_neighbors = torch.from_numpy(other_fine_inputs[DataDict.neighbors]).unsqueeze(0)
                        other_similarities = torch.from_numpy(other_fine_inputs[DataDict.similarities]).unsqueeze(0)
                        other_global = torch.from_numpy(other_fine_inputs[DataDict.global_descriptor]).unsqueeze(0)

                        all_global.append(other_global)
                        all_fts.append(other_features)
                        all_geos.append(other_geometrics)
                        all_nbs.append(other_neighbors)
                        all_sms.append(other_similarities)
                    fine_input = [
                        to_device(torch.concat(all_global, dim=0).unsqueeze(0), device=device),
                        to_device(torch.concat(all_fts, dim=0).unsqueeze(0), device=device),
                        to_device(torch.concat(all_geos, dim=0).unsqueeze(0), device=device),
                        to_device(torch.concat(all_nbs, dim=0).unsqueeze(0), device=device),
                        to_device(torch.concat(all_sms, dim=0).unsqueeze(0), device=device)]


                    final_score, local_score = model.forward_fine_matching(fine_input=fine_input)

                    all_score.append(local_score.flatten().detach().cpu().numpy())
                    torch.cuda.empty_cache()
                    other_idx += frames_num

                # current_flop = sum(flops) / total_rerank_num
                current_time = sum(times) / total_rerank_num
                # total_flops.append(current_flop)
                total_times.append(current_time)

                other_distance = 1 - np.concatenate(all_score)
                topk_local_indices = np.argsort(other_distance)

                if self.verbose:
                    saving_frames = [[scene_id, frame_id]] + [frames[database_frames[topk_global_indices[topk_local_indices[k]]]].tolist() for k in range(3)]
                    self.save_images(saving_frames=saving_frames, dataset=dataset, name="stage2")
                    s2_selected_frames = [frames[database_frames[topk_global_indices[topk_local_indices[k]]]].tolist() for k
                                          in range(total_rerank_num)]
                    stage2_frame_results[(dataset.scenes[scene_id], frame_id)] = [[dataset.scenes[f[0]], f[1]] for f in s2_selected_frames]
                fine_results = np.zeros((total_rerank_num,))
                for k in range(total_rerank_num):
                    selected_global_idx = topk_local_indices[k]
                    scene_other, frame_other = frames[database_frames[topk_global_indices[selected_global_idx]]]
                    if (scene_other == scene_id) and (frame_other in gt):
                        fine_results[k:] = 1
                stage2_results += fine_results

        if self.save_stage1:
            with open(join(self.database_output, "stage_1.pkl"), 'wb') as handle:
                pickle.dump(stage1_frame_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return stage1_results / float(total_num), stage2_results / float(total_num)