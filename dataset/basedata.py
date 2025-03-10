import copy
import os
import pickle
from abc import ABC, abstractmethod

import cv2
import numpy as np
import open3d as o3d
from collections import Counter
from os.path import join, exists
import networkx
from tqdm import tqdm


def depth_rgb_points(depth, rgb, intrinsic):
    fx, fy, mx, my = intrinsic[1, 1], intrinsic[0, 0], intrinsic[1, 2], intrinsic[0, 2]
    indices = np.indices(depth.shape)
    rows = indices[0].flatten()
    columns = indices[1].flatten()
    values = depth[rows, columns]
    z = values
    x = (rows - mx) * z / fx
    y = (columns - my) * z / fy
    available_indices = np.where(z != 0.0)
    # xyz, yxz, xzy, yzx, zxy, zyx
    rgbd_pts = np.stack((y[available_indices], x[available_indices], z[available_indices]), axis=1)
    # rgbd_pts=self.transform_points(rgbd_pts, align)

    colors = rgb[rows, columns] / 255.0
    rgbd_colors = colors[available_indices]
    return np.concatenate((rgbd_pts, rgbd_colors), axis=1)


class BaseData(ABC):
    def __init__(self, root, output_root, display=True, distance_threshold=6,
                 normal_radius=0.4, max_normal_points_num=100, point_threshold=4000, min_rgbd_points_num=3000,
                 downsample_voxel_size=0.02, overlap_downsample_voxel_size=0.05, minimum_voxel_number=1000,
                 frame_overlap_threshold=0.5, positive_overlap_threshold=0.2, negative_overlap_threshold=0.01, ):
        self.point_threshold = point_threshold
        self.min_rgbd_points_num = min_rgbd_points_num

        self.distance_threshold = distance_threshold
        self.save_visualize = display
        self.normal_radius = normal_radius
        self.max_normal_points_num = max_normal_points_num
        self.overlap_downsample_voxel_size = overlap_downsample_voxel_size
        self.downsample_voxel_size = downsample_voxel_size
        self.frame_overlap_threshold = frame_overlap_threshold
        self.positive_overlap_threshold = positive_overlap_threshold
        self.negative_overlap_threshold = negative_overlap_threshold
        self.minimum_voxel_number = minimum_voxel_number
        self.root = root
        self.output_root = output_root
        self.output_scans_root = join(output_root, "scans")
        if not exists(self.output_scans_root):
            os.makedirs(self.output_scans_root)

    @abstractmethod
    def select_similar_scenes(self, current_scene, all_scenes):
        pass

    def get_points_normals(self, pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.normal_radius, max_nn=self.max_normal_points_num))
        if pts.shape[1] > 8:
            all_pts = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals), pts[:, 6:]), axis=1)
        else:
            all_pts = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals), pts[:, 3:]), axis=1)
        return all_pts

    def _display_rgbd(self, rgb, depth, folder, name):
        rgb_folder = join(folder, "rgb")
        depth_folder = join(folder, "depth")
        if not exists(depth_folder):
            os.makedirs(depth_folder)
        if not exists(rgb_folder):
            os.makedirs(rgb_folder)
        rgb[:, :, [0, 1, 2]] = rgb[:, :, [2, 1, 0]]
        cv2.imwrite(join(rgb_folder, str(name) + ".jpg"), rgb)
        depth_normalized = np.array(depth.astype(float) / np.max(depth.astype(float)) * 255, dtype=int)
        cv2.imwrite(join(depth_folder, str(name) + '.png'), depth_normalized)

    def _save_rgbd(self, data, scene, frame):
        folder = join(self.output_scans_root, scene, "rgbd")
        if not exists(folder):
            os.makedirs(folder)
        with open(join(folder, "{}.pkl".format(frame)), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_frustum_indices(self, points, pose, K, width, height, distance_threshold):
        points_world_homogeneous = np.c_[points, np.ones(points.shape[0])]
        points_cam_homogeneous = np.matmul(points_world_homogeneous, np.linalg.inv(pose).T)
        points_2d = np.matmul(points_cam_homogeneous[:, :3], K.T)
        points_2d[:, :2] = points_2d[:, :2] / points_2d[:, 2, np.newaxis]
        c1 = (points_2d[:, 0] < width) & (points_2d[:, 0] >= 0)
        c2 = (points_2d[:, 1] < height) & (points_2d[:, 1] >= 0)
        c3 = points_2d[:, 2] > 0
        c4 = points_2d[:, 2] < distance_threshold
        c = np.where(c1 & c2 & c3 & c4)
        return c, points_cam_homogeneous[c][:, :3], points_2d

    def select_pos_neg(self, all_voxels):
        all_positives = []
        all_negatives = []
        for current_index in range(len(all_voxels)):
            current_voxels = all_voxels[current_index]
            positives = []
            negatives = []
            for index in range(len(all_voxels)):
                if index != current_index:
                    voxels = all_voxels[index]
                    overlap_set = list(current_voxels & voxels)
                    coverage = len(overlap_set) / len(current_voxels)
                    if coverage > self.positive_overlap_threshold:
                        positives.append(index)
                    elif coverage <= self.negative_overlap_threshold:
                        negatives.append(index)
            all_positives.append(copy.deepcopy(positives))
            all_negatives.append(copy.deepcopy(negatives))
        return all_positives, all_negatives

    def save_all_points(self, scene, all_points):
        # self.display_labels(all_points)
        process_points = o3d.geometry.PointCloud()
        process_points.points = o3d.utility.Vector3dVector(all_points[:, :3])
        process_points.colors = o3d.utility.Vector3dVector(all_points[:, 3:6])
        min_bound = process_points.get_min_bound()
        max_bound = process_points.get_max_bound()
        pcd, notsure, indices_relations = process_points.voxel_down_sample_and_trace(
            voxel_size=self.downsample_voxel_size,
            min_bound=min_bound,
            max_bound=max_bound)
        if all_points.shape[1] == 7:
            original_labels = all_points[:, 6]
            labels = []
            for idx in range(len(indices_relations)):
                current_indcies = np.array(indices_relations[idx])
                current_labels = original_labels[(current_indcies,)]
                if len(current_indcies) == 1:
                    labels.append(int(current_labels[0]))
                else:
                    c = Counter(current_labels.tolist())
                    value = int(c.most_common(1)[0][0])
                    labels.append(value)
            vertices = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.colors),
                                       np.array(labels)[:, np.newaxis]), axis=1)
        else:
            vertices = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.colors)), axis=1)
        # self.display_labels(vertices)
        # self.display_colors(vertices[:, :6])
        # print("test")
        folder_name = join(self.output_scans_root, scene)
        if not exists(folder_name):
            os.makedirs(folder_name)
        with open(join(folder_name, "all_points.pkl"), 'wb') as handle:
            pickle.dump(vertices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return vertices

    def downsample_points(self, all_points):
        process_points = o3d.geometry.PointCloud()
        process_points.points = o3d.utility.Vector3dVector(all_points[:, :3])
        process_points.colors = o3d.utility.Vector3dVector(all_points[:, 3:6])
        pcd = process_points.voxel_down_sample(voxel_size=self.overlap_downsample_voxel_size)
        rgbd_pts = np.asarray(pcd.points)
        rgbd_clrs = np.asarray(pcd.colors)
        all_points = np.concatenate((rgbd_pts, rgbd_clrs), axis=1)
        return all_points

    def _points_to_2d(self, scene, frame, points_2d, color, height, width):
        image = np.zeros((height, width, 4))
        for i in range(len(points_2d)):
            c, r = min(round(points_2d[i][0]), width - 1), min(round(points_2d[i][1]), height - 1)
            if image[r, c, 3] == 0:
                image[r, c, :3] = color[i]
                image[r, c, 3] = points_2d[i][2]
            elif image[r, c, 3] > points_2d[i][2]:
                image[r, c, :3] = color[i]
                image[r, c, 3] = points_2d[i][2]

        image = image[:, :, :3] * 255
        image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
        folder = join(self.output_scans_root, scene, "visualize", "pts_2d")
        if not exists(folder):
            os.makedirs(folder)
        cv2.imwrite(join(folder, str(frame) + ".jpg"), image)

    def combine_info(self, scenes, name):
        all_info = {}
        for index in tqdm(range(len(scenes)), desc="combine information " + name):
            scene = scenes[index]
            with open(join(self.output_scans_root, str(scene), "frame_info.pkl"), "rb") as handle:
                # print(scene)
                frame_info = pickle.load(handle)
            if len(frame_info["poses"]) > 1:
                similar_scenes = self.select_similar_scenes(current_scene=scene, all_scenes=scenes)
                if similar_scenes is not None:
                    all_info[str(scene)] = {
                        "positives": frame_info["positives"],
                        "negatives": frame_info["negatives"],
                        "similar_scenes": similar_scenes,
                        "poses": frame_info["poses"]
                    }

        with open(join(self.output_root, name + ".pkl"), 'wb') as handle:
            pickle.dump(all_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_key_frames(self, name="testing"):
        with open(join(self.output_root, name + ".pkl"), "rb") as handle:
            all_info = pickle.load(handle)

        for scene in all_info.keys():
            poses = all_info[scene]["poses"]
            negatives = all_info[scene]["negatives"]
            positives = all_info[scene]["positives"]

            Gn = networkx.Graph()
            Gp = networkx.Graph()
            for index in range(len(poses)):
                current_negatives = negatives[index]
                current_positives = positives[index]
                for other_idx in range(len(poses)):
                    if other_idx != index and other_idx not in current_negatives:
                        Gn.add_edge(index, other_idx)
                    if other_idx in current_positives:
                        Gp.add_edge(index, other_idx)
            negative_key_frames = list(networkx.dominating_set(Gn))
            positive_key_frames = list(networkx.dominating_set(Gp))
            all_info[scene].update({"negative_key_frames": negative_key_frames,
                                    "positive_key_frames": positive_key_frames})

        with open(join(self.output_root, name + ".pkl"), 'wb') as handle:
            pickle.dump(all_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _down_sample(self, pts, vx_size):
        process_points = o3d.geometry.PointCloud()
        process_points.points = o3d.utility.Vector3dVector(pts[:, :3])
        process_points.normals = o3d.utility.Vector3dVector(pts[:, 3:6])
        process_points.colors = o3d.utility.Vector3dVector(pts[:, 6:9])
        min_bound = process_points.get_min_bound()
        max_bound = process_points.get_max_bound()
        pcd, notsure, indices_relations = process_points.voxel_down_sample_and_trace(voxel_size=vx_size,
                                                                                     min_bound=min_bound,
                                                                                     max_bound=max_bound)
        if pts.shape[1] == 10:
            original_labels = pts[:, 9]
            labels = []
            for idx in range(len(indices_relations)):
                current_indcies = np.array(indices_relations[idx])
                current_labels = original_labels[(current_indcies,)]
                if len(current_indcies) == 1:
                    labels.append(int(current_labels[0]))
                else:
                    c = Counter(current_labels.tolist())
                    value = int(c.most_common(1)[0][0])
                    labels.append(value)
            vertices = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors),
                                       np.array(labels)[:, np.newaxis]), axis=1)
        else:
            vertices = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors)), axis=1)
        return vertices

    def downsample_frame(self, pts_folder, frame, output_folder):
        step = 0.005
        vx_size = 0.03 - step

        with open(join(pts_folder, frame), "rb") as handle:
            points = pickle.load(handle)

        if points.shape[1] < 8:
            points = self.get_points_normals(pts=points)

        processed = points
        while len(processed) > self.point_threshold:
            vx_size += step
            processed = self._down_sample(points, vx_size=vx_size)
        while (len(processed) < self.min_rgbd_points_num) and (vx_size > step):
            vx_size -= step
            processed = self._down_sample(points, vx_size=vx_size)
        if len(processed) < self.min_rgbd_points_num:
            print("the processed rgbd points {} are less than {}".format(len(processed), self.min_rgbd_points_num))

        if not exists(output_folder):
            os.makedirs(output_folder)
        with open(join(output_folder, frame), 'wb') as handle:
            pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_down_sample(self, batch, output_dir, total_batch=10, name="fused"):
        all_scenes = sorted(os.listdir(self.output_scans_root))

        each_batch = int(len(all_scenes) / total_batch)
        if batch == 0:
            scenes = all_scenes[:each_batch]
        elif batch == total_batch - 1:
            scenes = all_scenes[(total_batch - 1) * each_batch:]
        else:
            scenes = all_scenes[each_batch * batch:each_batch * (1 + batch)]

        for index in tqdm(range(len(scenes)), desc="process scenes"):
            scene = scenes[index]
            pts_folder = join(self.output_scans_root, scene, name)
            if os.path.exists(pts_folder):
                output_folder = join(output_dir, "scans", scene, name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                frames = os.listdir(pts_folder)
                for frame in frames:
                    self.downsample_frame(pts_folder, frame, output_folder)
