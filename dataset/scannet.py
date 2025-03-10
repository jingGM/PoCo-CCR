import copy
import json
import os
import pickle
import struct
import zlib
from os.path import join, exists

import cv2
import imageio
import pandas as pd
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

from dataset.basedata import BaseData, depth_rgb_points


Files = ["vh_clean_2", "vh_clean", "vh_clean_2.labels"]

# ScanNet-V2 configs:
COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class SceneInfo:
    sensor_name = "sensor_name"
    intrinsic_color = "intrinsic_color"
    extrinsic_color = "extrinsic_color"
    intrinsic_depth = "intrinsic_depth"
    extrinsic_depth = "extrinsic_depth"
    color_compression_type = "color_compression_type"
    depth_compression_type = "depth_compression_type"
    color_width = "color_width"
    color_height = "color_height"
    depth_width = "depth_width"
    depth_height = "depth_height"
    depth_shift = "depth_shift"
    depth_size = "depth_size"
    num_frames = "num_frames"


class RGBDFrame:
    def __init__(self, file_handle, scene_info):
        self.scene_info = scene_info
        self.camera_to_world = np.asarray(struct.unpack('f' * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(
            4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c' * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)

    def get_rgbd(self):
        rgb = self.decompress_color(self.scene_info[SceneInfo.color_compression_type])
        depth_data = self.decompress_depth(self.scene_info[SceneInfo.depth_compression_type])
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape((self.scene_info[SceneInfo.depth_height],
                                                                    self.scene_info[SceneInfo.depth_width]))
        depth = depth.astype(float) / 1000.0
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        return rgb, depth

    def get_pts(self, rgb, depth):
        return depth_rgb_points(rgb=rgb, depth=depth, intrinsic=self.scene_info[SceneInfo.intrinsic_depth])


class ScanNetProcessor(BaseData):
    def __init__(self, root, output_root, display=True, normal_radius=0.4, max_normal_points_num=100,
                 point_threshold=4000, min_rgbd_points_num=3000,
                 downsample_voxel_size=0.02, overlap_downsample_voxel_size=0.05,
                 minimum_voxel_number=1000, frame_overlap_threshold=0.5, distance_threshold=6,
                 positive_overlap_threshold=0.2, negative_overlap_threshold=0.01,
                 ply_file="vh_clean", agg_jason_extension="aggregation.json",
                 label_jason_extension="vh_clean.segs.json", label_config_file="scannetv2-labels.combined.tsv"):
        super(ScanNetProcessor, self).__init__(root=root, output_root=output_root, display=display,
                                               normal_radius=normal_radius, max_normal_points_num=max_normal_points_num,
                                               point_threshold=point_threshold, min_rgbd_points_num=min_rgbd_points_num,
                                               downsample_voxel_size=downsample_voxel_size,
                                               distance_threshold=distance_threshold,
                                               overlap_downsample_voxel_size=overlap_downsample_voxel_size,
                                               minimum_voxel_number=minimum_voxel_number,
                                               frame_overlap_threshold=frame_overlap_threshold,
                                               positive_overlap_threshold=positive_overlap_threshold,
                                               negative_overlap_threshold=negative_overlap_threshold)
        self.scans_root = join(root, "scans")

        self.ply_file = ply_file
        self.agg_jason_extension = agg_jason_extension
        self.label_jason_extension = label_jason_extension
        self.label_config_file = label_config_file

    def get_points(self, scene):
        filename = os.path.join(self.scans_root, scene, scene + "_" + self.ply_file + ".ply")
        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
            vertices[:, 0] = plydata['vertex'].data['x']
            vertices[:, 1] = plydata['vertex'].data['y']
            vertices[:, 2] = plydata['vertex'].data['z']
            vertices[:, 3] = plydata['vertex'].data['red'] / 255.0
            vertices[:, 4] = plydata['vertex'].data['green'] / 255.0
            vertices[:, 5] = plydata['vertex'].data['blue'] / 255.0
        return vertices

    def get_aggregation_labels(self, agg_filename):
        with open(agg_filename) as user_file:
            parsed_json = json.load(user_file)
        ids_dict = parsed_json['segGroups']
        labels = []
        segments_indices = []
        for label_dict in ids_dict:
            for seg in label_dict["segments"]:
                segments_indices.append(seg)
                label = label_dict["label"]
                labels.append(label)
        return segments_indices, labels

    def get_scene_labels(self, scene):
        label_filename = os.path.join(self.scans_root, scene, scene + "_" + self.label_jason_extension)
        with open(label_filename) as user_file:
            parsed_json = json.load(user_file)
        indices = parsed_json["segIndices"]
        return indices

    def get_label_relation(self):
        file_name = join(self.root, self.label_config_file)
        table = pd.read_csv(file_name, sep='\t')
        required_table = table[["raw_category", "nyu40id", "nyu40class"]]
        return required_table.to_numpy()

    def process_scene_points(self, scene, raw_labels, nyu40id):
        points = self.get_points(scene=scene)
        agg_filename = os.path.join(self.scans_root, scene,
                                    scene + "_" + self.ply_file + "." + self.agg_jason_extension)
        if exists(agg_filename):
            segments_indices, labels = self.get_aggregation_labels(agg_filename=agg_filename)
            points_indices = self.get_scene_labels(scene=scene)
            points_nyu_labels = []
            for index in points_indices:
                if index in segments_indices:
                    raw_label = labels[segments_indices.index(index)]
                    points_nyu_labels.append(nyu40id[raw_labels.index(raw_label)])
                else:
                    points_nyu_labels.append(0)
            all_points = np.concatenate((points, np.expand_dims(np.array(points_nyu_labels), axis=1)), axis=1)
            # self.display_labels(all_points)
            # self.display_colors(all_points)

            # folder_name = join(self.output_scans_root, scene)
            # if not exists(folder_name):
            #     os.makedirs(folder_name)
            # with open(join(folder_name, "all_points.pkl"), 'wb') as handle:
            #     pickle.dump(all_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return all_points
        else:
            return points

    def get_scene_info(self, scene):
        scene_file = join(self.scans_root, "{}/{}.sens".format(scene, scene))
        with open(scene_file, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert version == 4, "version check of ScanNet-V2: the version is {} instead of 4".format(version)
            strlen = struct.unpack('Q', f.read(8))[0]
            sensor_name = struct.unpack('c' * strlen, f.read(strlen))  # ''.join(),
            intrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            extrinsic_color = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            intrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            extrinsic_depth = np.asarray(struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            color_width = struct.unpack('I', f.read(4))[0]
            color_height = struct.unpack('I', f.read(4))[0]
            depth_width = struct.unpack('I', f.read(4))[0]
            depth_height = struct.unpack('I', f.read(4))[0]
            depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]

            scene_info = {
                SceneInfo.sensor_name: sensor_name,
                SceneInfo.intrinsic_color: intrinsic_color,
                SceneInfo.extrinsic_color: extrinsic_color,
                SceneInfo.intrinsic_depth: intrinsic_depth,
                SceneInfo.extrinsic_depth: extrinsic_depth,
                SceneInfo.color_compression_type: color_compression_type,
                SceneInfo.depth_compression_type: depth_compression_type,
                SceneInfo.color_width: color_width,
                SceneInfo.color_height: color_height,
                SceneInfo.depth_width: depth_width,
                SceneInfo.depth_height: depth_height,
                SceneInfo.depth_shift: depth_shift,
                SceneInfo.num_frames: num_frames,
            }
            frames = []
            poses = []
            distance_ranges = []
            for i in tqdm(range(num_frames), desc="process scene {}".format(scene)):
                frame = RGBDFrame(f, scene_info=scene_info)
                frames.append(frame)
                poses.append(frame.camera_to_world)
                distance_ranges.append(min(self.distance_threshold, np.max(frame.get_rgbd()[1])))
        return frames, scene_info, poses, distance_ranges

    def process_frames(self, scene):
        frames, scene_info, poses, distance_ranges = self.get_scene_info(scene=scene)

        # with open(join(self.output_scans_root, scene, "all_poses.pkl"), 'wb') as handle:
        #     pickle.dump(poses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(join(self.output_scans_root, scene, "scene_info.pkl"), 'wb') as handle:
            pickle.dump(scene_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(join(self.output_scans_root, scene, "distance_ranges.pkl"), 'wb') as handle:
        #     pickle.dump(distance_ranges, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return poses, scene_info, frames, distance_ranges

    def process_rgbd(self, scene, frames, frame_indices):
        for select_index in tqdm(range(len(frame_indices)), desc="process rgbd {}".format(scene)):
            frame_idx = frame_indices[select_index]
            frame = frames[frame_idx]
            rgb, depth = frame.get_rgbd()
            self._save_rgbd(data={"rgb": rgb, "depth": depth}, scene=scene, frame=select_index)
            if self.save_visualize:
                self._display_rgbd(rgb=rgb, depth=depth, folder=join(self.output_scans_root, scene, "visualize"),
                                   name=select_index)

    def load_points(self, scene):
        folder_name = join(self.output_scans_root, scene)
        with open(join(folder_name, "all_points.pkl"), "rb") as handle:
            all_points = pickle.load(handle)
        # self.display_labels(all_points[np.where(all_points[:, 6] == 40.0)])
        with open(join(self.output_scans_root, scene, "all_poses.pkl"), "rb") as handle:
            poses = np.array(pickle.load(handle))
        with open(join(self.output_scans_root, scene, "scene_info.pkl"), "rb") as handle:
            scene_info: SceneInfo = pickle.load(handle)
        with open(join(self.output_scans_root, scene, "distance_ranges.pkl"), "rb") as handle:
            distance_ranges = pickle.load(handle)
        return all_points, poses, scene_info, distance_ranges

    def process_frustum(self, scene, all_points, poses, K, width, height, frame_ranges):
        # for i in range(len(poses)):
        for i in tqdm(range(len(poses)), desc="process frustum {}".format(scene)):
            pose = poses[i]
            c, selected_pts, points_2d = self.get_frustum_indices(points=all_points[:, :3], pose=pose,
                                                                  K=K, width=width, height=height,
                                                                  distance_threshold=frame_ranges[i])
            selected_pts = np.concatenate((selected_pts, all_points[c][:, 3:]), axis=1)

            processed = self.get_points_normals(pts=selected_pts)
            folder = join(self.output_scans_root, scene, "fused")
            if not exists(folder):
                os.makedirs(folder)
            with open(join(folder, "{}.pkl".format(i)), 'wb') as handle:
                pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if self.save_visualize:
                self._points_to_2d(scene=scene, frame=i, points_2d=points_2d[c], color=selected_pts[:, 3:6],
                                   height=height, width=width)

    def select_poses(self, scene, intrinsic, width, height, all_poses, all_points, all_frame_ranges):
        last_voxels = None
        frame_indices = []
        frame_poses = []
        frame_voxels = []
        frame_ranges = []
        K = intrinsic[:3, :3]
        for index in tqdm(range(len(all_poses)), desc="process poses {}".format(scene)):
            pose = all_poses[index]
            c, selected_pts, _ = self.get_frustum_indices(points=all_points[:, :3], pose=pose, K=K, width=width,
                                                          height=height,
                                                          distance_threshold=all_frame_ranges[index])
            c = c[0]
            # selected_pts = np.concatenate((selected_pts, all_points[c][:, 3:6]), axis=1)
            # self.display_colors(selected_pts)
            if len(c) < self.minimum_voxel_number:
                continue
            if last_voxels is None:
                last_voxels = set(c.tolist())
                frame_indices.append(index)
                frame_poses.append(pose)
                frame_voxels.append(last_voxels)
                frame_ranges.append(all_frame_ranges[index])
            else:
                current_set = set(c.tolist())
                overlap_set = list(current_set & last_voxels)
                coverage = len(overlap_set) / max(len(current_set), len(last_voxels))
                if coverage < self.frame_overlap_threshold:
                    # self.display_overlap(ref=last_voxels, src=current_set, points=all_points)
                    last_voxels = set(c.tolist())
                    frame_indices.append(index)
                    frame_poses.append(pose)
                    frame_voxels.append(last_voxels)
                    frame_ranges.append(all_frame_ranges[index])

        pos, neg = self.select_pos_neg(all_voxels=frame_voxels)
        with open(join(self.output_scans_root, scene, "frame_info.pkl"), 'wb') as handle:
            pickle.dump({"positives": pos, "negatives": neg, "poses": frame_poses, "frame_relation": frame_indices, "intrinsic": intrinsic},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        return frame_poses, pos, neg, frame_indices, frame_ranges

    def select_similar_scenes(self, current_scene, all_scenes):
        scenes = []
        for scene in all_scenes:
            if (scene[:9] == current_scene[:9]) and (scene != current_scene):
                scenes.append(scene)
        return scenes

    def run_scenes(self, scenes=None):
        print(scenes)
        table = self.get_label_relation()
        raw_labels = table[:, 0].tolist()
        nyu40id = table[:, 1].tolist()
        nyu40class = table[:, 2].tolist()

        if scenes is None:
            scenes = os.listdir(self.scans_root)
            scenes.sort()
        for scene in scenes:
            all_points = self.process_scene_points(scene=scene, raw_labels=raw_labels, nyu40id=nyu40id)
            all_points = self.save_all_points(scene=scene, all_points=all_points)

            all_poses, scene_info, frames, all_frame_ranges = self.process_frames(scene=scene)
            # all_points, all_poses, scene_info, all_frame_ranges = self.load_points(scene=scene)

            width = scene_info[SceneInfo.depth_width]
            height = scene_info[SceneInfo.depth_height]

            downsampled_pts = self.downsample_points(all_points=all_points)
            frame_poses, pos, neg, frame_indices, frame_ranges = self.select_poses(scene=scene, intrinsic=scene_info[SceneInfo.intrinsic_depth],
                                                                                   width=width, height=height,
                                                                                   all_poses=all_poses,
                                                                                   all_points=downsampled_pts,
                                                                                   all_frame_ranges=all_frame_ranges)
            K = scene_info[SceneInfo.intrinsic_depth][:3, :3]
            self.process_frustum(scene=scene, all_points=all_points, poses=frame_poses,
                                 K=K, width=width, height=height, frame_ranges=frame_ranges)
            self.process_rgbd(scene, frames, frame_indices)
            # print("test")


if __name__ == '__main__':
    from dataset.scannet_scenes import all_scenes, training_scenes, evaluation_scenes, testing_scenes, sanity_scenes
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--scene', type=int, default=0, help="")
    parser.add_argument('--batch', type=int, default=1, help="")
    parser.add_argument('--stage', type=int, default=1,
                        help='1 process the single frames; 2 combine information of all frames for training; 3 select key frames for testing environment')
    args = parser.parse_args()

    scannet = ScanNetProcessor(display=True,
                               root="/media/jing/Data/data/scannet",
                               output_root="/home/jing/Documents/work/data/ScanNetPTS")
    # scannet = ScanNetProcessor(display=False,
    #                            root="/media/jing/data/data",
    #                            output_root="/media/jing/a21deed5-8521-47d3-9a63-dc4a282e4c00/ScanNetPTS")

    if args.stage == 1:
        if args.batch == 1:
            scannet.run_scenes(scenes=all_scenes[args.scene])
            # scannet.run_scenes(scenes=sanity_scenes)
        else:
            each_batch = int(len(all_scenes) / args.batch)
            if args.scene == 0:
                scannet.run_scenes(scenes=all_scenes[:each_batch])
            elif args.scene == args.batch - 1:
                scannet.run_scenes(scenes=all_scenes[(args.batch - 1) * each_batch:])
            else:
                scannet.run_scenes(scenes=all_scenes[each_batch * args.scene:each_batch * (1 + args.scene)])
    elif args.stage == 2:
        for scenes, name in [[training_scenes, "training"],
                             [evaluation_scenes, "evaluation"],
                             [testing_scenes, "testing"]]:
            scannet.combine_info(scenes=scenes, name=name)
    else:
        scannet.process_key_frames(name="testing")


