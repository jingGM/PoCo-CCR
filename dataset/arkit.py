import copy
import glob
import json
import os
import pickle
import random
import struct
import zlib
from os.path import join, exists

import cv2
import imageio
import pandas as pd
# import jason
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

from dataset.arkit_utils import get_bounding_boxes
from dataset.basedata import BaseData


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix


def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)

    return (ts, Rt)


def generate_point(
        rgb_image,
        depth_image,
        intrinsic,
        subsample=1,
        world_coordinate=True,
        pose=None,
):
    """Generate 3D point coordinates and related rgb feature
    Args:
        rgb_image: (h, w, 3) rgb
        depth_image: (h, w) depth
        intrinsic: (3, 3)
        subsample: int resize stride
        world_coordinate: bool
        pose: (4, 4) matrix
            transfer from camera to world coordindate
    Returns:
        points: (N, 3) point cloud coordinates
            in world-coordinates if world_coordinate==True
            else in camera coordinates
        rgb_feat: (N, 3) rgb feature of each point
    """
    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            u[d_filter] * d[d_filter],
            v[d_filter] * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    if world_coordinate:
        new_points_3d_padding = np.vstack(
            (new_points_3d, np.ones((1, new_points_3d.shape[1])))
        )
        world_coord_padding = np.dot(pose, new_points_3d_padding)
        new_points_3d = world_coord_padding[:3]

    rgb_feat = rgb_image[v, u][d_filter]

    return new_points_3d.T, rgb_feat


class DoDProcessor(BaseData):
    def __init__(self, root, output_root, display=True, normal_radius=0.4, max_normal_points_num=100, data_pkl=None,
                 point_threshold=4000, min_rgbd_points_num=3000, annotation_points_limit=50,
                 downsample_voxel_size=0.02, overlap_downsample_voxel_size=0.05,
                 minimum_voxel_number=1000, frame_overlap_threshold=0.5, distance_threshold=6, batch: int = 1,
                 positive_overlap_threshold=0.1, negative_overlap_threshold=0.01, scene_ind=0):
        super(DoDProcessor, self).__init__(root=root, output_root=output_root, display=display,
                                           normal_radius=normal_radius, max_normal_points_num=max_normal_points_num,
                                           point_threshold=point_threshold, min_rgbd_points_num=min_rgbd_points_num,
                                           downsample_voxel_size=downsample_voxel_size,
                                           distance_threshold=distance_threshold,
                                           overlap_downsample_voxel_size=overlap_downsample_voxel_size,
                                           minimum_voxel_number=minimum_voxel_number,
                                           frame_overlap_threshold=frame_overlap_threshold,
                                           positive_overlap_threshold=positive_overlap_threshold,
                                           negative_overlap_threshold=negative_overlap_threshold)
        self.annotation_points_limit = annotation_points_limit
        self.all_scenarios = None
        self.training_type, self.scenes = self.load_files(scene_ind, batch, data_pkl=data_pkl)
        print("scene number: ", len(self.scenes))

    def load_files(self, scene_ind, batch: int, data_pkl=None):
        if data_pkl is None:
            file_name = join(self.root, "metadata.csv")
            table = pd.read_csv(file_name, sep=',')
            self.all_scenarios = table[["video_id", "fold", "visit_id"]].to_numpy()
        else:
            with open(join(self.root, data_pkl), "rb") as handle:
                self.all_scenarios = pickle.load(handle)
            self.all_scenarios = np.stack((self.all_scenarios[2], self.all_scenarios[1],
                                           np.asarray(self.all_scenarios[0], dtype=float)), axis=-1)

        # with open(join(self.output_root, "data.pkl"), 'wb') as handle:
        #     pickle.dump(self.all_scenarios, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if batch == 1:
            return self.all_scenarios[:, 1], self.all_scenarios[:, 0]
        else:
            total_scenes = len(self.all_scenarios)
            each_batch = int(total_scenes / batch)
            if scene_ind == 0:
                return self.all_scenarios[:each_batch, 1], self.all_scenarios[:each_batch, 0]
            elif scene_ind == batch-1:
                return self.all_scenarios[(batch-1)*each_batch:, 1], self.all_scenarios[(batch-1)*each_batch:, 0]
            else:
                return self.all_scenarios[each_batch*scene_ind:each_batch*(1+scene_ind), 1], self.all_scenarios[each_batch*scene_ind:each_batch*(1+scene_ind), 0]

    def process_scene_points(self, scene, file_type):
        filename = os.path.join(self.root, file_type, scene, scene + "_3dod_mesh.ply")
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
        # self.display_colors(vertices)
        # vertices[:, [3, 5]] = vertices[:, [5, 3]]
        return vertices

    def get_scenes(self):
        existing_scenes = [int(val) for val in list(os.listdir(self.output_scans_root))]
        video_ids = []
        visit_ids = []
        for idx in range(len(self.all_scenarios)):
            if self.all_scenarios[idx][0] not in existing_scenes: #np.isnan(self.all_scenarios[idx][2]) or ():
                pass
            else:
                video_ids.append(self.all_scenarios[idx][0])
                visit_ids.append(self.all_scenarios[idx][2])

        scene_number = len(video_ids)
        testing_num = 100
        training_num = int((scene_number - testing_num) * 8.5 / 10)
        evaluation_num = scene_number - testing_num - training_num
        # training_num = int(scene_number * 7.0 / 10.0)
        # evaluation_num = int(scene_number * 2.0 / 10.0)
        # testing_num = scene_number - training_num - evaluation_num

        test_scenarios = []
        visited = []
        selected_indices = []
        indices = list(range(scene_number))
        random.shuffle(indices)
        while len(test_scenarios) < testing_num:
            random.shuffle(indices)
            index = indices.pop(0)
            visit_id = visit_ids[index]
            video_id = video_ids[index]
            if visit_id not in visited:
                selected_indices.append(index)
                test_scenarios.append(video_id)
                visited.append(visit_id)
            else:
                pass
        all_scenario_indices = set(list(range(scene_number)))
        selected_indices = set(selected_indices)

        evaluation_scenarios = []
        training_scenarios = []
        other_indices = list(all_scenario_indices - selected_indices)
        random.shuffle(other_indices)
        for i in range(len(other_indices)):
            if i < evaluation_num:
                evaluation_scenarios.append(video_ids[other_indices[i]])
            else:
                training_scenarios.append(video_ids[other_indices[i]])
        return training_scenarios, evaluation_scenarios, test_scenarios

    def process_frames(self, scene, file_type):
        root_path = join(self.root, file_type, scene, scene + "_frames")

        depth_folder = join(root_path, "lowres_depth")
        depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
        frame_ids = [os.path.basename(x) for x in depth_images]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        video_id = depth_folder.split('/')[-3]
        frame_ids = [x for x in frame_ids]
        frame_ids.sort()

        poses_from_traj = {}
        traj_file = os.path.join(root_path, 'lowres_wide.traj')
        with open(traj_file) as f:
            traj = f.readlines()
        for line in traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1]

        intrinsics = {}
        poses = []
        available_frame_ids = []
        # for idx in tqdm(range(len(frame_ids)), desc="process all frame {}".format(scene)):
        for idx in range(len(frame_ids)):
            frame_id = frame_ids[idx]
            # set intrinsics
            intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics", f"{video_id}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                            f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(root_path, "lowres_wide_intrinsics",
                                            f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("frame_id", frame_id)
                print(intrinsic_fn)
            intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)

            # set poses:
            frame_pose = None
            if str(frame_id) in poses_from_traj.keys():
                frame_pose = np.array(poses_from_traj[str(frame_id)])
            else:
                for my_key in list(poses_from_traj.keys()):
                    if abs(float(frame_id) - float(my_key)) < 0.005:
                        frame_pose = np.array(poses_from_traj[str(my_key)])
            if frame_pose is not None:
                available_frame_ids.append(frame_id)
                poses.append(frame_pose)
        return poses, intrinsics, available_frame_ids, video_id

    def _load_depth_image(self, root_path, fname):
        depth_image_path = os.path.join(root_path, "lowres_depth", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path, "does not exist")
        depth_image = cv2.imread(depth_image_path, -1) / 1000.0
        return depth_image

    def _load_process_rgbd_points(self, root_path, fname, intrinsic, pose):
        image_path = os.path.join(root_path, "lowres_wide", fname)
        rgb_image = cv2.imread(image_path)
        rgb_image[:, :, [0, 1, 2]] = rgb_image[:, :, [2, 1, 0]]
        im_height, im_width, im_channels = rgb_image.shape

        depth_image = self._load_depth_image(root_path=root_path, fname=fname)
        depth_height, depth_width = depth_image.shape

        if depth_height != im_height:
            rgb_image = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
            rgb_image[48: 48 + 192, 64: 64 + 256, :] = rgb_image
        (m, n, _) = rgb_image.shape

        pcd, rgb_feat = generate_point(rgb_image / 255.0, depth_image, intrinsic, 1, True, pose)
        points = np.concatenate((pcd, rgb_feat), axis=1)
        return rgb_image, depth_image, points

    def process_rgbd(self, scene, file_type, video_id, poses, intrinsics, frame_ids):
        root_path = join(self.root, file_type, scene, scene + "_frames")
        for idx in tqdm(range(len(frame_ids)), desc="process single rgbd {}".format(scene)):
            # for idx in range(len(frame_ids)):
            frame_id = frame_ids[idx]
            intrinsic = intrinsics[idx]
            pose = copy.deepcopy(poses[idx])

            fname = "{}_{}.png".format(video_id, frame_id)
            rgb_image, depth_image, points = self._load_process_rgbd_points(root_path=root_path, fname=fname,
                                                                            intrinsic=intrinsic, pose=pose)

            # folder = join(self.output_scans_root, scene, "single")
            # if not exists(folder):
            #     os.makedirs(folder)
            # with open(join(folder, "{}.pkl".format(idx)), 'wb') as handle:
            #     pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # self.display_colors(points)

            self._save_rgbd(data={"rgb": rgb_image, "depth": depth_image}, scene=scene, frame=idx)
            if self.save_visualize:
                self._display_rgbd(rgb=rgb_image, depth=depth_image, name=idx,
                                   folder=join(self.output_scans_root, scene, "visualize"))

    def load_points(self, scene):
        folder_name = join(self.output_scans_root, scene)
        with open(join(folder_name, "all_points.pkl"), "rb") as handle:
            all_points = pickle.load(handle)
        # with open(join(self.output_scans_root, scene, "all_poses.pkl"), "rb") as handle:
        #     poses = np.array(pickle.load(handle))
        # with open(join(self.output_scans_root, scene, "scene_info.pkl"), "rb") as handle:
        #     scene_info: SceneInfo = pickle.load(handle)
        # with open(join(self.output_scans_root, scene, "distance_ranges.pkl"), "rb") as handle:
        #     distance_ranges = pickle.load(handle)
        return all_points  # , poses, scene_info, distance_ranges

    def select_poses(self, scene, file_type, video_id, all_points, frame_ids, intrinsics, poses):
        root_path = join(self.root, file_type, scene, scene + "_frames")
        last_voxels = None
        frame_indices = []
        frame_poses = []
        frame_voxels = []
        frame_ranges = []
        frame_intrinsics = []

        for idx in tqdm(range(len(frame_ids)), desc="process poses {}".format(scene)):
            frame_id = frame_ids[idx]
            intrinsic = copy.deepcopy(intrinsics[frame_id])
            pose = copy.deepcopy(poses[idx])

            fname = "{}_{}.png".format(video_id, frame_id)
            depth_image = self._load_depth_image(root_path=root_path, fname=fname)
            depth_height, depth_width = depth_image.shape
            distance_threshold = min(self.distance_threshold, np.max(depth_image))
            c, selected_pts, _ = self.get_frustum_indices(points=all_points[:, :3], pose=pose, K=intrinsic,
                                                          width=depth_width, height=depth_height,
                                                          distance_threshold=distance_threshold)

            # rgb_image, depth_image, points = self._load_process_rgbd_points(root_path=root_path, fname=fname,
            #                                                                 intrinsic=intrinsic, pose=pose)
            # selected_pts = np.concatenate((selected_pts, all_points[c][:, 3:6]), axis=1)
            # self.display_colors(np.concatenate((selected_pts, points), axis=0))
            c = c[0]
            if len(c) < self.minimum_voxel_number:
                continue
            if last_voxels is None:
                last_voxels = set(c.tolist())
                frame_indices.append(frame_id)
                frame_poses.append(pose)
                frame_voxels.append(last_voxels)
                frame_intrinsics.append(intrinsic)
                frame_ranges.append([depth_height, depth_width, distance_threshold])
            else:
                current_set = set(c.tolist())
                overlap_set = list(current_set & last_voxels)
                coverage = len(overlap_set) / max(len(current_set), len(last_voxels))
                if coverage < self.frame_overlap_threshold:
                    # self.display_overlap(ref=last_voxels, src=current_set, points=all_points)
                    last_voxels = set(c.tolist())
                    frame_indices.append(frame_id)
                    frame_poses.append(pose)
                    frame_voxels.append(last_voxels)
                    frame_intrinsics.append(intrinsic)
                    frame_ranges.append([depth_height, depth_width, distance_threshold])
        # self.display_overlap(ref=frame_voxels[0], src=frame_voxels[18], points=all_points[:, :3])
        pos, neg = self.select_pos_neg(all_voxels=frame_voxels)
        with open(join(self.output_scans_root, scene, "frame_info.pkl"), 'wb') as handle:
            pickle.dump({"positives": pos, "negatives": neg, "poses": frame_poses, "frame_relation": frame_indices,
                         "intrinsics": frame_intrinsics}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return frame_poses, pos, neg, frame_indices, frame_ranges, frame_intrinsics

    def process_frustum(self, scene, all_points, intrinsics, poses, frame_ranges, file_type):
        annotations = []
        for idx in tqdm(range(len(poses)), desc="process frustum {}".format(scene)):
            intrinsic = copy.deepcopy(intrinsics[idx])
            pose = copy.deepcopy(poses[idx])
            depth_height, depth_width, distance_threshold = frame_ranges[idx]
            c, selected_pts, points_2d = self.get_frustum_indices(points=all_points[:, :3], pose=pose,
                                                                  K=intrinsic, width=depth_width, height=depth_height,
                                                                  distance_threshold=distance_threshold)
            selected_pts = np.concatenate((selected_pts, all_points[c][:, 3:]), axis=1)
            processed = self.get_points_normals(pts=selected_pts)

            folder = join(self.output_scans_root, scene, "fused")
            if not exists(folder):
                os.makedirs(folder)
            with open(join(folder, "{}.pkl".format(idx)), 'wb') as handle:
                pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            selected_bbx = get_bounding_boxes(
                annotation_file=join(self.root, file_type, scene, scene + "_3dod_annotation.json"),
                pts=processed[:, :3], pose=pose, points_limit=self.annotation_points_limit)
            annotations.append(selected_bbx)

            if self.save_visualize:
                self._points_to_2d(scene=scene, frame=idx, points_2d=points_2d[c], color=selected_pts[:, 3:6],
                                   height=depth_height, width=depth_width)
        with open(join(self.output_scans_root, scene, "annotations.pkl"), 'wb') as handle:
            pickle.dump(annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def select_similar_scenes(self, current_scene, all_scenes):
        ids = self.all_scenarios[:, 0].tolist()
        idx = ids.index(int(current_scene))
        if np.isnan(self.all_scenarios[idx][2]):
            return None
        else:
            sim_frames = self.all_scenarios[np.where(self.all_scenarios[:, 2] == self.all_scenarios[idx][2])]
            selected = []
            for scene in sim_frames[:, 0]:
                if scene != int(current_scene):
                    selected.append(str(scene))
            return selected

    def run_scenes(self):
        saved_scenarios = os.listdir(self.output_scans_root)
        downloaded_scenarios = os.listdir(join(self.root, "Training")) + os.listdir(join(self.root, "Validation"))

        for file_idx in range(len(self.scenes)):
            file_type = self.training_type[file_idx]
            scene = str(self.scenes[file_idx])
            if scene in saved_scenarios:
                pass
            elif scene not in downloaded_scenarios:
                print("scene is {} not downloaded".format(scene))
            else:
                ori_points = self.process_scene_points(scene=scene, file_type=file_type)
                all_points = self.save_all_points(scene=scene, all_points=ori_points)
                # all_points = self.load_points(scene=scene)

                all_poses, all_intrinsics, all_frame_ids, video_id = self.process_frames(scene=scene, file_type=file_type)

                downsampled_pts = self.downsample_points(all_points=all_points)
                poses, pos, neg, frame_indices, frame_ranges, intrinsics = self.select_poses(scene=scene,
                                                                                             file_type=file_type,
                                                                                             video_id=video_id,
                                                                                             all_points=downsampled_pts,
                                                                                             frame_ids=all_frame_ids,
                                                                                             intrinsics=all_intrinsics,
                                                                                             poses=all_poses)

                self.process_frustum(scene=scene, all_points=all_points, intrinsics=intrinsics,
                                     poses=poses, frame_ranges=frame_ranges, file_type=file_type)
                self.process_rgbd(scene, file_type, video_id, poses, intrinsics, frame_indices)


if __name__ == '__main__':
    from dataset.scannet_scenes import all_scenes
    import argparse

    # 41159453
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--scene', type=int, default=0, help="")
    parser.add_argument('--batch', type=int, default=10, help="")
    parser.add_argument('--stage', type=int, default=0,
                        help='1 process the single frames; 2 combine information of all frames for training; 3 select key frames for testing environment')
    args = parser.parse_args()

    # arkit = DoDProcessor(display=True, root="/media/jing/Data/data/3dod", scene_ind=args.scene,
    arkit = DoDProcessor(display=True, root="/media/jing/Data/data/3dod", scene_ind=args.scene,
                           output_root="/home/jing/Documents/work/data/ThreeDOD")
    # arkit = DoDProcessor(display=False, root="/scratch/jing/data/3dod", scene_ind=args.scene, batch=args.batch,
    #                        output_root="/media/jing/data/ThreeDOD")

    if args.stage == 1:
        arkit.run_scenes()
    elif args.stage == 2:
        testing_length = 200
        scene_num = len(arkit.scenes) - 200
        training_length = int(scene_num * 8.5 / 10)
        evaluation_length = scene_num - training_length
        for scenes, name in [[arkit.scenes[:training_length], "training"],
                             [arkit.scenes[training_length: training_length + evaluation_length], "evaluation"],
                             [arkit.scenes[training_length + evaluation_length:], "testing"]]:
            arkit.combine_info(scenes=scenes, name=name)
    else:
        arkit.process_key_frames(name="testing")

