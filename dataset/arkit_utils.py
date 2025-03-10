import json

import numpy as np
import scipy

from utils.functions import transform_pts

class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed",  # 0..5
    "sink", "washer", "toilet", "bathtub", "oven",  # 5..10
    "dishwasher", "fireplace", "stool", "chair", "table",  # 10..15
    "tv_monitor", "sofa",  # 15..17
]

# 3D Anchor-sizes of merged categories (dx, dy, dz)
'''
   Anchor box sizes are computed based on box corner order below:
       6 -------- 7
      /|         /|
     5 -------- 4 .
     | |        | |
     . 2 -------- 3
     |/         |/
     1 -------- 0 
'''


def compute_box_3d(size, center, rotmat):
    """Compute corners of a single box from rotation matrix
    Args:
        size: list of float [dx, dy, dz]
        center: np.array [x, y, z]
        rotmat: np.array (3, 3)
    Returns:
        corners: (8, 3)
    """
    l, h, w = [i / 2 for i in size]
    center = np.reshape(center, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(
        np.transpose(rotmat), np.vstack([x_corners, y_corners, z_corners])
    )
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def get_size(box):
    """
    Args:
        box: 8x3
    Returns:
        size: [dx, dy, dz]
    """
    distance = scipy.spatial.distance.cdist(box[0:1, :], box[1:5, :])
    l = distance[0, 2]
    w = distance[0, 0]
    h = distance[0, 3]
    return [l, w, h]


class ARKitDatasetConfig(object):
    def __init__(self):
        """
        init will set values for:
            self.class_names
            self.cls2label (after mapping)
            self.label2cls (after mapping)
            self.num_class

        Args:
        """
        # final training/val categories
        self.class_names = class_names
        self.label2cls = {}
        self.cls2label = {}
        for i, cls_ in enumerate(class_names):
            self.label2cls[i] = cls_
            self.cls2label[cls_] = i

        self.num_class = len(self.class_names)


def extract_gt(gt_fn):
    """extract original label data

    Args:
        gt_fn: str (file name of "annotation.json")
            after loading, we got a dict with keys
                'data', 'stats', 'comment', 'confirm', 'skipped'
            ['data']: a list of dict for bboxes, each dict has keys:
                'uid', 'label', 'modelId', 'children', 'objectId',
                'segments', 'hierarchy', 'isInGroup', 'labelType', 'attributes'
                'label': str
                'segments': dict for boxes
                    'centroid': list of float (x, y, z)?
                    'axesLengths': list of float (x, y, z)?
                    'normalizedAxes': list of float len()=9
                'uid'
            'comments':
            'stats': ...
    Returns:
        skipped: bool
            skipped or not
        boxes_corners: (n, 8, 3) box corners
            **world-coordinate**
        centers: (n, 3)
            **world-coordinate**
        sizes: (n, 3) full-sizes (no halving!)
        labels: list of str
        uids: list of str
    """
    gt = json.load(open(gt_fn, "r"))
    skipped = gt['skipped']
    if len(gt) == 0:
        boxes_corners = np.zeros((0, 8, 3))
        centers = np.zeros((0, 3))
        sizes = np.zeros((0, 3))
        labels, uids = [], []
        return skipped, boxes_corners, centers, sizes, labels, uids

    boxes_corners = []
    centers = []
    sizes = []
    labels = []
    uids = []
    for data in gt['data']:
        l = data["label"]
        for delimiter in [" ", "-", "/"]:
            l = l.replace(delimiter, "_")
        if l not in class_names:
            print("unknown category: %s" % l)
            continue

        rotmat = np.array(data["segments"]["obbAligned"]["normalizedAxes"]).reshape(
            3, 3
        )
        center = np.array(data["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        size = np.array(data["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(size.reshape(3).tolist(), center, rotmat)

        '''
            Box corner order that we return is of the format below:
                6 -------- 7
               /|         /|
              5 -------- 4 .
              | |        | |
              . 2 -------- 3
              |/         |/
              1 -------- 0 
        '''

        boxes_corners.append(box3d.reshape(1, 8, 3))
        size = np.array(get_size(box3d)).reshape(1, 3)
        center = np.mean(box3d, axis=0).reshape(1, 3)

        # boxes_corners.append(box3d.reshape(1, 8, 3))
        centers.append(center)
        sizes.append(size)
        # labels.append(l)
        labels.append(data["label"])
        uids.append(data["uid"])
    # if len(centers) == 0:
    try:
        centers = np.concatenate(centers, axis=0)
        sizes = np.concatenate(sizes, axis=0)
        boxes_corners = np.concatenate(boxes_corners, axis=0)
        return skipped, boxes_corners, centers, sizes, labels, uids
    except:
        boxes_corners = np.zeros((0, 8, 3))
        centers = np.zeros((0, 3))
        sizes = np.zeros((0, 3))
        labels, uids = [], []
        return skipped, boxes_corners, centers, sizes, labels, uids


def remove_boxes(boxes, points, points_limit=20):
    selected_boxes = []
    for bid in range(len(boxes)):
        box = boxes[bid]
        transformation = find_transformation(box)
        points_t = transform_pts(pts=points[:, :3], transform=transformation)

        y_max = np.linalg.norm(box[1] - box[0])
        x_max = -np.linalg.norm(box[3] - box[0])
        z_max = np.linalg.norm(box[4] - box[0])
        selece_indices = (points_t[:, 0] > x_max) * (points_t[:, 0] < 0) * (points_t[:, 1] > 0) * (
                points_t[:, 1] < y_max) * (points_t[:, 2] > 0) * (points_t[:, 2] < z_max)
        pts = points_t[np.where(selece_indices)]

        if len(pts) > points_limit:
            max_xyz = np.max(pts, axis=0)
            min_xyz = np.min(pts, axis=0)
            box_value = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]],
                                  [min_xyz[0], max_xyz[1], min_xyz[2]],
                                  [max_xyz[0], max_xyz[1], min_xyz[2]],
                                  [max_xyz[0], min_xyz[1], min_xyz[2]],
                                  [min_xyz[0], min_xyz[1], max_xyz[2]],
                                  [min_xyz[0], max_xyz[1], max_xyz[2]],
                                  [max_xyz[0], max_xyz[1], max_xyz[2]],
                                  [max_xyz[0], min_xyz[1], max_xyz[2]]])
            box_va = transform_pts(box_value, transform=np.linalg.inv(transformation))
            selected_boxes.append(box_va)
    return selected_boxes


def find_transformation(box):
    points_src = np.array([box[0], box[1], box[3], box[4]])
    y_max = np.linalg.norm(box[1] - box[0])
    x_max = -np.linalg.norm(box[3] - box[0])
    z_max = np.linalg.norm(box[4] - box[0])
    points_dst = np.array([[0, 0, 0],
                           [0, y_max, 0],
                           [x_max, 0, 0],
                           [0, 0, z_max]])
    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    points_src_centered = points_src - centroid_src
    points_dst_centered = points_dst - centroid_dst

    H = np.dot(points_src_centered.T, points_dst_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_dst.T - np.dot(R, centroid_src.T)
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    return transformation_matrix


def get_bounding_boxes(annotation_file, pts, pose, points_limit=50):
    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(annotation_file)

    pose = np.linalg.inv(pose)
    boxes_corners = np.array(boxes_corners)
    B, N, C = boxes_corners.shape
    boxes_corners_re = boxes_corners.reshape((B * N, C))
    boxes_corners_re = transform_pts(pts=boxes_corners_re, transform=pose).reshape((B, N, C))
    selected_boxes = remove_boxes(boxes_corners_re, points=pts, points_limit=points_limit)
    if len(selected_boxes) > 0:
        selected_boxes = np.stack(selected_boxes)
    return selected_boxes


if __name__ == "__main__":
    import os

    folder = ""
    scenes = os.listdir(folder)
    all_labels = {}
    for scene in scenes:
        skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(
            os.path.join(folder, scene, "{}_3dod_annotation.json".format(scene)))
        all_labels += set(labels)
    print(all_labels)
