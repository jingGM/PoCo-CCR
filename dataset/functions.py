import open3d as o3d
from collections import Counter

from dataset.arkit_utils import extract_gt
from utils.visualization import visualize_single, display_two_points, display_bbx, display_three_points
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import random


def check_colors(saving_colors, current, threshold=50):
    tag = False
    for color in saving_colors:
        if abs(color[0] - current[0]) + abs(color[1] - current[1]) + abs(color[2] - current[2]) < threshold:
            tag = True
    return tag


# arkit_labels = ["unclassified", 'stool', 'table', 'toilet', 'fireplace', 'washer', 'tv_monitor', 'sink', 'cabinet',
#                 'stove', 'chair', 'sofa', 'shelf', 'dishwasher', 'refrigerator', 'bed', 'bathtub', 'oven']

label_to_colour = [ [255,255,255],  #0 [0, 0, 0],  # black -> 'unclassified'
                    [228, 38, 137],
                    [197, 64, 205],
                    [148, 158, 49],
                    [147, 120, 91],
                    [117, 85, 31],
                    [100, 225, 75],
                    [94, 232, 122],
                    [172, 20, 254],
                    [247, 23, 93],
                    [3, 70, 31],
                    [154, 211, 228],
                    [138, 252, 177],
                    [202, 3, 153],
                    [198, 188, 33],
                    [189, 110, 171],
                    [27, 245, 79],
                    [128, 171, 178],
                    [159, 227, 107],
                    [218, 90, 172],
                    [235, 124, 142],
                    [243, 91, 123],
                    [39, 119, 108],
                    [227, 142, 50],
                    [178, 116, 75],
                    [216, 131, 15],
                    [238, 101, 71],
                    [18, 203, 99],
                    [91, 107, 142],
                    [70, 6, 180],
                    [173, 100, 3],
                    [139, 6, 178],
                    [65, 228, 246],
                    [238, 75, 193],
                    [216, 34, 5],
                    [197, 191, 221],
                    [84, 173, 16],
                    [234, 208, 156],
                    [88, 163, 167],
                    [150, 207, 25],
                    [23, 35, 195], #40
                    ]

def display_overlap(ref, src, points):
    overlap = ref & src
    sref = ref - overlap
    ssrc = src - overlap
    display_three_points(a=points[(list(sref),)], b=points[(list(ssrc),)], c=points[(list(overlap),)])


def display_labels(points):
    label_colors = np.array([label_to_colour[int(key)] for key in points[:, -1]]) / 255.0
    visualize_single(np.concatenate((points[:, :3], label_colors), axis=1))


def display_boxes(pts, color, boxes_corners, labels):
    edge_indeces = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
    another_indices = [1, 3, 4, 2, 5, 3, 6, 7, 5, 7, 6, 7]
    edges_r = boxes_corners[:, edge_indeces, :] #np.reshape(, (-1, 3))
    edges_s = boxes_corners[:, another_indices, :] #np.reshape(, (-1, 3))

    edge_color = []
    for label in labels:
        # edge_color.append(np.array(label_to_colour[arkit_labels.index(label)]).reshape((1, 3)).repeat(len(edges_s), axis=0))
        # edge_color.append(np.array(label_to_colour[arkit_labels.index(label)] + [255]))
        edge_color.append(np.array([255, 0, 0, 255]))
    edge_color = np.stack(edge_color, axis=0) / 255.
    display_bbx(pts=pts, color=color, bbxr=edges_r, bbxs=edges_s, point_size=5, edge_size=20, edge_color=edge_color)

    print("test")


if __name__ == "__main__":
    import pickle

    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt("/home/jing/Documents/work/data/test/41048095/41048095_3dod_annotation.json")
    with open("/home/jing/Documents/work/data/test/41048095/all_points.pkl", "rb") as handle:
        all_pts = pickle.load(handle)
    max_z = max(all_pts[:, 2])
    pts = all_pts[np.where(all_pts[:, 2] < max_z - 0.5)]

    display_boxes(pts=pts[:, :3], color=pts[:, 3:6], boxes_corners=boxes_corners, sizes=sizes, labels=labels)
    print("test")

    # colors = [np.array([0, 0, 0]), np.array([70, 70, 70])]
    # for i in range(40):
    #     new_color = np.array([random.random(), random.random(), random.random()])
    #     tag = True
    #     while check_colors(saving_colors=colors, current=new_color):
    #         new_color = np.round(np.array([random.random(), random.random(), random.random()]) * 255.0).astype(int)
    #     colors.append(new_color)
    # print(np.array(colors))
    #
    # # Visualization
    # fig, ax = plt.subplots(figsize=(10, 2))
    # for i, color in enumerate(colors):
    #     ax.fill_between([i, i + 1], 0, 1, color=color / 255.0)
    #
    # ax.set_xlim(0, 40)
    # ax.set_ylim(0, 1)
    # ax.axis('off')
    # plt.show()