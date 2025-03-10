import copy

import open3d
import open3d as o3d
import torch
import numpy as np
from multiprocessing import Process

def visualize_single(pts):
    voxels = o3d.geometry.PointCloud()
    voxels.points = o3d.utility.Vector3dVector(pts[:, :3])

    if pts.shape[1] >= 9:
        voxels.normals = o3d.utility.Vector3dVector(pts[:, 3:6])
        voxels.colors = o3d.utility.Vector3dVector(pts[:, 6:9])
    elif pts.shape[1] >= 6:
        voxels.colors = o3d.utility.Vector3dVector(pts[:, 3:6])
    o3d.visualization.draw_geometries([voxels])


def display_bbx(pts, bbxr, bbxs, color=None, edge_size=2, edge_color=None, frame=None, frame_size=0.2, point_size=2):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)
    if color is None:
        color = [1, 0, 0, 1]
    add_cloud(widget3d.scene, "ref", pts, color, point_size)

    if frame is not None:
        add_frame(scene=widget3d.scene, name="ref_frame", frame_size=frame_size, frame=frame)

    for idx in range(len(bbxr)):
        add_edges(widget3d.scene, "edges_{}".format(idx), bbxr[idx], bbxs[idx], edge_color[idx], edge_size)
    app.run()


def display_single_points(pts, color=None, point_size=5, frame_size=0.2, frame_coordinate=None, file_name="",
                          center=[0, 0, 0]):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)
    if color is None:
        color = [1, 0, 0, 1]
    add_cloud(widget3d.scene, "ref", pts, color, point_size)

    if frame_coordinate is None:
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif frame_coordinate.shape[0] == 3 and len(frame_coordinate.shape) == 1:
        pose = np.array([[1, 0, 0, frame_coordinate[0]], [0, 1, 0, frame_coordinate[1]],
                         [0, 0, 1, frame_coordinate[2]], [0, 0, 0, 1]])
    elif frame_coordinate.shape == (3, 4):
        pose = np.eye(4)
        pose[:3, :] = frame_coordinate
    else:
        pose = frame_coordinate
    add_frame(scene=widget3d.scene, name="ref_frame", frame_size=frame_size, frame=pose)

    widget3d.setup_camera(60, widget3d.scene.bounding_box, center)
    # if file_name:
    #     widget3d.capture_screen_image(file_name)
    app.run()


def display_frames(pts, color=None, frames=None, point_size=5, frame_size=0.5, center=[0, 0, 0]):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)
    if color is None:
        color = [1, 0, 0, 1]
    add_cloud(widget3d.scene, "ref", pts, color, point_size)

    if frames is not None:
        for i in range(len(frames)):
            add_frame(scene=widget3d.scene, name="{}_frames".format(i), frame_size=frame_size, frame=frames[i])

    widget3d.setup_camera(60, widget3d.scene.bounding_box, center)
    # if file_name:
    #     widget3d.capture_screen_image(file_name)
    app.run()

def display_points(pts, clrs, point_size=5):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)

    for i in range(len(pts)):
        add_cloud(widget3d.scene, "pts_{}".format(i), pts[i], clrs[i], point_size)
    app.run()

def display_two_points(ref, src, ref_color=None, src_color=None, other=None, point_size=5, src_size=5, edge_size=2, draw_edges=False):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)

    add_frame(scene=widget3d.scene, name="ref_frame", frame_size=0.2, frame=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    if ref_color is None:
        ref_color = [1, 0, 0, 1]
    add_cloud(widget3d.scene, "ref", ref, ref_color, point_size)

    if src_color is None:
        src_color = [0, 0, 1, 1]
    add_cloud(widget3d.scene, "src", src, src_color, src_size)
    if other is not None and len(other) > 0:
        add_cloud(widget3d.scene, "other", other, [1, 1, 0, 1], point_size)
    if draw_edges:
        add_edges(widget3d.scene, "edges", ref, src, [0, 0.3, 0, 1], edge_size)
    app.run()

def display_three_points(a, b, c, point_size=5):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)

    add_frame(scene=widget3d.scene, name="ref_frame", frame_size=0.2, frame=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    a_color = [1, 0, 0, 1]
    b_color = [0, 1, 0, 1]
    c_color = [0, 0, 1, 1]

    add_cloud(widget3d.scene, "a", a, a_color, point_size)
    add_cloud(widget3d.scene, "b", b, b_color, point_size)
    add_cloud(widget3d.scene, "c", c, c_color, point_size)
    app.run()


def add_frame(scene, name, frame, frame_size):
    if isinstance(frame, torch.Tensor):
        try:
            frame = frame.cpu().numpy()
        except:
            frame = frame.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=np.array([0.0, 0.0, 0.0]))
    mesh = copy.deepcopy(mesh)
    mesh.rotate(frame[:3, :3], center=(0, 0, 0))
    mesh.translate((frame[0, 3], frame[1, 3], frame[2, 3]))

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    mtl.shader = "defaultUnlit"
    scene.add_geometry(name, mesh, mtl)


def add_cloud(scene, name, pts, color, size):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = size

    if isinstance(pts, torch.Tensor):
        try:
            pts = pts.cpu().numpy()
        except:
            pts = pts.detach().cpu().numpy()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if len(np.asarray(color).shape) > 1:
        cloud.colors = o3d.utility.Vector3dVector(color)
    else:
        material.base_color = color
    scene.add_geometry(name, cloud, material)


def add_edges(scene, name, pt0, pt1, color, size):
    if isinstance(pt0, torch.Tensor):
        pt0 = pt0.cpu().numpy()
    if isinstance(pt1, torch.Tensor):
        pt1 = pt1.cpu().numpy()
    assert pt0.shape[0] == pt1.shape[0], Exception("the two poins have different sizes")
    lns = np.array([[i, i + len(pt0)] for i in range(len(pt0))])
    if isinstance(lns, torch.Tensor):
        lns = lns.cpu().numpy()
    assert lns.shape[1] == 2, Exception("the lines shape is not correct")
    lines = open3d.geometry.LineSet()
    lines.points = open3d.utility.Vector3dVector(np.concatenate((pt0, pt1), axis=0)[:, :3])
    lines.lines = open3d.utility.Vector2iVector(lns)
    # ln_c = np.zeros((lns.shape[0], 3))
    # ln_c[:, 1] += 1

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    color = np.array(color)
    if len(color.shape) == 1:
        material.base_color = color
    else:
        lines.colors = open3d.utility.Vector3dVector(color)
    material.point_size = size
    scene.add_geometry(name, lines, material)


# def display_two_points(ref, src, lns=None, point_size=2, draw_origin=True):
#     if isinstance(ref, torch.Tensor):
#         ref = ref.cpu().numpy()
#     if isinstance(src, torch.Tensor):
#         src = src.cpu().numpy()
#     ref_points = ref[:, :3]
#     src_points = src[:, :3]
#     all_points = np.concatenate((src_points, ref_points), axis=0)
#
#     src_color = np.zeros_like(src_points)
#     src_color[:, 0] += 1
#     ref_color = np.zeros_like(ref_points)
#     ref_color[:, [0, 1]] += 1
#     point_colors = np.concatenate((src_color, ref_color), axis=0)
#
#
#
#     vis = open3d.visualization.Visualizer()
#     vis.create_window()
#     vis.get_render_option().background_color = np.zeros(3)
#     vis.get_render_option().point_size = point_size
#     if draw_origin:
#         axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#         vis.add_geometry(axis_pcd)
#
#     pts = open3d.geometry.PointCloud()
#     pts.points = open3d.utility.Vector3dVector(all_points[:, :3])
#     pts.colors = open3d.utility.Vector3dVector(point_colors)
#     vis.add_geometry(pts)
#
#     if isinstance(lns, torch.Tensor):
#         lns = lns.cpu().numpy()
#     assert lns.shape[1] == 2, Exception("the lines shape is not correct")
#     ln_c = np.zeros((lns.shape[0], 3))
#     ln_c[:, 1] += 1
#     lines = open3d.geometry.LineSet()
#     lines.points = open3d.utility.Vector3dVector(all_points[:, :3])
#     lines.lines = open3d.utility.Vector2iVector(lns)
#     lines.colors = open3d.utility.Vector3dVector(ln_c)
#     vis.add_geometry(lines)
#
#     vis.run()
#     vis.destroy_window()