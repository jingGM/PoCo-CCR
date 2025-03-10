```data
├── scans
│   ├── scene0 
│       ├── scene_info.pkl: scene information including intrinsic and extrinsic parameters of camera
│       ├── all_points.pkl: all points of the scenarios: xyz,rgb,label
│       ├── all_poses.pkl: poses of all frames before selecting key frames
│       ├── distance_ranges.pkl: the maximum points distance in each frame. This is used to clip the point cloud in each frustum
│       ├── frame_info.pkl: all the frame information of the scenario, including poses, positive and negative frame indices in the scenario, and also the relation between the current key frame index to the original frame index. 
│       ├── rgbd: the folder includes all the rgb and depth images of the key frames 
│       ├── fused: the folder includes all the frustum point clouds of the key frames
│       └── visualize: this folder is only for visualization. Set the display variable to Truth in the data generation function, the folder will not be generated;
│           ├── depth: depth images of the key frames
│           ├── rgb: rgb images of the key frames
│           └── pts_2d: the frustum project back to the image plane and generate the rgb images for the key frames
│   ├── scene1
│   ├── :
```