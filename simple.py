#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from utils import all_devices, get_frame

def align_point_clouds(source, target, threshold=0.002):
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def convert_to_point_cloud(depth, color):
    depth_image = o3d.geometry.Image(depth)
    color_image = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, convert_rgb_to_intensity=False
    )
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd

pipelines = all_devices()
try:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    while True:
        all_images = []        
        for pipeline in pipelines:
            frame = get_frame(pipeline)
            if not frame:
                continue

            all_images.append(frame)

        # to point cloud
        point_clouds = []
        for depth, color in all_images:
            pcd = convert_to_point_cloud(depth, color)
            # color = color.astype(np.float32) / 255.0
            # pcd.colors = o3d.utility.Vector3dVector(color)
            point_clouds.append(pcd)

        # Align and merge point clouds
        combined_pcd = point_clouds[0]
        for i in range(1, len(point_clouds)):
            transformation = align_point_clouds(point_clouds[i], combined_pcd)
            point_clouds[i].transform(transformation)
            combined_pcd += point_clouds[i]

        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.0001)
        
        # Update visualization
        vis.clear_geometries()
        vis.add_geometry(combined_pcd)
        vis.poll_events()
        vis.update_renderer()
        
finally:
    vis.destroy_window()
    for pipeline in pipelines:
        pipeline.stop()
