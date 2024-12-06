# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
import datetime
import time

import pyrealsense2 as rs
import yaml

from datareader import *
from estimater import *
from mask import *

from foundationpose.lcm_systems.pose_publisher import CubePoseLcmPublisher


def get_world_T_cam_from_yaml(file_name: str) -> np.ndarray:
    with open(file_name) as file:
        data_loaded = yaml.safe_load(file)
    cam_T_world = np.array(data_loaded["tf_world_to_camera"]["data"]).reshape(4, 4)
    world_T_cam = np.eye(4)
    world_T_cam[:3, :3] = cam_T_world[:3, :3].T
    world_T_cam[:3, 3] = -cam_T_world[:3, :3].T @ cam_T_world[:3, 3]
    return world_T_cam


def get_intrinsic_matrix_from_yaml(file_name: str) -> np.ndarray:
    with open(file_name) as file:
        data_loaded = yaml.safe_load(file)
    return np.array(data_loaded["camera_matrix"]["data"]).reshape(3, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    assert code_dir != ""
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/../assets/cube/cube.obj",
    )
    parser.add_argument(
        "--intrinsic_params_file",
        type=str,
        default=f"{code_dir}/../camera_params/realsense_d435i/cv2_rgb_camera_intrinsics_848_480.yaml",
    )
    parser.add_argument(
        "--extrinsic_params_file",
        type=str,
        default=f"{code_dir}/../camera_params/realsense_d435i/cv2_rgb_camera_extrinsics_848_480.yaml",
    )
    parser.add_argument("--est_refine_iter", type=int, default=20)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    print(f"This is the mesh file: {args.mesh_file}")
    mesh = trimesh.load(args.mesh_file, process=False, maintain_order=True)
    print("Loaded mesh file")

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("Estimator initialization done")

    create_mask()
    mask = cv2.imread("mask.png")

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 60)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    # clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    cam_K = get_intrinsic_matrix_from_yaml(args.intrinsic_params_file)
    world_T_cam = get_world_T_cam_from_yaml(args.extrinsic_params_file)
    pose_correction = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    lcm_pose_publisher = CubePoseLcmPublisher()

    # ------------ Pose Estimation Loop ------------ #
    i = 0
    estimating = True
    time.sleep(3)  # Sleep for 3 seconds to allow the camera to warm up

    checking_pose_estimation = True

    try:
        while estimating:
            start_time = time.perf_counter()
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data()) / 1e3
            color_image = np.asanyarray(color_frame.get_data())

            # Scale depth image to mm
            depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

            if cv2.waitKey(1) == 13:
                estimating = False
                break

            logging.info(f"i:{i}")

            H, W = cv2.resize(color_image, (848, 480)).shape[:2]
            color = cv2.resize(color_image, (W, H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(
                depth_image_scaled, (W, H), interpolation=cv2.INTER_NEAREST
            )

            depth[(depth < 0.1) | (depth >= np.inf)] = 0

            if i == 0:
                if len(mask.shape) == 3:
                    for c in range(3):
                        if mask[..., c].sum() > 0:
                            mask = mask[..., c]
                            break
                mask = (
                    cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    .astype(bool)
                    .astype(np.uint8)
                )

                pose = est.register(
                    K=cam_K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=args.est_refine_iter,
                )

                if debug >= 3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f"{debug_dir}/model_tf.obj")
                    xyz_map = depth2xyzmap(depth, cam_K)
                    valid = depth >= 0.1
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(f"{debug_dir}/scene_complete.ply", pcd)

            else:
                pose = est.track_one(
                    rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter
                )
            cam_to_object = pose @ pose_correction
            cam_to_object[2, 3] -= 0.005
            obj_pose_in_world = world_T_cam @ cam_to_object

            if (debug == 0 and checking_pose_estimation) or debug >= 1:
                os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
                np.savetxt(f"{debug_dir}/ob_in_cam/{i}.txt", pose.reshape(4, 4))

                vis = draw_posed_3d_box(
                    cam_K, img=color, ob_in_cam=cam_to_object, bbox=bbox, linewidth=1
                )
                vis = draw_xyz_axis(
                    color,
                    ob_in_cam=cam_to_object,
                    scale=0.1,
                    K=cam_K,
                    thickness=1,
                    transparency=0,
                    is_input_rgb=True,
                )
                cv2.imshow("debug", vis[..., ::-1])
                key = cv2.waitKey(1)

                # (Debug 0) Close GUI window after checking intial pose estimation is correct
                if debug == 0 and (key == ord("q")):
                    cv2.destroyWindow("debug")
                    checking_pose_estimation = False

            if debug >= 2:
                os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
                imageio.imwrite(f"{debug_dir}/track_vis/{i}.png", vis)

            lcm_pose_publisher.pub_pose(
                obj_pose_in_world, int(time.perf_counter() * 1e6)
            )
            print(obj_pose_in_world)
            i += 1
            print(f"Pose estimation time: {time.perf_counter() - start_time}")

    finally:
        pipeline.stop()
