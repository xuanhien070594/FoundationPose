"""This script is used to record video with visualization of estimated pose and target pose using webcam."""

import pathlib
import select

import click
import numpy as np
import yaml
import cv2
import time
import lcm
from datetime import datetime
from image_acquisition import (
    WebcamImageAcquisition,
    WebCameraSettings,
    create_folder_if_not_exists,
)
from Utils import draw_xyz_axis, draw_posed_3d_box
from foundationpose.lcm_systems.pose_subscriber import CubePoseLcmSubscriber

GREEN = (0, 255, 0)
PURPLE = (255, 51, 255)


def get_cam_T_world_from_yaml(file_name: str) -> np.ndarray:
    with open(file_name) as file:
        data_loaded = yaml.safe_load(file)
    return np.array(data_loaded["tf_world_to_camera"]["data"]).reshape(4, 4)


def get_intrinsic_matrix_from_yaml(file_name: str) -> np.ndarray:
    with open(file_name) as file:
        data_loaded = yaml.safe_load(file)
    return np.array(data_loaded["camera_matrix"]["data"]).reshape(3, 3)


def draw_axes_and_bbox(
    frame: np.ndarray,
    intrinsic_matrix: np.ndarray,
    cam_T: np.ndarray,
    bbox: np.ndarray,
    axis_scale: float = 0.05,
    line_thickness: float = 2,
    line_transparency: float = 0,
    is_input_rgb: bool = True,
    line_color: bool = None,
):
    annotated = draw_xyz_axis(
        frame,
        ob_in_cam=cam_T,
        scale=axis_scale,
        K=intrinsic_matrix,
        thickness=line_thickness,
        transparency=line_transparency,
        is_input_rgb=is_input_rgb,
    )
    annotated = draw_posed_3d_box(
        intrinsic_matrix,
        img=annotated,
        ob_in_cam=cam_T,
        bbox=bbox,
        line_color=line_color,
    )
    return annotated


@click.command
@click.option(
    "--webcam_alias",
    type=str,
    default="brio",
    help="Alias for the webcam to be used for recording",
)
@click.option(
    "--video_folder", type=str, help="Path to the folder that stores the video files"
)
@click.option(
    "--intrinsic_calibration_filename",
    type=str,
    default="cv2_rgb_camera_intrinsics.yaml",
    help="""Filename used for saving intrinsic calibration
                        data or loading it""",
)
@click.option(
    "--extrinsic_calibration_filename",
    type=str,
    default="cv2_rgb_camera_extrinsics.yaml",
    help="""Filename used for saving intrinsic calibration
                         data.""",
)
def main(
    webcam_alias: str,
    video_folder: str,
    intrinsic_calibration_filename: str,
    extrinsic_calibration_filename: str,
):
    code_dir = pathlib.Path(__file__).resolve().parent
    video_folder = pathlib.Path(video_folder)
    create_folder_if_not_exists(video_folder)
    video_file = pathlib.Path(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    video_path = video_folder / video_file

    camera_params_dir = code_dir / "../camera_params/brio_webcam"
    intrinsic_calibration_filepath = camera_params_dir / intrinsic_calibration_filename
    extrinsic_calibration_filepath = camera_params_dir / extrinsic_calibration_filename
    cam_T_world = get_cam_T_world_from_yaml(extrinsic_calibration_filepath)
    intrinsic_matrix = get_intrinsic_matrix_from_yaml(intrinsic_calibration_filepath)

    # Define the estimated and target pose of the cube
    cur_estimated_cube_pose = np.eye(4)
    cur_estimated_cube_pose[:3, 3] = np.array([0.0, 0.0, 0.0325])
    cur_target_cube_pose = np.eye(4)
    cur_target_cube_pose[:3, 3] = np.array([0.0, 0.0, 0.1])
    bbox = np.array([[-0.0325, -0.0325, -0.0325], [0.0325, 0.0325, 0.0325]])

    # waiting_state_lcm_msg_timeout = 1e-4
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
    cube_state_lcm_channel = "CUBE_STATE"
    waiting_state_lcm_msg_timeout = 1e-4
    cube_state_subscriber = CubePoseLcmSubscriber(
        lc, cube_state_lcm_channel, waiting_state_lcm_msg_timeout
    )

    video_cap = WebcamImageAcquisition(WebCameraSettings(webcam_alias))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        video_path.as_posix(),
        fourcc,
        video_cap.settings.frame_rate,
        (video_cap.settings.width, video_cap.settings.height),
    )
    time.sleep(3)  # Wait for the camera to stabilize

    while True:
        rfds, wfds, efds = select.select(
            [lc.fileno()], [], [], cube_state_subscriber.timeout
        )
        if rfds:
            lc.handle()

        cam_T_object = cam_T_world @ cur_estimated_cube_pose
        cam_T_target = cam_T_world @ cur_target_cube_pose

        frame = video_cap.capture_image()

        annotated_frame = draw_axes_and_bbox(
            frame, intrinsic_matrix, cam_T_object, bbox, line_color=GREEN
        )
        annotated_frame = draw_axes_and_bbox(
            annotated_frame,
            intrinsic_matrix,
            cam_T_target,
            bbox,
            line_color=PURPLE,
        )
        video_writer.write(annotated_frame)

        cv2.imshow("frame", annotated_frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord("q"):
            break

    video_cap.close_connection()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
