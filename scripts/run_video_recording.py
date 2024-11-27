"""
This script records video with visualization of estimated pose and target pose using a webcam.
"""

import pathlib
import select
import signal
import sys
import time
from datetime import datetime
from threading import Event

import click
import cv2
import lcm
import numpy as np
import yaml
from foundationpose.lcm_systems.pose_subscriber import CubePoseLcmSubscriber
from tqdm import tqdm

from Utils import draw_xyz_axis, draw_posed_3d_box
from image_acquisition import (
    WebcamImageAcquisition,
    WebCameraSettings,
    create_folder_if_not_exists,
)

# Constants for colors
GREEN = (0, 255, 0)
PURPLE = (255, 51, 255)

# Global stop event
stop_event = Event()


# Signal handling for graceful shutdown
def handle_signal(signum, frame):
    print("\nSignal received. Stopping recording...")
    stop_event.set()


signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl-C
signal.signal(signal.SIGTERM, handle_signal)  # Handle SIGTERM


# Utility functions
def load_yaml_matrix(file_path: str, key: str) -> np.ndarray:
    """Load a matrix from a YAML file."""
    with open(file_path) as file:
        data = yaml.safe_load(file)
    return np.array(data[key]["data"]).reshape(
        -1, 4 if key == "tf_world_to_camera" else 3
    )


def draw_axes_and_bbox(
    frame, intrinsic_matrix, cam_T, bbox, axis_scale=0.05, line_color=None, skip_bbox=False
):
    """Annotate a frame with axes and bounding box."""
    annotated = draw_xyz_axis(
        frame, ob_in_cam=cam_T, scale=axis_scale, K=intrinsic_matrix, thickness=2
    )
    if skip_bbox:
        return annotated
    return draw_posed_3d_box(
        intrinsic_matrix,
        img=annotated,
        ob_in_cam=cam_T,
        bbox=bbox,
        line_color=line_color,
    )


def annotate_and_save_frames(
    frames, poses, targets, video_writer, intrinsic_matrix, bbox
):
    """Add annotations to frames and save them to a video."""
    print(f"There are total {len(frames)} frames.")
    print("Annotating and saving video...")

    for frame, cam_T_object, cam_T_target in tqdm(
        zip(frames, poses, targets),
        total=len(frames),
        desc="Processing frames",
        unit="frame",
    ):
        if frame is None:
            continue
        annotated_frame = draw_axes_and_bbox(
            frame, intrinsic_matrix, cam_T_object, bbox, line_color=GREEN
        )
        annotated_frame = draw_axes_and_bbox(
            annotated_frame, intrinsic_matrix, cam_T_target, bbox, line_color=PURPLE
        )
        video_writer.write(annotated_frame)


# Main script
@click.command()
@click.option(
    "--webcam_alias", type=str, default="brio", help="Alias for the webcam to use."
)
@click.option("--video_folder", type=str, help="Path to store recorded video files.")
@click.option(
    "--intrinsic_calibration_filename",
    type=str,
    default="cv2_rgb_camera_intrinsics.yaml",
    help="File for intrinsic camera calibration data.",
)
@click.option(
    "--extrinsic_calibration_filename",
    type=str,
    default="cv2_rgb_camera_extrinsics.yaml",
    help="File for extrinsic camera calibration data.",
)
def main(
    webcam_alias,
    video_folder,
    intrinsic_calibration_filename,
    extrinsic_calibration_filename,
):
    # Setup directories and file paths
    code_dir = pathlib.Path(__file__).resolve().parent
    video_folder = pathlib.Path(video_folder)
    create_folder_if_not_exists(video_folder)
    video_file = datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
    video_path = video_folder / video_file

    # Load camera calibration data
    camera_params_dir = code_dir / "../camera_params/brio_webcam"
    intrinsic_matrix = load_yaml_matrix(
        camera_params_dir / intrinsic_calibration_filename, "camera_matrix"
    )
    cam_T_world = load_yaml_matrix(
        camera_params_dir / extrinsic_calibration_filename, "tf_world_to_camera"
    )

    # Default poses and bounding box
    cur_estimated_pose = np.eye(4)
    cur_estimated_pose[:3, 3] = [-0.0325, 0.0325, 0.0325]
    cur_target_pose = np.eye(4)
    cur_target_pose[:3, 3] = [0.0, 0.0, 0.1]
    bbox = np.array([[-0.0325, -0.0325, -0.0325], [0.0325, 0.0325, 0.0325]])

    # Initialize LCM and camera
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
    cube_state_subscriber = CubePoseLcmSubscriber(lc, "CUBE_STATE", timeout=1e-4)
    video_cap = WebcamImageAcquisition(WebCameraSettings(webcam_alias))
    video_writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_cap.settings.frame_rate,
        (video_cap.settings.width, video_cap.settings.height),
    )
    time.sleep(3)  # Allow camera to stabilize

    # Frame storage
    frames, estimated_poses, target_poses = [], [], []

    # Start capturing frames
    print("Recording... Press Ctrl-C to stop.")
    while not stop_event.is_set():
        if select.select([lc.fileno()], [], [], cube_state_subscriber.timeout)[0]:
            lc.handle()

        frame = video_cap.capture_image()
        cur_estimated_pose = cube_state_subscriber.get_data()
        cam_T_object = cam_T_world @ cur_estimated_pose
        cam_T_target = cam_T_world @ cur_target_pose

        frames.append(frame)
        estimated_poses.append(cam_T_object)
        target_poses.append(cam_T_target)

    # Annotate and save frames if recording was interrupted
    if stop_event.is_set() and frames:
        annotate_and_save_frames(
            frames, estimated_poses, target_poses, video_writer, intrinsic_matrix, bbox
        )

    # Cleanup resources
    video_cap.close_connection()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
