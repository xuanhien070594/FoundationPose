"""
This script records video with visualization of estimated pose and target pose using a webcam.
"""

import pathlib
import pickle
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

from image_acquisition import (
    WebcamImageAcquisition,
    WebCameraSettings,
    create_folder_if_not_exists,
)
from Utils import load_yaml_matrix

# Global stop event
stop_event = Event()


# Signal handling for graceful shutdown
def handle_signal(signum, frame):
    print("\nSignal received. Stopping recording...")
    stop_event.set()


signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl-C
signal.signal(signal.SIGTERM, handle_signal)  # Handle SIGTERM


# Main script
@click.command()
@click.option(
    "--webcam_alias", type=str, default="brio", help="Alias for the webcam to use."
)
@click.option(
    "--video_folder",
    type=str,
    default="video_data",
    help="Path to store recorded video files.",
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
    extrinsic_calibration_filename,
):
    # Setup directories and file paths
    code_dir = pathlib.Path(__file__).resolve().parent.parent
    video_folder = code_dir / video_folder
    create_folder_if_not_exists(video_folder)
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = f"{prefix}.mp4"
    saved_poses_file = f"{prefix}_poses.pkl"
    video_path = video_folder / video_file
    saved_poses_path = video_folder / saved_poses_file

    # Load camera calibration data
    camera_params_dir = code_dir / "camera_params/brio_webcam"
    cam_T_world = load_yaml_matrix(
        camera_params_dir / extrinsic_calibration_filename, "tf_world_to_camera"
    )

    # Initialize LCM and camera
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
    cube_state_subscriber = CubePoseLcmSubscriber(lc, "CUBE_STATE", timeout=1e-4)
    cube_target_state_subscriber = CubePoseLcmSubscriber(
        lc, "CUBE_TARGET_STATE", timeout=1e-4
    )
    video_cap = WebcamImageAcquisition(WebCameraSettings(webcam_alias))
    video_writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_cap.settings.frame_rate,
        (video_cap.settings.width, video_cap.settings.height),
    )
    time.sleep(3)  # Allow camera to stabilize

    estimated_poses, target_poses = [], []

    # Start capturing frames
    print("Recording... Press Ctrl-C to stop.")
    while not stop_event.is_set():
        if select.select([lc.fileno()], [], [], cube_state_subscriber.timeout)[0]:
            lc.handle()
        cur_estimated_pose = cube_state_subscriber.get_data()
        cur_target_pose = cube_target_state_subscriber.get_data()

        if cur_estimated_pose is None:
            print("No estimated pose received. Skipping frame...")
            continue

        cam_T_object = cam_T_world @ cur_estimated_pose

        if cur_target_pose is None:
            cam_T_target = None
        else:
            cam_T_target = cam_T_world @ cur_target_pose

        frame = video_cap.capture_image()

        video_writer.write(frame)
        estimated_poses.append(cam_T_object)
        target_poses.append(cam_T_target)

    # Annotate and save frames if recording was interrupted
    if stop_event.is_set():
        print("Recording stopped. Saving video...")
        saved_poses = {
            "estimated_poses": estimated_poses,
            "target_poses": target_poses,
        }
        with open(saved_poses_path, "wb") as file:
            pickle.dump(saved_poses, file)

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
