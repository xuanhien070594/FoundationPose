"""
This script is used to annotate each frame of a video with visualization of estimated pose and target pose.
"""

import pathlib
import pickle
import sys

import click
import cv2
import numpy as np
from tqdm import tqdm

from Utils import annotate_and_save_single_frame, load_yaml_matrix


# Main script
@click.command()
@click.option(
    "--video_folder",
    type=str,
    default="video_data",
    help="Path to store recorded video files.",
)
@click.option("--video_file", type=str, help="Name of video we want to annotate.")
@click.option(
    "--saved_poses_file", type=str, help="Filename that stores all saved poses."
)
@click.option(
    "--intrinsic_calibration_filename",
    type=str,
    default="cv2_rgb_camera_intrinsics.yaml",
    help="File for intrinsic camera calibration data.",
)
def main(
    video_folder,
    video_file,
    saved_poses_file,
    intrinsic_calibration_filename,
):
    # Setup directories and file paths
    code_dir = pathlib.Path(__file__).resolve().parent.parent
    video_folder = code_dir / video_folder
    video_path = video_folder / video_file
    annotated_video_path = video_folder / f"annotated_{video_file}"
    saved_poses_filepath = video_folder / saved_poses_file

    # Load camera calibration data
    camera_params_dir = code_dir / "camera_params/brio_webcam"
    intrinsic_matrix = load_yaml_matrix(
        camera_params_dir / intrinsic_calibration_filename, "camera_matrix"
    )

    # Default poses and bounding box
    cur_estimated_pose = np.eye(4)
    cur_estimated_pose[:3, 3] = [-0.0325, 0.0325, 0.0325]
    cur_target_pose = np.eye(4)
    cur_target_pose[:3, 3] = [0.0, 0.0, 0.1]
    bbox = np.array([[-0.0325, -0.0325, -0.0325], [0.0325, 0.0325, 0.0325]])

    # Load video and setup video writer
    video_cap = cv2.VideoCapture(video_path.as_posix())
    num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(
        annotated_video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )

    # Load saved poses
    with open(saved_poses_filepath.as_posix(), "rb") as file:
        saved_poses = pickle.load(file)
    estimated_poses = saved_poses["estimated_poses"][:-1]  # Last pose is not used
    target_poses = saved_poses["target_poses"][:-1]

    assert (
        num_frames == len(estimated_poses) == len(target_poses)
    ), f"Mismatch in number of frames {num_frames} and recorded poses {len(estimated_poses)}"

    print("Processing the video ...")
    for cam_T_object, cam_T_target in tqdm(
        zip(estimated_poses, target_poses),
        total=len(estimated_poses),
        desc="Processing frames",
        unit="frame",
    ):
        ret, frame = video_cap.read()
        annotate_and_save_single_frame(
            frame, cam_T_object, cam_T_target, video_writer, intrinsic_matrix, bbox
        )

    # Cleanup resources
    video_cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
