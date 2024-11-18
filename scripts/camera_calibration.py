"""This script is used to calibrate intrinsic and extrinsic parameters of a camera using a chessboard pattern."""

from random import choices

from charuco_board_handler import CharucoBoardHandler
import argparse
import numpy as np
import cv2
import yaml
import os
import glob
import click


BOARD_SIZE_X = 7
BOARD_SIZE_Y = 5
BOARD_SQUARE_SIZE = 0.04
BOARD_MARKER_SIZE = 0.03
BOARD_HEIGHT = 0.0055


def rodrigues_to_matrix(rvec):
    """Convert Rodrigues vector to homogeneous transformation matrix

    Args:
        rvec (array-like):  Rotation in Rodrigues format as returned by OpenCV.

    Returns:
        quaternion (array-like):  Given rotation as a 4x4 homogeneous
        transformation matrix.
    """
    rvec = np.asarray(rvec)

    # convert the Rodrigues vector to a quaternion
    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3], _ = cv2.Rodrigues(rvec)

    return rotation_matrix


def calibrate_intrinsic_parameters(calibration_data, calibration_results_file):
    """Calibrate intrinsic parameters of the camera given different images
    taken for the Charuco board from different views, the resulting parameters
    are saved to the provided filename.

    Args:
        calibration_data (str):  directory of the stored images of the
        Charuco board.
        calibration_results_file (str):  filepath that will be used to write
        the calibration results in.
    """
    handler = CharucoBoardHandler(
        BOARD_SIZE_X, BOARD_SIZE_Y, BOARD_SQUARE_SIZE, BOARD_MARKER_SIZE
    )

    pattern = os.path.join(calibration_data, "*.png")
    files = glob.glob(pattern)
    camera_matrix, dist_coeffs, error = handler.calibrate(files, visualize=True)
    camera_info = dict()
    camera_info["camera_matrix"] = dict()
    camera_info["camera_matrix"]["rows"] = 3
    camera_info["camera_matrix"]["cols"] = 3
    camera_info["camera_matrix"]["data"] = camera_matrix.flatten().tolist()
    camera_info["distortion_coefficients"] = dict()
    camera_info["distortion_coefficients"]["rows"] = 1
    camera_info["distortion_coefficients"]["cols"] = 5
    camera_info["distortion_coefficients"]["data"] = dist_coeffs.flatten().tolist()

    with open(calibration_results_file, "w") as outfile:
        yaml.dump(
            camera_info,
            outfile,
            default_flow_style=False,
        )
    return camera_matrix, dist_coeffs


def calibrate_extrinsic_parameters(
    intrinsics_calibration_filename: str,
    charuco_centralized_image_filename: str,
    extrinsic_calibration_filename: str,
    impose_cube=True,
):
    """Calibrate extrinsic parameters of the camera given one image taken for
    the Charuco board centered at (0, 0, 0) the resulting parameters are
    saved to the provided filename and a virtual cube is imposed on the
    board for verification.

    Args:
        intrinsics_calibration_filename (str):  filepath that will be used to read
        the intrinsic calibration results.
        charuco_centralized_image_filename (str): filename of the image
        taken for the Charuco board centered at (0, 0, 0).
        extrinsic_calibration_filename (str):  filepath that will be used
        to write the extrinsic calibration results in.
        impose_cube (bool): boolean whether to output a virtual cube
        imposed on the first square of the board or not.
    """
    with open(intrinsics_calibration_filename) as file:
        calibration_data = yaml.safe_load(file)

    def config_matrix(data):
        return np.array(data["data"]).reshape(data["rows"], data["cols"])

    camera_matrix = config_matrix(calibration_data["camera_matrix"])
    dist_coeffs = config_matrix(calibration_data["distortion_coefficients"])

    handler = CharucoBoardHandler(
        BOARD_SIZE_X,
        BOARD_SIZE_Y,
        BOARD_SQUARE_SIZE,
        BOARD_MARKER_SIZE,
        camera_matrix,
        dist_coeffs,
    )

    rvec, tvec = handler.detect_board_in_image(
        charuco_centralized_image_filename, visualize=False
    )
    # rotation around the y-axis (pattern coordinate system)
    yrot = np.array([0, 1, 0]) * np.pi
    yMat = cv2.Rodrigues(yrot)[0]

    projection_matrix = rodrigues_to_matrix(rvec)
    projection_matrix[3, 3] = 1

    Tvec = np.array([0.0, 0.0, BOARD_HEIGHT], dtype="float32").reshape((3, 1))
    tvec += projection_matrix[:3, :3] @ Tvec
    projection_matrix[0:3, 3] = tvec[:, 0]

    projection_matrix[:3, :3] = projection_matrix[:3, :3] @ yMat
    rvec, _ = cv2.Rodrigues(projection_matrix[:3, :3])

    calibration_data["tf_world_to_camera"] = dict()
    calibration_data["tf_world_to_camera"]["rows"] = 4
    calibration_data["tf_world_to_camera"]["cols"] = 4
    calibration_data["tf_world_to_camera"][
        "data"
    ] = projection_matrix.flatten().tolist()

    with open(extrinsic_calibration_filename, "w") as outfile:
        yaml.dump(
            calibration_data,
            outfile,
            default_flow_style=False,
        )

    if impose_cube:
        new_object_points = (
            np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [-1, 1, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [-1, 0, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float32,
            )
        ) * BOARD_SQUARE_SIZE

        # cube
        point_pairs = (
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )

        img = cv2.imread(charuco_centralized_image_filename)
        imgpoints, _ = cv2.projectPoints(
            new_object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )

        for p1, p2 in point_pairs:
            cv2.line(
                img,
                tuple(map(int, tuple(imgpoints[p1, 0]))),
                tuple(map(int, tuple(imgpoints[p2, 0]))),
                [200, 200, 0],
                thickness=2,
            )

        # cv2.imshow("Imposed Cube", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("imposed_cube.png", img)


@click.command
@click.option(
    "--action",
    type=click.Choice(
        [
            "intrinsic_calibration",
            "extrinsic_calibration",
        ]
    ),
    help="""Action that is executed.""",
)
@click.option(
    "--intrinsic_calibration_filename",
    type=str,
    help="""Filename used for saving intrinsic calibration
                        data or loading it""",
)
@click.option(
    "--calibration_data",
    type=str,
    help="""Path to the calibration data directory .""",
)
@click.option(
    "--extrinsic_calibration_filename",
    type=str,
    help="""Filename used for saving intrinsic calibration
                         data.""",
)
@click.option(
    "--image_view_filename",
    type=str,
    help="""Image with charuco centralized at the (0, 0, 0)
                        position.""",
)
def main(
    action,
    intrinsic_calibration_filename,
    calibration_data,
    extrinsic_calibration_filename,
    image_view_filename,
):
    if action == "intrinsic_calibration":
        if not intrinsic_calibration_filename:
            raise RuntimeError("intrinsic_calibration_filename not specified.")
        if not calibration_data:
            raise RuntimeError("calibration_data not specified.")
        calibrate_intrinsic_parameters(calibration_data, intrinsic_calibration_filename)
    elif action == "extrinsic_calibration":
        if not intrinsic_calibration_filename:
            raise RuntimeError("intrinsic_calibration_filename not specified.")
        if not extrinsic_calibration_filename:
            raise RuntimeError("extrinsic_calibration_filename not specified.")
        if not image_view_filename:
            raise RuntimeError("image_view_filename not specified.")
        calibrate_extrinsic_parameters(
            intrinsic_calibration_filename,
            image_view_filename,
            extrinsic_calibration_filename,
            impose_cube=True,
        )
    else:
        raise ValueError("Unknown action.")


if __name__ == "__main__":
    main()
