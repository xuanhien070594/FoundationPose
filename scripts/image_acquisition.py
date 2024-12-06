"""This module is responsible for acquiring images from the webcam only."""

import time
import datetime
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import click
import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class WebCameraSettings:
    camera_alias: str
    width: int = 1920
    height: int = 1080
    frame_rate: int = 30


@dataclass
class RealSenseCameraSettings:
    """Settings for the RealSense camera."""

    rgb_width: int = 640
    rgb_height: int = 480
    rgb_color_format: rs.format = rs.format.bgr8
    rgb_frame_rate: int = 30  # fps

    depth_width: int = 640
    depth_height: int = 480
    depth_color_format: rs.format = rs.format.z16
    depth_frame_rate: int = 30  # fps


class ImageAcquisition(ABC):
    def __init__(self, settings: Union[WebCameraSettings, RealSenseCameraSettings]):
        self.settings = settings

    @abstractmethod
    def set_camera_settings(self) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def capture_image(self):
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def close_connection(self):
        raise NotImplemented("Method not implemented")


class WebcamImageAcquisition(ImageAcquisition):
    # (key, value) is (cam_id, full_cam_name)
    CAMERA_ALIAS_NAME_MAPPING = {"brio": "Logitech BRIO"}

    def __init__(self, settings: WebCameraSettings):
        super().__init__(settings)

        if self.settings.camera_alias not in self.CAMERA_ALIAS_NAME_MAPPING.keys():
            raise ValueError(
                f"Invalid camera alias! Valid camera alias are: {self.CAMERA_ALIAS_NAME_MAPPING.keys()}"
            )
        self.camera_name = self.CAMERA_ALIAS_NAME_MAPPING[self.settings.camera_alias]
        self.camera_index = self.find_camera_by_name(self.camera_name)

        if self.camera_index == -1:
            raise RuntimeError(
                f"No camera with id {self.camera_index} and name {self.camera_name}"
            )

        # retrieve camera handler and configure settings
        self.camera = cv2.VideoCapture(self.camera_index)
        self.set_camera_settings()

    def set_camera_settings(self) -> None:
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def capture_image(self) -> np.ndarray:
        ret, img = self.camera.read()
        if ret is None:
            print("Failed to capture image")
        return img

    def close_connection(self) -> None:
        self.camera.release()

    @staticmethod
    def get_camera_name_with_device_index(index):
        """
        Get the name of the camera at the specified index using v4l2-ctl.

        Args:
            index (int): The index of the /dev/video device (e.g., 0 for /dev/video0).

        Returns:
            str: The name of the camera, or None if not found.
        """
        device = f"/dev/video{index}"
        try:
            output = subprocess.check_output(
                ["v4l2-ctl", "--device", device, "--info"], stderr=subprocess.DEVNULL
            )
            output = output.decode("utf-8")
            for line in output.splitlines():
                if "Card type" in line:
                    return line.split(":")[1].strip()
        except subprocess.CalledProcessError:
            return None
        return None

    @staticmethod
    def find_camera_by_name(desired_name: str):
        """
        Find the index of a camera device by its name.

        Args:
            desired_name (str): The name of the desired camera.

        Returns:
            int: The index of the matching camera, or -1 if not found.
        """
        for index in range(10):  # Test up to 10 video devices (adjust as needed)
            name = WebcamImageAcquisition.get_camera_name_with_device_index(index)
            if name is not None and desired_name in name:
                return index
        return -1


class RealsenseImageAcquisition(ImageAcquisition):
    def __init__(self, settings: RealSenseCameraSettings):
        super().__init__(settings)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.set_camera_settings()
        self.pipeline.start(self.config)

    def set_camera_settings(self) -> None:
        self.config.enable_stream(
            rs.stream.color,
            self.settings.rgb_width,
            self.settings.rgb_height,
            self.settings.rgb_color_format,
            self.settings.rgb_frame_rate,
        )

    def capture_image(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def close_connection(self) -> None:
        self.pipeline.stop()


def create_folder_if_not_exists(folder_path: Path) -> None:
    """
    Creates a folder if it does not already exist.

    This function checks if the specified folder path exists. If it doesn't, it creates
    the folder along with any necessary parent directories. If the folder already exists,
    it simply prints a message indicating so.

    Args:
        folder_path (Path): The path of the folder to create.

    Returns:
        None

    Prints:
        A message indicating whether the folder was created or already existed.

    Raises:
        OSError: If there's an error creating the directory (e.g., insufficient permissions).
    """
    # Check if the path exists and is a directory
    if not folder_path.exists():
        # Create the directory (including any necessary parent directories)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder created: {folder_path.as_posix()}")
    else:
        print(f"Folder already exists: {folder_path.as_posix()}")


@click.command()
@click.option("--camera_type", default="webcam", help="The type of camera to use.")
@click.option(
    "--webcam_alias", default="brio", help="The alias (short name) of camera to use."
)
@click.option("--image_folder", default=None, help="The folder to save the images to.")
@click.option(
    "--show_img", is_flag=True, help="Whether to display the image in a window."
)
def main(camera_type: str, webcam_alias: str, image_folder: str, show_img: bool):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_folder_path = Path(image_folder) / f"{camera_type}_{timestamp}"
    create_folder_if_not_exists(image_folder_path)

    img_count = 0
    while True:
        if camera_type == "webcam":
            camera = WebcamImageAcquisition(
                WebCameraSettings(camera_alias=webcam_alias, frame_rate=1)
            )
        elif camera_type == "realsense":
            camera = RealsenseImageAcquisition(
                RealSenseCameraSettings(rgb_width=848, rgb_height=480)
            )  # TODO: Configure settings from CLI
        else:
            raise ValueError("Invalid camera type")
        time.sleep(
            1
        )  # Important: Allow time for the camera to warm up, if not returned images will be black
        img = camera.capture_image()
        print(f"Successfully captured image\n")

        if show_img:
            cv2.imshow(f"{camera_type}", img)

        save_img = input("Save image? (y/n):\n ")

        if save_img == "y":
            img_path = image_folder_path / f"img_{img_count}.png"
            cv2.imwrite(str(img_path), img)
            print(f"Image saved to {img_path}\n")
        continue_acquire_img = input("Capture more images? (y/n): ")

        if continue_acquire_img != "y":
            break

        img_count += 1

        # When everything done, release the capture
        camera.close_connection()


if __name__ == "__main__":
    main()
