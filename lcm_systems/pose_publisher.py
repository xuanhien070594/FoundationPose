import lcm
import numpy as np
from foundationpose.lcm_systems.lcm_types.lcmt_object_state import lcmt_object_state
from scipy.spatial.transform import Rotation


class CubePoseLcmPublisher:
    def __init__(self, lcm_channel: str = "CUBE_STATE"):
        self.cube_pose_lcm_channel = lcm_channel
        self.lc = lcm.LCM()

    def pub_pose(self, cube_pose, timestamp):
        pose_msg = lcmt_object_state()
        pose_msg.utime = timestamp
        pose_msg.num_positions = 7
        pose_msg.num_velocities = 6
        pose_msg.position_names = [
            "cube_qw",
            "cube_qx",
            "cube_qy",
            "cube_qz",
            "cube_x",
            "cube_y",
            "cube_z",
        ]
        pose_msg.velocity_names = [
            "cube_wx",
            "cube_wy",
            "cube_wz",
            "cube_vx",
            "cube_vy",
            "cube_vz",
        ]
        cube_pos = cube_pose[:3, 3]
        cube_quat = Rotation.from_matrix(cube_pose[:3, :3]).as_quat(scalar_first=True)
        pose_msg.position = np.concatenate([cube_quat, cube_pos]).tolist()
        pose_msg.velocity = np.zeros(pose_msg.num_velocities).tolist()
        self.lc.publish(self.cube_pose_lcm_channel, pose_msg.encode())
