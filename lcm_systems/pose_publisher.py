import lcm
import numpy as np
from lcm_types.lcmt_object_state import lcmt_object_state


class CubePoseLcmPublisher:
    def __init__(self):
        self.cube_pose_lcm_channel = "CUBE_STATE"
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
        pose_msg.position = np.concatenate(
            [
                cube_pose.orientation[[3]],
                cube_pose.orientation[:3],
                cube_pose.position,
            ]
        ).tolist()
        pose_msg.velocity = np.zeros(pose_msg.num_velocities).tolist()
        self.lc.publish(self.cube_pose_lcm_channel, pose_msg.encode())
