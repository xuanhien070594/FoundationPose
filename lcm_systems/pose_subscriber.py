from typing import Optional

import lcm
import numpy as np
from foundationpose.lcm_systems.lcm_types.lcmt_object_state import lcmt_object_state
from scipy.spatial.transform import Rotation


class CubePoseLcmSubscriber:
    def __init__(
        self,
        lcm_obj: lcm.LCM,
        channel: str,
        timeout: float,
        queue_size: int = 1,
    ) -> None:
        self.lcm_obj = lcm_obj
        self.channel = channel
        self.timeout = timeout
        self.queue_size = queue_size
        self.subscriber = self.lcm_obj.subscribe(self.channel, self.callback)
        self.subscriber.set_queue_capacity(queue_size)
        self.cube_pos: Optional[np.ndarray] = None
        self.cube_orientation: Optional[float] = None

    def callback(self, channel: str, data) -> None:
        msg = lcmt_object_state.decode(data)
        self.cube_pos = np.array(msg.position)[4:]

        # only get rotation angle around z-axis
        self.cube_orientation = Rotation.from_quat(
            np.array(msg.position[:4]), scalar_first=True
        ).as_matrix()

    def _get_cube_pose(self) -> np.ndarray:
        cube_pose = np.eye(4)
        if self.cube_pos is None or self.cube_orientation is None:
            return None
        cube_pose[:3, :3] = self.cube_orientation
        cube_pose[:3, 3] = self.cube_pos
        return cube_pose

    def get_data(self):
        return self._get_cube_pose()
