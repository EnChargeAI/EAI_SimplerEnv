from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


@dataclass
class ManiSkill2Config:
    # Required parameters
    env_name: str  # required=True in argparse
    task_name: str
    episode_id: int = 0

    # Policy settings
    policy_model: str = "rt1"
    policy_setup: str = "google_robot"
    ckpt_path: Optional[str] = None

    # Environment settings
    scene_name: str = "google_pick_coke_can_1_v4"
    enable_raytracing: bool = False
    robot: str = "google_robot_static"
    obs_camera_name: Optional[str] = None
    action_scale: float = 1.0

    # Control settings
    control_freq: int = 3
    sim_freq: int = 513
    max_episode_steps: int = 80
    rgb_overlay_path: Optional[str] = None

    # Robot initialization ranges
    robot_variation_mode: str = "xy"  # choices: ["xy", "episode_xy"]
    robot_episode_range: List[int] = field(default_factory=lambda: [0, 60])
    robot_init_x_range: List[float] = field(default_factory=lambda: [0.35, 0.35, 1])  # start, end, len
    robot_init_y_range: List[float] = field(default_factory=lambda: [0.20, 0.20, 1])
    robot_init_rot_quat_center: List[float] = field(default_factory=lambda: [1, 0, 0, 0])
    robot_init_rot_rpy_range: List[float] = field(default_factory=lambda: [0, 0, 1, 0, 0, 1, 0, 0, 1])

    # Object variation settings
    obj_variation_mode: str = "xy"  # choices: ["xy", "episode", "episode_xy"]
    obj_episode_range: List[int] = field(default_factory=lambda: [0, 60])
    obj_init_x_range: List[float] = field(default_factory=lambda: [-0.35, -0.12, 5])
    obj_init_y_range: List[float] = field(default_factory=lambda: [-0.02, 0.42, 5])

    # Additional settings
    additional_env_build_kwargs: Optional[Dict[str, Any]] = None
    additional_env_save_tags: Optional[str] = None
    logging_dir: str = "./results"
    tf_memory_limit: int = 3072
    octo_init_rng: int = 0

    # Computed attributes (will be set in __post_init__)
    robot_init_xs: List[float] = field(init=False)
    robot_init_ys: List[float] = field(init=False)
    robot_init_quats: List[List[float]] = field(init=False)
    obj_init_xs: Optional[List[float]] = field(init=False, default=None)
    obj_init_ys: Optional[List[float]] = field(init=False, default=None)

    def __post_init__(self):
        """argparseの後処理と同等の計算を実行"""
        # Robot pose calculations
        if self.robot_variation_mode == "xy":
            self.robot_init_xs = parse_range_tuple(self.robot_init_x_range)
            self.robot_init_ys = parse_range_tuple(self.robot_init_y_range)

        # Robot orientation calculations
        self.robot_init_quats = []
        for r in parse_range_tuple(self.robot_init_rot_rpy_range[:3]):
            for p in parse_range_tuple(self.robot_init_rot_rpy_range[3:6]):
                for y in parse_range_tuple(self.robot_init_rot_rpy_range[6:]):
                    quat = (Pose(q=euler2quat(r, p, y)) * Pose(q=self.robot_init_rot_quat_center)).q
                    self.robot_init_quats.append(quat)

        # Object position calculations
        if self.obj_variation_mode == "xy":
            self.obj_init_xs = parse_range_tuple(self.obj_init_x_range)
            self.obj_init_ys = parse_range_tuple(self.obj_init_y_range)

        # Update logging info if using different camera
        if self.obs_camera_name is not None:
            if self.additional_env_save_tags is None:
                self.additional_env_save_tags = f"obs_camera_{self.obs_camera_name}"
            else:
                self.additional_env_save_tags = self.additional_env_save_tags + f"_obs_camera_{self.obs_camera_name}"

        # Validate obj_variation_mode
        if self.obj_variation_mode not in ["xy", "episode", "episode_xy"]:
            raise ValueError(f"obj_variation_mode must be 'xy' or 'episode', got {self.obj_variation_mode}")
