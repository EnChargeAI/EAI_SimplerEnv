from collections import OrderedDict
import sapien.core as sapien
import numpy as np
from transforms3d.euler import euler2quat
from typing import Dict, Optional

from mani_skill2_real2sim.envs.custom_scenes.base_env import CustomSceneEnv
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name, vectorize_pose


@register_env("GoogleRobotPutSpoonOnTowel-v0", max_episode_steps=120)
class GoogleRobotPutSpoonOnTowelEnv(CustomSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
    DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets" # As per CustomSceneEnv
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def __init__(self, 
                 robot_uid="google_robot_static", 
                 obj_init_options: Optional[Dict] = None,
                 spoon_init_options: Optional[Dict] = None,
                 towel_init_options: Optional[Dict] = None,
                 *args, **kwargs):
        
        self.obj_init_options = obj_init_options if obj_init_options is not None else {}
        self.spoon_init_options = spoon_init_options if spoon_init_options is not None else {}
        self.towel_init_options = towel_init_options if towel_init_options is not None else {}

        self.spoon_model_id = "bridge_spoon_generated_modified"
        self.towel_model_id = "table_cloth_generated_shorter"
        self.model_ids = [self.spoon_model_id, self.towel_model_id]
        
        super().__init__(robot_uid=robot_uid, model_ids=self.model_ids, *args, **kwargs)

    def _load_actors(self):
        super()._load_arena_helper()

        # Load spoon
        self.spoon_obj = self._build_actor_helper(
            model_id=self.spoon_model_id,
            scene=self._scene,
            scale=self.model_db[self.spoon_model_id].get("scale", 1.0),
            density=self.model_db[self.spoon_model_id].get("density", 1000.0), # Default density if not specified
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0
            ),
            name=self.spoon_model_id
        )
        self.spoon_obj.set_damping(0.1, 0.1)

        # Load towel
        self.towel_obj = self._build_actor_helper(
            model_id=self.towel_model_id,
            scene=self._scene,
            scale=self.model_db[self.towel_model_id].get("scale", 1.0),
            density=self.model_db[self.towel_model_id].get("density", 200.0), # Default density for cloth-like objects
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0
            ),
            name=self.towel_model_id
        )
        self.towel_obj.set_damping(0.1, 0.1)
        
        self.episode_source_obj = self.spoon_obj
        self.episode_target_obj = self.towel_obj
        self.episode_objs = [self.spoon_obj, self.towel_obj]


    def get_obj_bbox_world_size(self, obj_actor: sapien.Actor):
        bbox_min = np.array(self.model_db[obj_actor.name]['bbox']['min'])
        bbox_max = np.array(self.model_db[obj_actor.name]['bbox']['max'])
        scale = self.model_db[obj_actor.name].get("scale", 1.0)
        if isinstance(scale, (float, int)):
            scale = np.array([scale, scale, scale])
        return (bbox_max - bbox_min) * scale

    def _initialize_actors(self):
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0])) # Move robot far away

        # Initialize spoon pose
        spoon_pose = sapien.Pose(p=[0.0, 0.2, self.scene_table_height + 0.05], q=euler2quat(0, 0, 0))
        self.spoon_obj.set_pose(spoon_pose)
        self._settle(0.5)

        # Initialize towel pose
        towel_pose = sapien.Pose(p=[0.0, -0.2, self.scene_table_height + 0.01], q=euler2quat(0, 0, 0))
        # Adjust towel z position using its bounding box height
        towel_bbox_z_half_size = self.get_obj_bbox_world_size(self.towel_obj)[2] / 2
        towel_pose.set_p([towel_pose.p[0], towel_pose.p[1], self.scene_table_height + towel_bbox_z_half_size])
        self.towel_obj.set_pose(towel_pose)
        self._settle(0.5)

        # Store initial poses and bbox sizes
        self.spoon_initial_pose = self.spoon_obj.pose
        self.towel_initial_pose = self.towel_obj.pose
        self.spoon_bbox_size = self.get_obj_bbox_world_size(self.spoon_obj)
        self.towel_bbox_size = self.get_obj_bbox_world_size(self.towel_obj)


    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            spoon_moved_significantly=False,
            towel_moved_significantly=False,
            spoon_on_towel=False,
            spoon_is_flat=False,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._initialize_episode_stats()
        return obs, info

    def _get_obs_extra(self):
        obs = OrderedDict()
        if self.spoon_obj is not None: # Check if objects are loaded
            obs["spoon_pose"] = vectorize_pose(self.spoon_obj.pose)
        if self.towel_obj is not None:
            obs["towel_pose"] = vectorize_pose(self.towel_obj.pose)
        if hasattr(self, 'tcp') and self.tcp is not None and self.spoon_obj is not None: # Ensure tcp and spoon_obj exist
            obs["tcp_to_spoon_pos"] = self.spoon_obj.pose.p - self.tcp.pose.p
        return obs

    def evaluate(self, **kwargs):
        if self.spoon_obj is None or self.towel_obj is None:
            if not hasattr(self, 'episode_stats'): # Ensure stats are initialized if reset wasn't properly called
                 self._initialize_episode_stats()
            return {"success": False, **self.episode_stats}

        # Update movement stats first, as they don't depend on other flags
        if hasattr(self, 'spoon_initial_pose') and self.spoon_initial_pose is not None: # Check if initial pose is available
            self.episode_stats["spoon_moved_significantly"] = np.linalg.norm(self.spoon_obj.pose.p - self.spoon_initial_pose.p) > 0.05
        else: # Should not happen if reset is called correctly
            self.episode_stats["spoon_moved_significantly"] = False 
            
        if hasattr(self, 'towel_initial_pose') and self.towel_initial_pose is not None: # Check if initial pose is available
            self.episode_stats["towel_moved_significantly"] = np.linalg.norm(self.towel_obj.pose.p - self.towel_initial_pose.p) > 0.05
        else: # Should not happen if reset is called correctly
             self.episode_stats["towel_moved_significantly"] = False


        source_obj_pose = self.spoon_obj.pose
        target_obj_pose = self.towel_obj.pose
        
        # Ensure bbox sizes are available
        if not hasattr(self, 'towel_bbox_size') or not hasattr(self, 'spoon_bbox_size'):
            # This case should ideally not be reached if initialization is correct.
            # Fallback to re-calculating or returning error/default stats.
            # For now, let's assume they are initialized. If not, this will error out.
            # A more robust solution might re-calculate them here or ensure they exist.
            # For simplicity, we proceed assuming they exist.
            pass


        target_obj_half_length_bbox = self.towel_bbox_size / 2
        source_obj_half_length_bbox = self.spoon_bbox_size / 2

        offset = source_obj_pose.p - target_obj_pose.p
        
        xy_flag = (np.abs(offset[0]) <= target_obj_half_length_bbox[0] and
                   np.abs(offset[1]) <= target_obj_half_length_bbox[1])
        
        spoon_bottom_z = source_obj_pose.p[2] - source_obj_half_length_bbox[2]
        towel_top_z = target_obj_pose.p[2] + target_obj_half_length_bbox[2]
        z_flag = np.abs(spoon_bottom_z - towel_top_z) < 0.02

        self.episode_stats["spoon_on_towel"] = xy_flag and z_flag

        spoon_z_orientation = source_obj_pose.to_transformation_matrix()[:3, :3] @ np.array([0, 0, 1])
        is_flat = np.dot(spoon_z_orientation, np.array([0, 0, 1])) > 0.85
        self.episode_stats["spoon_is_flat"] = is_flat
        
        success = (self.episode_stats["spoon_on_towel"] and
                   self.episode_stats["spoon_is_flat"] and
                   self.episode_stats["spoon_moved_significantly"] and
                   not self.episode_stats["towel_moved_significantly"])
        
        return {"success": success, **self.episode_stats}

    def get_language_instruction(self, **kwargs):
        return "put the spoon on the towel"

if __name__ == "__main__":
    # Example of how to test the environment (optional, for quick checks)
    env = GoogleRobotPutSpoonOnTowelEnv()
    print("Environment created.")
    # Add more test logic if needed
