from collections import OrderedDict
from typing import List, Optional, Dict, Any

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomSceneEnv # Using CustomSceneEnv as a base
from mani_skill2_real2sim.envs.custom_scenes.grasp_single_in_scene import GraspSingleInSceneEnv # For some methods
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnInSceneEnv # For evaluation logic


@register_env("PutSpoonOnClothInCokeScene-v0", max_episode_steps=120) # Increased steps for a more complex task
class PutSpoonOnClothInCokeSceneEnv(CustomSceneEnv):
    spoon_obj: sapien.Actor
    cloth_obj: sapien.Actor
    coke_obj: sapien.Actor

    DEFAULT_ASSET_ROOT = ASSET_DIR / "custom"
    DEFAULT_SCENE_ROOT = ASSET_DIR / "hab2_bench_assets" / "stages"
    DEFAULT_MODEL_JSON = ASSET_DIR / "custom" / "info_put_spoon_on_cloth_task_v0.json"

    def __init__(
        self,
        robot_uid="google_robot_static", # RT-1 typically uses google_robot
        robot_init_qpos_noise_scale=0.02,
        control_freq=3, # Default for google_robot RT-1 tasks
        sim_freq=513,
        scene_name="google_pick_coke_can_1_v4", # Coke scene
        obj_init_options: Optional[Dict] = None, # For coke can
        spoon_init_options: Optional[Dict] = None,
        cloth_init_options: Optional[Dict] = None,
        success_require_src_completely_on_target=False,
        **kwargs,
    ):
        # model_db_files = [
        #     ASSET_DIR / "custom" / "info_pick_custom_v0.json",
        #     ASSET_DIR / "custom" / "info_bridge_custom_v0.json"
        # ]
        self.coke_model_id = "opened_coke_can"
        self.spoon_model_id = "bridge_spoon_generated_modified"
        self.cloth_model_id = "table_cloth_generated_shorter" # from PutSpoonOnTableClothInScene

        self.obj_init_options = obj_init_options if obj_init_options is not None else {}
        self.spoon_init_options = spoon_init_options if spoon_init_options is not None else {}
        self.cloth_init_options = cloth_init_options if cloth_init_options is not None else {}

        # Ensure these models are available via model_db by listing them
        # The actual model_db is loaded by the base CustomSceneEnv based on its DEFAULT_MODEL_JSON
        # or a provided model_db_override. We assume these IDs are in available JSONs.
        # If not, one might need to pass a custom model_db_path or model_db_override.
        self.model_ids = [self.coke_model_id, self.spoon_model_id, self.cloth_model_id]


        self._success_require_src_completely_on_target = success_require_src_completely_on_target
        self.consecutive_grasp_on_spoon = 0 # For evaluate

        super().__init__(
            robot_uids=robot_uid,
            robot_init_qpos_noise_scale=robot_init_qpos_noise_scale,
            control_freq=control_freq,
            sim_freq=sim_freq,
            scene_name=scene_name,
            **kwargs,
        )
        # Override default model_ids from base if it uses a single model_id concept
        # self.model_ids is used by _load_model_db if DEFAULT_MODEL_JSON needs filtering.


    def _load_model_db(self):
        # Ensure all necessary models are loaded into self.model_db
        # This might involve merging from multiple JSONs or ensuring one JSON has all.
        # For simplicity, assuming CustomSceneEnv loads a comprehensive JSON or
        # we list all required model_ids for it to find.
        # info_pick_custom_v0.json has opened_coke_can
        # info_bridge_custom_v0.json has bridge_spoon_generated_modified, table_cloth_generated_shorter
        # We'll rely on the default behavior of CustomSceneEnv or provide a merged JSON if needed.
        # For now, we assume the default JSON or a specified one in a script can find these.
        # If using specific JSONs, model_db_path might need to be set.
        # Let's assume the default asset_root and model_db logic will find them if specified in self.model_ids.
        # CustomSceneEnv by default loads info_pick_custom_v0.json and info_bridge_custom_v0.json
        # if they are part of its default search or if its model_ids includes items from them.
        # The base class `CustomSceneEnv` has logic for `_get_default_model_json_path`
        # which can be a list.
        pass # Rely on base class logic, ensuring self.model_ids is correctly populated before super().__init__

    def _load_actors(self):
        self._load_arena_helper()

        # Load Coke Can
        self.coke_obj = self._build_actor_helper(
            model_id=self.coke_model_id,
            scene=self._scene,
            scale=self.model_db[self.coke_model_id].get("scale", 1.0), # Get scale from model_db if defined
            density=self.model_db[self.coke_model_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.coke_obj.name = self.coke_model_id
        self.coke_obj.set_damping(0.1, 0.1)


        # Load Spoon (Source Object for the put-on task)
        self.spoon_obj = self._build_actor_helper(
            model_id=self.spoon_model_id,
            scene=self._scene,
            scale=self.model_db[self.spoon_model_id].get("scale", 1.0),
            density=self.model_db[self.spoon_model_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.spoon_obj.name = self.spoon_model_id
        self.spoon_obj.set_damping(0.1, 0.1)


        # Load Cloth (Target Receptacle for the put-on task)
        self.cloth_obj = self._build_actor_helper(
            model_id=self.cloth_model_id,
            scene=self._scene,
            scale=self.model_db[self.cloth_model_id].get("scale", 1.0),
            density=self.model_db[self.cloth_model_id].get("density", 200), # Cloth might be lighter
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction + 0.2, # Cloth might have higher friction
                dynamic_friction=self.obj_dynamic_friction + 0.2,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.cloth_obj.name = self.cloth_model_id
        self.cloth_obj.set_damping(0.1, 0.1)
        
        # For evaluation logic compatibility with PutOnInSceneEnv
        self.episode_source_obj = self.spoon_obj
        self.episode_target_obj = self.cloth_obj
        self.episode_objs = [self.spoon_obj, self.cloth_obj, self.coke_obj] # All relevant objects


    def _initialize_actors(self):
        # Move the robot far away to avoid collision during initial object placement
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # 1. Place Coke Can (similar to GraspSingleInSceneEnv)
        coke_init_xy_range_low = self.obj_init_options.get("init_xy_range_low", [-0.35, -0.02])
        coke_init_xy_range_high = self.obj_init_options.get("init_xy_range_high", [-0.12, 0.42])
        coke_init_xy = self._episode_rng.uniform(coke_init_xy_range_low, coke_init_xy_range_high, [2])
        
        coke_init_z = self.obj_init_options.get("init_z", self.scene_table_height + 0.2) # Fall height
        coke_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0]) # Default upright

        # Random orientation for coke can (optional, from GraspSingleCustomOrientationInSceneEnv)
        if self.obj_init_options.get("randomize_orientation", False):
            orientation_key = self._episode_rng.choice(["upright", "laid_vertically", "lr_switch"])
            orientations_dict = { # from GraspSingleCustomOrientationInSceneEnv
                "upright": euler2quat(np.pi / 2, 0, 0),
                "laid_vertically": euler2quat(0, 0, np.pi / 2),
                "lr_switch": euler2quat(0, 0, np.pi),
            }
            coke_init_rot_quat = orientations_dict[orientation_key]


        self.coke_obj.set_pose(sapien.Pose(np.hstack([coke_init_xy, coke_init_z]), coke_init_rot_quat))
        self.coke_obj.lock_motion(0, 0, 0, 1, 1, 0) # Lock x,y rotation
        self._settle(0.5)
        self.coke_obj.lock_motion(0, 0, 0, 0, 0, 0)
        self.coke_obj.set_pose(self.coke_obj.pose) # Wake up
        self._settle(0.5)
        self.coke_obj_height_after_settle = self.coke_obj.pose.p[2]
        self.episode_obj_xyzs_after_settle = [] # For PutOnInSceneEnv eval


        # 2. Place Cloth
        cloth_init_xy_range_low = self.cloth_init_options.get("init_xy_range_low", [-0.05, 0.15]) # Different area
        cloth_init_xy_range_high = self.cloth_init_options.get("init_xy_range_high", [0.15, 0.35])
        cloth_init_xy = self._episode_rng.uniform(cloth_init_xy_range_low, cloth_init_xy_range_high, [2])
        cloth_init_z = self.cloth_init_options.get("init_z", self.scene_table_height + 0.1) # Cloth is thin
        cloth_init_rot_quat = self.cloth_init_options.get("init_rot_quat", [1,0,0,0]) # Flat

        self.cloth_obj.set_pose(sapien.Pose(np.hstack([cloth_init_xy, cloth_init_z]), cloth_init_rot_quat))
        self.cloth_obj.lock_motion(0,0,0,1,1,0) # Lock x,y rotation for cloth as well
        self._settle(0.5)
        self.cloth_obj.lock_motion(0,0,0,0,0,0)
        self.cloth_obj.set_pose(self.cloth_obj.pose)
        self._settle(0.5)
        self.episode_target_obj_xyz_after_settle = self.cloth_obj.pose.p.copy()
        self.episode_target_obj_bbox_world = self.get_obj_bbox_world(self.cloth_obj) # For eval
        self.episode_obj_xyzs_after_settle.append(self.cloth_obj.pose.p.copy())


        # 3. Place Spoon (relative to cloth or in a defined region)
        # For simplicity, place spoon next to the cloth.
        # User wants to maintain spoon/cloth relative relationship from Bridge "put spoon on cloth"
        # PutSpoonOnTableClothInScene has xy_configs like:
        # np.array([grid_pos_1, grid_pos_2]) where grid_pos can be e.g. [-0.16-0.075, 0.00-0.075]
        # Let's place spoon at an offset from the cloth's final position
        
        spoon_offset = self.spoon_init_options.get("offset_from_cloth", np.array([0.0, -0.20])) # e.g., 20cm in -y
        spoon_init_xy = self.cloth_obj.pose.p[:2] + spoon_offset
        
        # Ensure spoon is within reasonable table bounds if offset is large
        # Table for google_pick_coke_can_1_v4 is roughly x=[-0.6, 0.6], y=[-0.6, 0.6] at z=0.8
        # Need to clip spoon_init_xy if necessary, or use a placement region:
        spoon_init_xy_range_low = self.spoon_init_options.get("init_xy_range_low", [self.cloth_obj.pose.p[0] - 0.1, self.cloth_obj.pose.p[1] - 0.25])
        spoon_init_xy_range_high = self.spoon_init_options.get("init_xy_range_high", [self.cloth_obj.pose.p[0] + 0.1, self.cloth_obj.pose.p[1] - 0.15])
        if not self.spoon_init_options.get("use_offset_from_cloth", True): # Allow independent placement
            spoon_init_xy = self._episode_rng.uniform(spoon_init_xy_range_low, spoon_init_xy_range_high, [2])

        spoon_init_z = self.spoon_init_options.get("init_z", self.scene_table_height + 0.1)
        spoon_init_rot_quat = self.spoon_init_options.get("init_rot_quat", euler2quat(0,0, np.pi/self._episode_rng.uniform(1.5, 2.5))) # Random Z rotation for spoon

        self.spoon_obj.set_pose(sapien.Pose(np.hstack([spoon_init_xy, spoon_init_z]), spoon_init_rot_quat))
        self.spoon_obj.lock_motion(0,0,0,1,1,0)
        self._settle(0.5)
        self.spoon_obj.lock_motion(0,0,0,0,0,0)
        self.spoon_obj.set_pose(self.spoon_obj.pose)
        self._settle(0.5)
        self.episode_source_obj_xyz_after_settle = self.spoon_obj.pose.p.copy()
        self.episode_source_obj_bbox_world = self.get_obj_bbox_world(self.spoon_obj) # For eval
        self.episode_obj_xyzs_after_settle.insert(0, self.spoon_obj.pose.p.copy()) # Spoon as first obj
        self.episode_obj_xyzs_after_settle.append(self.coke_obj.pose.p.copy())

        # Store initial poses for evaluation if needed by PutOnInSceneEnv's criteria
        self.episode_source_obj_initial_pose = self.spoon_obj.pose
        self.episode_target_obj_initial_pose = self.cloth_obj.pose
        
    def get_obj_bbox_world(self, obj_actor: sapien.Actor):
        # Helper to get world-aligned bbox for evaluation
        # This is a simplified version. True world AABB might need to consider rotation.
        # PutOnInSceneEnv uses self.model_db[obj.name]['bbox'] and obj.scale
        obj_model_info = self.model_db[obj_actor.name]
        if "bbox" in obj_model_info:
            bbox_min = np.array(obj_model_info["bbox"]["min"])
            bbox_max = np.array(obj_model_info["bbox"]["max"])
            # Assuming scale is uniformly applied or already baked into model_db scale for bbox
            scale = obj_actor.scale if hasattr(obj_actor, 'scale') else getattr(self, 'model_scale', 1.0)
            if isinstance(scale, float):
                scale = np.array([scale,scale,scale])
            if obj_actor.name == self.spoon_model_id:
                 scale = self.model_db[self.spoon_model_id].get("scale", np.array([1,1,1]))
            elif obj_actor.name == self.cloth_model_id:
                 scale = self.model_db[self.cloth_model_id].get("scale", np.array([1,1,1]))
            elif obj_actor.name == self.coke_model_id:
                 scale = self.model_db[self.coke_model_id].get("scale", np.array([1,1,1]))

            return (bbox_max - bbox_min) * scale # size of bbox
        return np.array([0.05, 0.05, 0.05]) # Default fallback

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                spoon_pose=vectorize_pose(self.spoon_obj.pose),
                cloth_pose=vectorize_pose(self.cloth_obj.pose),
                coke_pose=vectorize_pose(self.coke_obj.pose), # Coke pose for context
                tcp_to_spoon_pos=self.spoon_obj.pose.p - self.tcp.pose.p,
            )
        return obs

    def _initialize_episode_stats(self):
        # From PutOnInSceneEnv
        self.episode_stats = OrderedDict(
            moved_correct_obj=False, # Spoon moved
            moved_wrong_obj=False, # Cloth or Coke moved significantly
            is_src_obj_grasped=False, # Spoon grasped
            consecutive_grasp=False, # Consistent grasp on spoon
            src_on_target=False, # Spoon on cloth
        )
        # Additional stats if needed
        self.episode_stats["coke_can_perturbed"] = False


    def evaluate(self, **kwargs):
        # Combine evaluation logic from PutOnInSceneEnv for spoon on cloth
        # and optionally check if coke can was disturbed too much.

        # Spoon on Cloth part (adapted from PutOnInSceneEnv.evaluate)
        source_obj_pose = self.spoon_obj.pose # self.episode_source_obj.pose
        target_obj_pose = self.cloth_obj.pose # self.episode_target_obj.pose

        # whether moved the correct object (spoon)
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2] - source_obj_pose.p[:2]
        )
        
        # Check if other objects (cloth, coke) were moved too much
        cloth_xy_move_dist = np.linalg.norm(
            self.episode_target_obj_xyz_after_settle[:2] - target_obj_pose.p[:2]
        )
        coke_xy_move_dist = np.linalg.norm(
            self.coke_obj_height_after_settle[:2] - self.coke_obj.pose.p[:2] # Note: using height_after_settle for xyz
        )
        
        # Correct object moved if spoon moved, and cloth/coke didn't move more than spoon
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and \
                            (cloth_xy_move_dist < source_obj_xy_move_dist or cloth_xy_move_dist < 0.02) and \
                            (coke_xy_move_dist < source_obj_xy_move_dist or coke_xy_move_dist < 0.03)

        # Wrong object moved if cloth or coke moved significantly AND more than spoon
        moved_wrong_obj = ((cloth_xy_move_dist > 0.03 and cloth_xy_move_dist > source_obj_xy_move_dist) or \
                           (coke_xy_move_dist > 0.03 and coke_xy_move_dist > source_obj_xy_move_dist))


        # whether the source object (spoon) is grasped
        is_src_obj_grasped = self.agent.check_grasp(self.spoon_obj, max_angle=65) # Check grasp on spoon
        if is_src_obj_grasped:
            self.consecutive_grasp_on_spoon += 1
        else:
            self.consecutive_grasp_on_spoon = 0
        consecutive_grasp = self.consecutive_grasp_on_spoon >= 3 # Reduced from 5 for quicker check

        # whether the source object (spoon) is on the target object (cloth)
        tgt_obj_half_length_bbox = self.episode_target_obj_bbox_world / 2
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        
        # Looser xy_flag: center of spoon over any part of cloth's bbox
        xy_flag = (np.abs(offset[0]) <= tgt_obj_half_length_bbox[0] + 0.01) and \
                  (np.abs(offset[1]) <= tgt_obj_half_length_bbox[1] + 0.01)

        # z_flag: spoon is above cloth and not too high
        # Ensure spoon's bottom is above cloth's top, within a margin
        # Spoon's lowest point: pos_src[2] - src_obj_half_length_bbox[2]
        # Cloth's highest point: pos_tgt[2] + tgt_obj_half_length_bbox[2]
        z_contact_lower_bound = pos_tgt[2] + tgt_obj_half_length_bbox[2] - 0.01 # cloth top - epsilon
        z_contact_upper_bound = pos_tgt[2] + tgt_obj_half_length_bbox[2] + src_obj_half_length_bbox[2] + 0.03 # cloth top + spoon height + margin

        z_flag = (pos_src[2] > z_contact_lower_bound) and \
                 (pos_src[2] < z_contact_upper_bound)


        src_on_target_bbox = xy_flag and z_flag
        src_on_target_contact = False

        if src_on_target_bbox and self._success_require_src_completely_on_target: # More stringent contact check
            contacts = self._scene.get_contacts()
            src_touching_target = False
            src_touching_other_non_robot = False
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            
            for contact in contacts:
                actor_a_name = contact.actor0.name
                actor_b_name = contact.actor1.name

                is_contact_with_spoon = False
                other_actor_name = None

                if actor_a_name == self.spoon_obj.name:
                    is_contact_with_spoon = True
                    other_actor_name = actor_b_name
                elif actor_b_name == self.spoon_obj.name:
                    is_contact_with_spoon = True
                    other_actor_name = actor_a_name
                
                if is_contact_with_spoon:
                    # Check if spoon is touching cloth
                    if other_actor_name == self.cloth_obj.name:
                        src_touching_target = True
                    # Check if spoon is touching something else (not cloth, not robot, not table)
                    elif other_actor_name not in robot_link_names and \
                         other_actor_name != self.cloth_obj.name and \
                         "table" not in other_actor_name and \
                         "Arena" not in other_actor_name and \
                         other_actor_name != self.coke_obj.name: # Allow contact with coke for now
                        # Sum impulses to check if contact is significant
                        contact_impulse_sum = np.sum([np.linalg.norm(point.impulse) for point in contact.points])
                        if contact_impulse_sum > 1e-4:
                            src_touching_other_non_robot = True
                            break 
            if src_touching_target and not src_touching_other_non_robot:
                src_on_target_contact = True
            src_on_target = src_on_target_contact
        else:
            src_on_target = src_on_target_bbox # Use bbox based if not requiring complete contact

        success = src_on_target and moved_correct_obj # and not moved_wrong_obj (optional strictness)

        # Update episode stats
        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        self.episode_stats["consecutive_grasp"] = self.episode_stats["consecutive_grasp"] or consecutive_grasp
        self.episode_stats["coke_can_perturbed"] = coke_xy_move_dist > 0.05 # Example threshold for perturbation

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target_bbox=src_on_target_bbox,
            src_on_target_contact=src_on_target_contact,
            src_on_target=src_on_target,
            coke_can_perturbed=self.episode_stats["coke_can_perturbed"],
            episode_stats=self.episode_stats,
            success=success,
        )

    def get_language_instruction(self, **kwargs):
        # Based on user request and PutSpoonOnTableClothInScene
        return "put the spoon on the cloth"

    def _setup_prepackaged_env_init_config(self):
        # From GraspSingleOpenedCokeCanInSceneEnv
        ret = {}
        ret["robot_uid"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret["control_mode"] = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        # Visual matching specific, can be kept or removed if not primary focus
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_coke_can_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        return ret

    def _additional_prepackaged_config_reset(self, options: dict) -> bool:
        # From GraspSingleOpenedCokeCanInSceneEnv
        # Sets robot initial pose and URDF variations for visual matching
        options.setdefault("robot_init_options", {})
        options["robot_init_options"].update({
            "init_xy": [0.35, 0.20],
            "init_rot_quat": [0, 0, 0, 1],
        })
        
        # URDF variation logic (can be kept for visual matching consistency)
        # new_urdf_version = self._episode_rng.choice(
        #     [
        #         None, # Default URDF
        #         "recolor_tabletop_visual_matching_1",
        #         "recolor_tabletop_visual_matching_2",
        #         "recolor_cabinet_visual_matching_1",
        #     ]
        # )
        # urdf_version_changed = False
        # if new_urdf_version != self.urdf_version:
        #     self.urdf_version = new_urdf_version
        #     # self._configure_agent() # This would reload the robot
        #     urdf_version_changed = True
        # return urdf_version_changed
        return False # Simpler: don't change URDF for now unless explicitly requested

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if options is None:
            options = {}
        
        # Allow overriding init options for each object via options dict
        self.obj_init_options = options.get("coke_init_options", self.obj_init_options)
        self.spoon_init_options = options.get("spoon_init_options", self.spoon_init_options)
        self.cloth_init_options = options.get("cloth_init_options", self.cloth_init_options)

        self.consecutive_grasp_on_spoon = 0
        self._initialize_episode_stats()

        # reconfigure = options.get("reconfigure", False)
        # if self.prepackaged_config: # prepackaged_config is a CustomSceneEnv attribute
        #     _reconfigure = self._additional_prepackaged_config_reset(options)
        #     reconfigure = reconfigure or _reconfigure
        # options["reconfigure"] = reconfigure
        
        # We need to manage the model_ids for the base class correctly.
        # CustomSceneEnv uses self.model_id for single object. We have multiple.
        # We've set self.model_ids in __init__.
        # The `_set_model` in some base classes might pick one. We load all three.

        obs, info = super().reset(seed=seed, options=options)

        info.update({
            "coke_model_id": self.coke_model_id,
            "spoon_model_id": self.spoon_model_id,
            "cloth_model_id": self.cloth_model_id,
            # Poses relative to robot base might be useful
            "spoon_init_pose_wrt_robot_base": self.agent.robot.pose.inv() * self.spoon_obj.pose,
            "cloth_init_pose_wrt_robot_base": self.agent.robot.pose.inv() * self.cloth_obj.pose,
            "coke_init_pose_wrt_robot_base": self.agent.robot.pose.inv() * self.coke_obj.pose,
        })
        return obs, info