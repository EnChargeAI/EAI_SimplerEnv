from typing import Any, Dict, List

from . import random_envs
from ..policies.base import AiroaBasePolicy
from .config import ManiSkill2Config
from .evaluate import _run_single_evaluation
from .maniskill2_evaluator import run_maniskill2_eval_single_episode, get_robot_control_mode
import numpy as np

# グローバルカウンター
global_episode_counter = 0

def get_next_episode_id():
    global global_episode_counter
    current_id = global_episode_counter
    global_episode_counter += 1
    return current_id

def custom_run_single_evaluation_with_counter(env_policy: AiroaBasePolicy, cfg: ManiSkill2Config, ckpt_path: str, start_episode_id: int = 0) -> List[bool]:
    """各オブジェクトエピソードごとに異なるepisode_idを使用する評価関数"""
    if cfg.additional_env_build_kwargs:
        if "urdf_version" in cfg.additional_env_build_kwargs:
            kwargs_info = cfg.additional_env_build_kwargs["urdf_version"]
        else:
            kwargs_info = cfg.additional_env_build_kwargs
    else:
        kwargs_info = None

    print(f"  ▶️  Running: env={cfg.env_name}, scene={cfg.scene_name}, kwargs={kwargs_info}")
    
    control_mode = get_robot_control_mode(cfg.robot, cfg.policy_model)
    success_arr = []
    
    # obj_variation_mode == "episode"の場合のカスタム処理
    if cfg.obj_variation_mode == "episode":
        rng = np.random.RandomState(42)
        sampled_ids = rng.choice(range(36), size=cfg.obj_episode_range[1], replace=True)
        
        for idx, obj_episode_id in enumerate(sampled_ids):
            # 各タスクで0からスタートするepisode_id
            episode_id = idx
            
            success = run_maniskill2_eval_single_episode(
                model=env_policy,
                task_name=cfg.task_name,
                episode_id=episode_id,
                ckpt_path=cfg.ckpt_path,
                robot_name=cfg.robot,
                env_name=cfg.env_name,
                scene_name=cfg.scene_name,
                robot_init_x=cfg.robot_init_x_range[0],
                robot_init_y=cfg.robot_init_y_range[0],
                robot_init_quat=np.array(cfg.robot_init_rot_quat_center),
                control_mode=control_mode,
                obj_episode_id=obj_episode_id,
                additional_env_build_kwargs=cfg.additional_env_build_kwargs,
                rgb_overlay_path=cfg.rgb_overlay_path,
                control_freq=cfg.control_freq,
                sim_freq=cfg.sim_freq,
                max_episode_steps=cfg.max_episode_steps,
                enable_raytracing=cfg.enable_raytracing,
                additional_env_save_tags=cfg.additional_env_save_tags,
                obs_camera_name=cfg.obs_camera_name,
                logging_dir="./results",
            )
            success_arr.append(success)
    else:
        # 他のバリエーションモードは元の実装を使用
        return _run_single_evaluation(env_policy, cfg, ckpt_path)
    
    # 成功率を出力
    print(f"  ✅  Success Rate: {np.mean(success_arr):.2%}")
    return success_arr


def widowx_task1_pick_object(env_policy: AiroaBasePolicy, ckpt_path: str, episode_id: int, control_freq: int = 5) -> List[List[bool]]:
    max_time = 24
    max_episode_steps = max_time * control_freq

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=max_episode_steps,
        env_name="GraspRandomObjectInScene-v0",
        scene_name="bridge_table_1_v1",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task1_pick_object",
        episode_id=episode_id,
    )
    results.append(custom_run_single_evaluation_with_counter(env_policy, cfg, ckpt_path, episode_id))

    return results


def widowx_task2_stack_cube(env_policy: AiroaBasePolicy, ckpt_path: str, episode_id: int, control_freq: int = 5) -> List[List[bool]]:
    max_time = 24
    max_episode_steps = max_time * control_freq

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=max_episode_steps,
        env_name="StackRandomGreenYellowCubeInScene-v0",
        scene_name="bridge_table_1_v1",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task2_stack_cube",
        episode_id=episode_id,
    )
    results.append(custom_run_single_evaluation_with_counter(env_policy, cfg, ckpt_path, episode_id))

    return results


def widowx_task3_put_object_on_top(env_policy: AiroaBasePolicy, ckpt_path: str, episode_id: int, control_freq: int = 5) -> List[List[bool]]:
    max_time = 24
    max_episode_steps = max_time * control_freq

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=max_episode_steps,
        env_name="PutRandomObjectOnRandomTopInScene-v0",
        scene_name="bridge_table_1_v1",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task3_put_object_on_top",
        episode_id=episode_id,
    )
    results.append(custom_run_single_evaluation_with_counter(env_policy, cfg, ckpt_path, episode_id))

    return results


def widowx_task4_put_object_in_basket(env_policy: AiroaBasePolicy, ckpt_path: str, episode_id: int, control_freq: int = 5) -> List[List[bool]]:
    max_time = 48
    max_episode_steps = max_time * control_freq

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx_sink_camera_setup",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=max_episode_steps,
        env_name="PutRandomObjectInBasketScene-v0",
        scene_name="bridge_table_1_v2",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        robot_init_x_range=[0.127, 0.127, 1],
        robot_init_y_range=[0.06, 0.06, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task4_put_object_in_basket",
        episode_id=episode_id,
    )
    results.append(custom_run_single_evaluation_with_counter(env_policy, cfg, ckpt_path, episode_id))

    return results
