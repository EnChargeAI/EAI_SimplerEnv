import random
from typing import Any, Dict, List, Tuple

import numpy as np

from ..policies.base import AiroaBasePolicy
from .config import ManiSkill2Config
from .maniskill2_evaluator import maniskill2_evaluator


def calculate_robust_score(results: List[List[bool]], penalty_factor: float = 0.5) -> float:
    if not results:
        return 0.0
    success_rates = [np.mean(run) for run in results if run]
    if not success_rates:
        return 0.0
    mean_success_rate = np.mean(success_rates)
    std_dev = np.std(success_rates)
    robust_score = mean_success_rate - penalty_factor * std_dev
    return max(0.0, robust_score)


def _run_single_evaluation(env_policy: AiroaBasePolicy, cfg: ManiSkill2Config, ckpt_path: str) -> List[bool]:
    if cfg.additional_env_build_kwargs:
        if "urdf_version" in cfg.additional_env_build_kwargs:
            kwargs_info = cfg.additional_env_build_kwargs["urdf_version"]
        else:
            kwargs_info = cfg.additional_env_build_kwargs
    else:
        kwargs_info = None

    print(f"  ▶️  Running: env={cfg.env_name}, scene={cfg.scene_name}, kwargs={kwargs_info}")
    success_arr = maniskill2_evaluator(env_policy, cfg)
    print(f"  ✅  Success Rate: {np.mean(success_arr):.2%}")
    return success_arr


def pick_object_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- pick_object_visual_matching ---")
    results: List[List[bool]] = []

    direction_orientationions_arr = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]

    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 8],
        obj_init_y_range=[-0.02, 0.42, 8],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 5],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
    )

    for urdf in urdf_versions:
        for orientation in direction_orientationions_arr:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleRandomObjectInScene-v0",
                scene_name="google_pick_coke_can_1_v4",
                rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
                additional_env_build_kwargs={**orientation, "urdf_version": urdf},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def pick_object_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- pick_object_variant_agg ---")
    results: List[List[bool]] = []

    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 4],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
    )

    # base
    for orientation in object_orientation:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs={**orientation},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # table textures (baked)
    baked_scenes = ["Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"]
    for scene in baked_scenes:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleRandomObjectInScene-v0",
                scene_name=scene,
                additional_env_build_kwargs={**orientation},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # backgrounds
    bg_scenes = ["google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2"]
    for scene in bg_scenes:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleRandomObjectInScene-v0",
                scene_name=scene,
                additional_env_build_kwargs={**orientation},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # lightings (darker / brighter)
    for orientation in object_orientation:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs={**orientation, "slightly_darker_lighting": True},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs={**orientation, "slightly_brighter_lighting": True},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # camera orientations
    alt_envs = ["GraspSingleRandomObjectAltGoogleCameraInScene-v0", "GraspSingleRandomObjectAltGoogleCamera2InScene-v0"]
    for env in alt_envs:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name=env,
                scene_name="google_pick_coke_can_1_v4",
                additional_env_build_kwargs={**orientation},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


# scenes = ["google_pick_coke_can_1_v4",
#           "google_pick_coke_can_1_v4_alt_background",
#           "google_pick_coke_can_1_v4_alt_background_2",
#           "Baked_sc1_staging_objaverse_cabinet1_h870",
#           "Baked_sc1_staging_objaverse_cabinet2_h870"]
# object_orientation = [
#     {"lr_switch": True},
#     {"upright": True},
#     {"laid_vertically": True},
# ]
# envs = ["GraspSingleRandomObjectInScene-v0", "GraspSingleRandomObjectAltGoogleCameraInScene-v0", "GraspSingleRandomObjectAltGoogleCamera2InScene-v0"]
# lightings = [None, "darker", "brighter"]

# trail_num = 30
# for t in range(trial_num):
#     random_kwargs() = np.random(scenes, object_orientation, envs, lightings)
#     cfg = ManiSkill2Config(**base_kwargs, **random_kwargs)


def pick_object_variant_agg(
    env_policy: "AiroaBasePolicy",
    ckpt_path: str,
    num_trials: int = 30,
) -> List[List[bool]]:
    print("\n--- pick_object_variant_agg (single-pool, one-pick-per-trial) ---")

    results: List[List[bool]] = []

    scenes = [
        "google_pick_coke_can_1_v4",
        "google_pick_coke_can_1_v4_alt_background",
        "google_pick_coke_can_1_v4_alt_background_2",
        "Baked_sc1_staging_objaverse_cabinet1_h870",
        "Baked_sc1_staging_objaverse_cabinet2_h870",
    ]
    envs = [
        "GraspSingleRandomObjectInScene-v0",
        "GraspSingleRandomObjectAltGoogleCameraInScene-v0",
        "GraspSingleRandomObjectAltGoogleCamera2InScene-v0",
    ]
    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]
    lightings = [None, "darker", "brighter"]

    base_kwargs: Dict[str, Any] = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 4],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
    )

    # 単一プール（どれが選ばれたか識別できるよう (kind, value) で保持）
    pool = (
        [("scene", s) for s in scenes]
        + [("env", e) for e in envs]
        + [("orientation", o) for o in object_orientation]
        + [("lighting", l) for l in lightings]
    )

    # 現在の設定（初期値）
    current = {
        "scene": scenes[0],
        "env": envs[0],
        "orientation": object_orientation[0],
        "lighting": None,  # デフォ照明
    }

    for _ in range(num_trials):
        kind, value = random.choice(pool)  # ← 毎回1つだけ選ぶ
        current[kind] = value  # 選ばれた種類だけ更新

        # additional_env_build_kwargs を構築
        add_kwargs = dict(current["orientation"])
        if current["lighting"] == "darker":
            add_kwargs["slightly_darker_lighting"] = True
        elif current["lighting"] == "brighter":
            add_kwargs["slightly_brighter_lighting"] = True

        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name=current["env"],
            scene_name=current["scene"],
            additional_env_build_kwargs=add_kwargs,
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def pick_object_among_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- pick_object_among_visual_matching ---")
    results: List[List[bool]] = []

    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]

    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 5],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
    )

    for urdf in urdf_versions:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleRandomObjectDistractorInScene-v0",
                scene_name="google_pick_coke_can_1_v4",
                rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
                additional_env_build_kwargs={**orientation, "urdf_version": urdf, "distractor_config": "less"},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def pick_object_among_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- pick_object_among_variant_agg ---")
    results: List[List[bool]] = []

    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 5],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
    )

    # base
    for orientation in object_orientation:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectDistractorInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs={**orientation, "distractor_config": "less"},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # table textures (baked)
    baked_scenes = ["Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"]
    for scene in baked_scenes:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleRandomObjectDistractorInScene-v0",
                scene_name=scene,
                additional_env_build_kwargs={**orientation, "distractor_config": "less"},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # backgrounds
    bg_scenes = ["google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2"]
    for scene in bg_scenes:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name="GraspSingleRandomObjectDistractorInScene-v0",
                scene_name=scene,
                additional_env_build_kwargs={**orientation, "distractor_config": "less"},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # lightings (darker / brighter)
    for orientation in object_orientation:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectDistractorInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs={**orientation, "slightly_darker_lighting": True, "distractor_config": "less"},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectDistractorInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs={**orientation, "slightly_brighter_lighting": True, "distractor_config": "less"},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # camera orientations
    alt_envs = ["GraspSingleRandomObjectDistractorAltGoogleCameraInScene-v0", "GraspSingleRandomObjectDistractorAltGoogleCamera2InScene-v0"]
    for env in alt_envs:
        for orientation in object_orientation:
            cfg = ManiSkill2Config(
                **base_kwargs,
                env_name=env,
                scene_name="google_pick_coke_can_1_v4",
                additional_env_build_kwargs={**orientation, "distractor_config": "less"},
            )
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def drawer_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- drawer_visual_matching ---")
    results: List[List[bool]] = []

    env_names = [
        "OpenTopDrawerCustomInScene-v0",
        "OpenMiddleDrawerCustomInScene-v0",
        "OpenBottomDrawerCustomInScene-v0",
        "CloseTopDrawerCustomInScene-v0",
        "CloseMiddleDrawerCustomInScene-v0",
        "CloseBottomDrawerCustomInScene-v0",
    ]
    urdf_versions = ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]

    base = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=113,
        ckpt_path=ckpt_path,
        robot_init_rot_quat_center=[0, 0, 0, 1],
        obj_init_x_range=[0, 0, 1],
        obj_init_y_range=[0, 0, 1],
        scene_name="dummy_drawer",
    )

    # 9 overlay poses (A0/A1/A2/B0/B1/B2/C0/C1/C2)
    overlay_poses = [
        # A0/A1/A2
        dict(
            robot_init_x_range=[0.644, 0.644, 1],
            robot_init_y_range=[-0.179, -0.179, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png",
        ),
        dict(
            robot_init_x_range=[0.765, 0.765, 1],
            robot_init_y_range=[-0.182, -0.182, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.02, -0.02, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png",
        ),
        dict(
            robot_init_x_range=[0.889, 0.889, 1],
            robot_init_y_range=[-0.203, -0.203, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.06, -0.06, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png",
        ),
        # B0/B1/B2
        dict(
            robot_init_x_range=[0.652, 0.652, 1],
            robot_init_y_range=[0.009, 0.009, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png",
        ),
        dict(
            robot_init_x_range=[0.752, 0.752, 1],
            robot_init_y_range=[0.009, 0.009, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png",
        ),
        dict(
            robot_init_x_range=[0.851, 0.851, 1],
            robot_init_y_range=[0.035, 0.035, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png",
        ),
        # C0/C1/C2
        dict(
            robot_init_x_range=[0.665, 0.665, 1],
            robot_init_y_range=[0.224, 0.224, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png",
        ),
        dict(
            robot_init_x_range=[0.765, 0.765, 1],
            robot_init_y_range=[0.222, 0.222, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.025, -0.025, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png",
        ),
        dict(
            robot_init_x_range=[0.865, 0.865, 1],
            robot_init_y_range=[0.222, 0.222, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.025, -0.025, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png",
        ),
    ]

    add_base = dict(station_name="mk_station_recolor", light_mode="simple", disable_bad_material=True)

    for urdf in urdf_versions:
        add_kwargs = {**add_base, "urdf_version": urdf}
        for env_name in env_names:
            for pose in overlay_poses:
                cfg = ManiSkill2Config(
                    **base,
                    env_name=env_name,
                    additional_env_build_kwargs=add_kwargs,
                    **pose,
                )
                results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def drawer_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- drawer_variant_agg ---")
    results: List[List[bool]] = []

    env_names = [
        "OpenTopDrawerCustomInScene-v0",
        "OpenMiddleDrawerCustomInScene-v0",
        "OpenBottomDrawerCustomInScene-v0",
        "CloseTopDrawerCustomInScene-v0",
        "CloseMiddleDrawerCustomInScene-v0",
        "CloseBottomDrawerCustomInScene-v0",
    ]

    base = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=113,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.65, 0.85, 3],
        robot_init_y_range=[-0.2, 0.2, 3],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
        obj_init_x_range=[0, 0, 1],
        obj_init_y_range=[0, 0, 1],
    )

    # base (enable raytracing)
    for env_name in env_names:
        cfg = ManiSkill2Config(**base, env_name=env_name, scene_name="frl_apartment_stage_simple", enable_raytracing=True)
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # backgrounds (shader_dir=rt)
    for scene in ["modern_bedroom_no_roof", "modern_office_no_roof"]:
        merged = {"shader_dir": "rt"}
        for env_name in env_names:
            cfg = ManiSkill2Config(**base, env_name=env_name, scene_name=scene, additional_env_build_kwargs=merged)
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # lightings
    for light in ["brighter", "darker"]:
        merged = {"shader_dir": "rt", "light_mode": light}
        for env_name in env_names:
            cfg = ManiSkill2Config(**base, env_name=env_name, scene_name="frl_apartment_stage_simple", additional_env_build_kwargs=merged)
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # new cabinets
    for station in ["mk_station2", "mk_station3"]:
        merged = {"shader_dir": "rt", "station_name": station}
        for env_name in env_names:
            cfg = ManiSkill2Config(**base, env_name=env_name, scene_name="frl_apartment_stage_simple", additional_env_build_kwargs=merged)
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def move_near_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- move_near_variant_agg ---")
    results: List[List[bool]] = []

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=80,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.21, 0.21, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        ckpt_path=ckpt_path,
    )

    # base
    cfg = ManiSkill2Config(**base_kwargs, env_name="MoveNearGoogleInScene-v0", scene_name="google_pick_coke_can_1_v4")
    results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # distractor(no_distractor=True)
    cfg = ManiSkill2Config(
        **base_kwargs,
        env_name="MoveNearGoogleInScene-v0",
        scene_name="google_pick_coke_can_1_v4",
        additional_env_build_kwargs={"no_distractor": True},
    )
    results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # backgrounds
    for scene in ["google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2"]:
        cfg = ManiSkill2Config(**base_kwargs, env_name="MoveNearGoogleInScene-v0", scene_name=scene)
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # lighting
    for k in [{"slightly_darker_lighting": True}, {"slightly_brighter_lighting": True}]:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="MoveNearGoogleInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            additional_env_build_kwargs=k,
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # table textures (baked)
    for scene in ["Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"]:
        cfg = ManiSkill2Config(**base_kwargs, env_name="MoveNearGoogleInScene-v0", scene_name=scene)
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # camera orientations
    for env in ["MoveNearAltGoogleCameraInScene-v0", "MoveNearAltGoogleCamera2InScene-v0"]:
        cfg = ManiSkill2Config(**base_kwargs, env_name=env, scene_name="google_pick_coke_can_1_v4")
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def move_near_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- move_near_visual_matching ---")
    results: List[List[bool]] = []

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.21, 0.21, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        ckpt_path=ckpt_path,
    )

    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]
    for urdf in urdf_versions:
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="MoveNearGoogleBakedTexInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png",
            additional_env_save_tags="baked_except_bpb_orange",
            additional_env_build_kwargs={"urdf_version": urdf},
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def put_in_drawer_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- put_in_drawer_visual_matching ---")
    results: List[List[bool]] = []

    env_names = ["PlaceIntoClosedTopDrawerCustomInScene-v0"]
    urdf_versions = ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]

    base = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=400,
        ckpt_path=ckpt_path,
        robot_init_rot_quat_center=[0, 0, 0, 1],
        obj_init_x_range=[-0.08, -0.02, 3],
        obj_init_y_range=[-0.02, 0.08, 3],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 3],
    )

    overlay_poses = [
        # A0
        dict(
            robot_init_x_range=[0.644, 0.644, 1],
            robot_init_y_range=[-0.179, -0.179, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png",
        ),
        # B0
        dict(
            robot_init_x_range=[0.652, 0.652, 1],
            robot_init_y_range=[0.009, 0.009, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png",
        ),
        # C0
        dict(
            robot_init_x_range=[0.665, 0.665, 1],
            robot_init_y_range=[0.224, 0.224, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png",
        ),
    ]

    add_base = dict(station_name="mk_station_recolor", light_mode="simple", disable_bad_material=True, model_ids="baked_apple_v2")

    for urdf in urdf_versions:
        add_kwargs = {**add_base, "urdf_version": urdf}
        for env_name in env_names:
            for pose in overlay_poses:
                cfg = ManiSkill2Config(
                    **base,
                    env_name=env_name,
                    scene_name="dummy_drawer",
                    additional_env_build_kwargs=add_kwargs,
                    **pose,
                )
                results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def put_in_drawer_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    print("\n--- put_in_drawer_variant_agg ---")
    results: List[List[bool]] = []

    common = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=400,
        ckpt_path=ckpt_path,
        additional_env_build_kwargs={"model_ids": "apple"},
        robot_init_rot_quat_center=[0, 0, 0, 1],
        obj_init_x_range=[-0.08, -0.02, 3],
        obj_init_y_range=[-0.02, 0.08, 3],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 3],
        robot_init_x_range=[0.65, 0.65, 1],
        robot_init_y_range=[-0.2, 0.2, 3],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
    )

    env_names = ["PlaceIntoClosedTopDrawerCustomInScene-v0"]

    # base (enable raytracing)
    for env_name in env_names:
        cfg = ManiSkill2Config(**common, env_name=env_name, scene_name="frl_apartment_stage_simple", enable_raytracing=True)
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # backgrounds (shader_dir=rt)
    for scene in ["modern_bedroom_no_roof", "modern_office_no_roof"]:
        merged = common["additional_env_build_kwargs"].copy()
        merged["shader_dir"] = "rt"
        for env_name in env_names:
            cfg = ManiSkill2Config(**common, env_name=env_name, scene_name=scene, additional_env_build_kwargs=merged)
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # lightings
    for light_mode in ["brighter", "darker"]:
        merged = common["additional_env_build_kwargs"].copy()
        merged.update({"shader_dir": "rt", "light_mode": light_mode})
        for env_name in env_names:
            cfg = ManiSkill2Config(**common, env_name=env_name, scene_name="frl_apartment_stage_simple", additional_env_build_kwargs=merged)
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # new cabinets
    for station in ["mk_station2", "mk_station3"]:
        merged = common["additional_env_build_kwargs"].copy()
        merged.update({"shader_dir": "rt", "station_name": station})
        for env_name in env_names:
            cfg = ManiSkill2Config(**common, env_name=env_name, scene_name="frl_apartment_stage_simple", additional_env_build_kwargs=merged)
            results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


# ======================================================================
# 総合評価（重み付け・スコアリングは従来どおり）
# ======================================================================
SIM_WEIGHT = 0.3
VISUAL_MATCHING_WEIGHT = 0.7


def run_comprehensive_evaluation(env_policy: AiroaBasePolicy, ckpt_path: str) -> Dict[str, float]:
    print("=" * 80)
    print(f"🚀 STARTING COMPREHENSIVE EVALUATION 🚀")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Weights: Sim={SIM_WEIGHT}, VisualMatching={VISUAL_MATCHING_WEIGHT}")
    print("=" * 80)

    vm_results: List[List[bool]] = []
    sim_results: List[List[bool]] = []

    vm_results += pick_object_visual_matching(env_policy, ckpt_path)
    sim_results += pick_object_variant_agg(env_policy, ckpt_path)

    vm_results += pick_object_among_visual_matching(env_policy, ckpt_path)
    sim_results += pick_object_among_variant_agg(env_policy, ckpt_path)

    vm_results += drawer_visual_matching(env_policy, ckpt_path)
    sim_results += drawer_variant_agg(env_policy, ckpt_path)

    vm_results += move_near_visual_matching(env_policy, ckpt_path)
    sim_results += move_near_variant_agg(env_policy, ckpt_path)

    vm_results += put_in_drawer_visual_matching(env_policy, ckpt_path)
    sim_results += put_in_drawer_variant_agg(env_policy, ckpt_path)

    # ロバストスコア
    sim_score = calculate_robust_score(sim_results)
    vm_score = calculate_robust_score(vm_results)

    total_weight = SIM_WEIGHT + VISUAL_MATCHING_WEIGHT
    final_score = 0.0 if total_weight == 0 else (sim_score * SIM_WEIGHT + vm_score * VISUAL_MATCHING_WEIGHT) / total_weight

    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY 📊")
    print("-" * 80)
    print(f"Simulation Score (Robust):            {sim_score:.4f}")
    print(f"  - Total Simulation Runs: {len(sim_results)}")
    print(f"Visual Matching Score (Robust):       {vm_score:.4f}")
    print(f"  - Total Visual Matching Runs: {len(vm_results)}")
    print("-" * 80)
    print(f"🏆 Final Weighted Score:               {final_score:.4f}")
    print("=" * 80)

    return {
        "final_score": final_score,
        "simulation_robust_score": sim_score,
        "visual_matching_robust_score": vm_score,
    }
