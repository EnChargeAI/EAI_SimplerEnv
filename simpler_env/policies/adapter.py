from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from simpler_env.utils.geometry import euler2axangle, mat2euler, quat2mat


class BaseAdapter:
    def __init__(self, policy):
        self.policy = policy

    def reset(self, task_description):
        pass

    def preprocess(self, image: np.ndarray, prompt: str, eef_pos: np.ndarray) -> Dict:
        pass

    def postprocess(self, outputs: Dict) -> Dict:
        pass

    def step(self, image: np.ndarray, prompt: str, eef_pos: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        inputs = self.preprocess(image, prompt, eef_pos)
        outputs = self.policy.step(inputs)
        state_gripper = inputs["state"][-1]
        action_gripper = outputs["actions"][-1]
        # print(f"state: {state_gripper} action: {action_gripper}")
        final_outputs = self.postprocess(outputs)
        simpler_outputs = {
            "world_vector": outputs["actions"][:3],
            "rot_axangle": outputs["actions"][3:6],
            "gripper": outputs["actions"][6:],
            "terminate_episode": outputs["terminate_episode"],
        }
        final_simpler_outputs = {
            "world_vector": final_outputs["actions"][:3],
            "rot_axangle": final_outputs["actions"][3:6],
            "gripper": final_outputs["actions"][6:],
            "terminate_episode": final_outputs["terminate_episode"],
        }
        return simpler_outputs, final_simpler_outputs

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(256, 256),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array([np.concatenate([a["world_vector"], a["rot_axangle"], a["gripper"]], axis=-1) for a in predicted_raw_actions])
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


# from https://github.com/allenzren/open-pi-zero/blob/main/src/agent/env_adapter/simpler.py
class AiroaToSimplerBridgeAdapter(BaseAdapter):
    def __init__(self, policy):
        super().__init__(policy)
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

    def reset(self, task_description):
        pass

    def preprocess(self, image: np.ndarray, prompt: str, eef_pos: np.ndarray) -> dict:

        proprio = eef_pos
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )

        inputs = {
            "image": image,
            "prompt": prompt,
            "state": proprio,
        }
        return inputs

    def postprocess(self, outputs: dict) -> dict:
        action = outputs["actions"]
        roll, pitch, yaw = action[3:6]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = action[-1]
        action_gripper = 2.0 * (action_gripper > 0.5) - 1.0

        action = np.concatenate(
            [
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [action_gripper],
            ]
        )
        return {
            "actions": action,
            "terminate_episode": outputs["terminate_episode"],
        }


class AiroaToSimplerFractalAdapter(BaseAdapter):
    def __init__(self, policy):
        super().__init__(policy)
        # Constants
        self.sticky_gripper_num_repeat = 15  # same used in Octo. Note this is for every individual action, not every action chunk. Control freq is 3Hz, so roughly sticky for 5 seconds.

    def reset(self, task_description):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0

    def preprocess(self, image: np.ndarray, prompt: str, eef_pos: np.ndarray) -> dict:
        """convert wxyz quat from simpler to xyzw used in fractal"""
        quat_xyzw = np.roll(eef_pos[3:7], -1)
        gripper_width = eef_pos[7]  # from simpler, 0 for close, 1 for open continuous
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        proprio = np.concatenate(
            (
                eef_pos[:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )

        # H W C [0, 255]
        inputs = {
            "image": image,
            "prompt": prompt,
            "state": proprio,
        }
        return inputs

    def postprocess(self, outputs: dict) -> dict:
        action = outputs["actions"]
        roll, pitch, yaw = action[3:6]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 open, 1 close for simpler

        gripper_action = action[-1]

        gripper_action = (gripper_action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -gripper_action
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -action
        # self.previous_gripper_action = action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # sticky closing
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # reaching maximum sticky
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        action = np.concatenate(
            [
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [relative_gripper_action],
            ]
        )

        return {
            "actions": action,
            "terminate_episode": outputs["terminate_episode"],
        }
