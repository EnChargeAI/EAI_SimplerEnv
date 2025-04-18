from collections import defaultdict
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_agents
from tf_agents.trajectories import time_step as ts
from transforms3d.euler import euler2axangle


class RT1DropoutWrapper(tf.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = tf.saved_model.load(model_path)

    def __call__(self, time_step, policy_state, training=True):
        return self.model.action(time_step, policy_state, training=training)


class RT1Inference:
    def __init__(
        self,
        saved_model_path: str = "rt_1_x_tf_trained_for_002272480_step",
        lang_embed_model_path: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_width: int = 320,
        image_height: int = 256,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ) -> None:
        self.lang_embed_model = hub.load(lang_embed_model_path)
        self.policy_model = RT1DropoutWrapper(saved_model_path)
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    def _initialize_model(self) -> None:
        self.observation = {
            "image": tf.zeros((self.image_height, self.image_width, 3), dtype=tf.uint8),
            "natural_language_embedding": tf.zeros((512,), dtype=tf.float32),
        }
        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        self.policy_state = self.policy_model.model.get_initial_state(batch_size=1)

    def _resize_image(self, image: np.ndarray | tf.Tensor) -> tf.Tensor:
        image = tf.image.resize_with_pad(image, target_width=self.image_width, target_height=self.image_height)
        return tf.cast(image, tf.uint8)

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([task_description])[0]
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self, task_description: str) -> None:
        self._initialize_model()
        self._initialize_task_description(task_description)

    def _unnormalize_action_widowx_bridge(self, action):
        def _rescale_action(actions, low, high, post_max=1.0, post_min=-1.0):
            resc = (actions - low) / (high - low) * (post_max - post_min) + post_min
            return np.clip(resc, post_min, post_max)

        action["world_vector"] = _rescale_action(action["world_vector"], -1.75, 1.75, 0.05, -0.05)
        action["rotation_delta"] = _rescale_action(action["rotation_delta"], -1.4, 1.4, 0.25, -0.25)
        return action

    def step(self, image: np.ndarray, task_description: Optional[str] = None, training: bool = False):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self.observation["image"] = image
        self.observation["natural_language_embedding"] = self.task_description_embedding
        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))

        policy_step = self.policy_model(self.tfa_time_step, self.policy_state, training=training)
        raw_action = policy_step.action

        if self.policy_setup == "google_robot":
            raw_action["gripper_closedness_action"] = tf.where(
                tf.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                tf.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"]
            )
        if self.unnormalize_action:
            raw_action = self.unnormalize_action_fxn(raw_action)
        for k in raw_action:
            raw_action[k] = np.asarray(raw_action[k])

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        rot_delta = raw_action["rotation_delta"]
        angle = np.linalg.norm(rot_delta)
        axis = rot_delta / angle if angle > 1e-6 else np.array([0.0, 1.0, 0.0])
        action["rot_axangle"] = axis * angle * self.action_scale

        raw_grip = raw_action["gripper_closedness_action"]
        if self.invert_gripper_action:
            raw_grip = -raw_grip
        action["gripper"] = raw_grip
        if self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (action["gripper"] > 0.0) - 1.0

        action["terminate_episode"] = raw_action["terminate_episode"]
        self.policy_state = policy_step.state

        return raw_action, action

    def mc_dropout_inference(self, image, task_description=None, num_samples=10):
        raw_actions_list = []
        processed_actions = []

        dropout_found = False
        for layer in self.policy_model.model.submodules:
            if isinstance(layer, tf.keras.layers.Dropout):
                print(f"Dropout found in: {layer.name}")
                dropout_found = True

        if not dropout_found:
            print("\u26a0\ufe0f No Dropout layers found. MC Dropout will NOT work.")

        for _ in range(num_samples):
            raw_action, action = self.step(image, task_description, training=True)
            raw_actions_list.append(raw_action)
            processed_actions.append(action)

        mean_action = {k: np.mean(np.stack([a[k] for a in processed_actions]), axis=0) for k in processed_actions[0]}
        std_action = {k: np.std(np.stack([a[k] for a in processed_actions]), axis=0) for k in processed_actions[0]}

        return raw_actions_list, mean_action, std_action
