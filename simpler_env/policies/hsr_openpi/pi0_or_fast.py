from typing import Dict

import numpy as np

from simpler_env.policies.base import AiroaBasePolicy


class OpenpiToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy

    def step(self, obs: Dict) -> Dict:
        outputs = self.policy.infer(obs)
        outputs["terminate_episode"] = np.zeros(outputs["actions"].shape[0])
        return outputs

    def reset(self) -> None:
        self.policy.reset()
