import argparse

from openpi_client import action_chunk_broker

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from simpler_env.evaluation.fractal_tasks import run_comprehensive_evaluation
from simpler_env.policies.adapter import AiroaToSimplerFractalAdapter
from simpler_env.policies.hsr_openpi.pi0_or_fast import OpenpiToAiroaPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = OpenpiToAiroaPolicy(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=_policy_config.create_trained_policy(
                _config.get_config("pi0_fractal_low_mem_finetune"),
                ckpt_path,
            ),
            action_horizon=10,
        ),
    )

    env_policy = AiroaToSimplerFractalAdapter(policy=policy)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=env_policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
