import argparse

from simpler_env.evaluation.evaluate_bridge import run_comprehensive_evaluation_bridge
from simpler_env.policies.openpi.pi0_or_fast import OpenPiFastInference


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = OpenPiFastInference(saved_model_path=ckpt_path, policy_setup="widowx_bridge", action_scale=1.0)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation_bridge(env_policy=policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
