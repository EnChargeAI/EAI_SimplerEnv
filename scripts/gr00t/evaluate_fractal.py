import argparse

from simpler_env.evaluation.fractal_tasks import run_comprehensive_evaluation
from simpler_env.policies.gr00t.gr00t_model import Gr00tInference


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--control-freq", type=int, default=3, help="Set control frequency (default->3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = Gr00tInference(saved_model_path=ckpt_path, policy_setup="google_robot", action_scale=1.0)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=policy, ckpt_path=args.ckpt_path, control_freq=args.control_freq)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
