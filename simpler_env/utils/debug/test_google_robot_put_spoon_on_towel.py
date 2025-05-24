import gymnasium as gym
import numpy as np
import mani_skill2_real2sim.envs # Import to register custom environments

def run_test():
    env_id = "GoogleRobotPutSpoonOnTowel-v0"
    
    print(f"Creating environment with ID: {env_id}")
    try:
        env = gym.make(
            env_id,
            obs_mode="state_dict",
            robot_uid="google_robot_static", # Explicitly set, though default in env
            control_mode="arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
            render_mode="human" 
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    print("Environment created successfully.")

    try:
        obs, info = env.reset()
        print("Initial observation:", obs)
        print("Initial info:", info)
    except Exception as e:
        print(f"Error resetting environment: {e}")
        env.close()
        return

    env.render() # Initial render

    # Construct a zero action for the specified control mode
    # For "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
    # the action is a 7-dim array: 6 for arm pose (delta_xyz, delta_axis_angle), 1 for gripper (target_delta_pos).
    action = np.zeros(7) 

    print(f"Running a small loop for a few steps with zero action: {action}")
    for i in range(100): # Run for 100 steps
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            print(f"Step: {i+1}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
            
            if terminated or truncated:
                print(f"Episode finished at step {i+1}. Terminated: {terminated}, Truncated: {truncated}")
                break
        except Exception as e:
            print(f"Error during step {i+1}: {e}")
            break
            
    print("Closing environment.")
    env.close()

if __name__ == "__main__":
    run_test()
