import argparse
import colorsys
import os
import sys
from datetime import datetime

sys.path.insert(0, "../")
import env_visualizer
import json
import numpy as np
from policy.agent import Agent
from utils import logger as logger
from config_manager import ConfigManager


def initialize_config_manager(config_file):
    """Initialize and load configuration manager.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        ConfigManager: An initialized ConfigManager instance with the loaded configuration.
    """
    config_manager = ConfigManager()
    config_manager.load_config(config_file)
    return config_manager


if __name__ == "__main__":
    # Get the project root directory, which is the parent directory of the current script
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run experiment with different training data and model index.')

    # Use short parameters -m for model_index, and set required=True to ensure mandatory input
    parser.add_argument('-m', type=int, required=True, help='Index for the model in the data set (required)')
    parser.add_argument(
        '-v', "--video", action="store_true", help="Whether to save video of the episode"
    )
    args = parser.parse_args()

    module_list = [
        "baseline_exp_data_3_2025-02-23-03-37-03",
    ]  # Adjust the module list to your experiment data folder

    module = module_list[args.m]
    folder_name = f"experiment_data/{module}"
    filename = "exp_results.json"

    file_path = os.path.join(project_root, folder_name, filename)

    with open(file_path, "r") as f:
        exp_data = json.load(f)

    logger.info(f"Loaded data from {file_path}")

    model_name = [
        "TERL",
        "IQN",
        "MEAN",
        "DQN",
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_file = os.path.join(project_root, "config", f"exp_config_large_scale.yaml")  # Adjust the path to your config file

    config_manager = initialize_config_manager(config_file)

    def generate_hsv_colors(num_colors):
        """Generate a list of HSV-based colors.

        Args:
            num_colors (int): Number of colors to generate.

        Returns:
            list: A list of RGB color tuples.
        """
        return [colorsys.hsv_to_rgb(i / num_colors, 0.8, 0.9) for i in range(num_colors)]

    colors = generate_hsv_colors(120)

    dt = datetime.now()
    create_timestamp = dt.strftime("%Y-%m-%d-%H-%M")

    schedule_id = 2
    agent_id = 0
    ep_id = 1

    if args.video:
        for agent_id in range(len(exp_data["all_trajectories_exp"][schedule_id])):
            save_dir = os.path.join(current_dir, module, model_name[agent_id], create_timestamp)
            os.makedirs(save_dir, exist_ok=True)
            ev = env_visualizer.EnvVisualizer(video_plots=True)
            ev.agent_name = model_name[agent_id]
            ev.env.reset_with_eval_config(exp_data["all_eval_configs_exp"][schedule_id][agent_id][ep_id])
            ev.init_visualize()
            ev.draw_video_plots(trajectories=exp_data["all_trajectories_exp"][schedule_id][agent_id][ep_id],
                                colors=colors, save_dir=save_dir)
            print(f"{model_name[agent_id]} done.")
    else:
        for agent_id in range(len(exp_data["all_trajectories_exp"][schedule_id])):
            save_dir = os.path.join(current_dir, module, model_name[agent_id], create_timestamp)
            os.makedirs(save_dir, exist_ok=True)
            stamp = f"schedule_{schedule_id}_agent_{agent_id}_ep_{ep_id}"
            ev = env_visualizer.EnvVisualizer()
            ev.env.reset_with_eval_config(exp_data["all_eval_configs_exp"][schedule_id][agent_id][ep_id])
            ev.init_visualize()
            ev.play_episode(trajectories=exp_data["all_trajectories_exp"][schedule_id][agent_id][ep_id],
                            colors=colors)
            ev.save_figure(f"{stamp}.pdf", save_dir)
