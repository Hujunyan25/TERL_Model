import os
import platform
import sys
import numpy as np
import concurrent.futures
import env_visualizer
from policy.agent import Agent
import json
import argparse
from utils import logger

# Add parent directory to system path to allow importing modules from it
sys.path.insert(0, "../")

# Define the path to the model list file
MODEL_LIST_FILE = "model_list.json"


def load_model_list():
    """
    Load the model list from a JSON file. If the file does not exist, create an empty list.

    Returns:
        list: The loaded model list or an empty list if the file does not exist.
    """
    if os.path.exists(MODEL_LIST_FILE):
        with open(MODEL_LIST_FILE, 'r') as f:
            return json.load(f)
    return []


def save_model_list(model_list):
    """
    Save the model list to a JSON file.

    Args:
        model_list (list): The list of models to save.
    """
    with open(MODEL_LIST_FILE, 'w') as f:
        json.dump(model_list, f, indent=2)


def add_new_model(model_list, new_model):
    """
    Add a new model name to the list.

    Args:
        model_list (list): The current list of models.
        new_model (str): The new model name to add.

    Returns:
        list: The updated model list.
    """
    if new_model and new_model not in model_list:
        model_list.append(new_model)
        save_model_list(model_list)
        print(f"Model '{new_model}' has been added to the list.")
    return model_list


def list_models(model_list):
    """
    List all available models.

    Args:
        model_list (list): The list of models to display.
    """
    if not model_list:
        print("The model list is empty.")
    else:
        print("Available models:")
        for i, model in enumerate(model_list):
            print(f"{i + 1}. {model}")


def check_and_load_eval_file(eval_file_path):
    """
    Check if the evaluation file exists and attempt to load it.

    Args:
        eval_file_path (str): The path to the evaluation file.

    Returns:
        np.ndarray: The loaded evaluation data.

    Raises:
        FileNotFoundError: If the evaluation file does not exist.
        Exception: If an error occurs during file loading.
    """
    if not os.path.isfile(eval_file_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file_path}")

    try:
        data = np.load(eval_file_path, allow_pickle=True)
        print("File loaded successfully.")
        print("Contents:", data.files)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        raise


def initialize_env_visualizer(seed):
    """
    Initialize the environment visualizer.

    Args:
        seed (int): The random seed.

    Returns:
        EnvVisualizer: An initialized instance of the environment visualizer.
    """
    return env_visualizer.EnvVisualizer(seed=seed)


def load_and_visualize_evaluation(ev, config_file, eval_file, eval_id, eval_episode, colors, filename, save_path):
    """
    Load evaluation configuration and data, and visualize the specified evaluation episode.

    Args:
        ev (EnvVisualizer): The environment visualizer instance.
        config_file (str): Path to the configuration file.
        eval_file (str): Path to the evaluation file.
        eval_id (int): Evaluation ID.
        eval_episode (int): Evaluation episode ID.
        colors (list): List of colors for different elements.
        filename (str): Name of the file to save.
        save_path (str): Path to save the visualization result.
    """
    ev.load_eval_config_and_episode(config_file=config_file, eval_file=eval_file)
    ev.play_eval_episode(eval_id=eval_id, episode_id=eval_episode, colors=colors)
    ev.save_figure(filename, save_path)


def create_save_filename(eval_episode, eval_id, training_data_index, training_date):
    """
    Create the folder and filename for saving the file.

    Args:
        eval_episode (int): Evaluation episode ID.
        eval_id (int): Evaluation ID.
        training_data_index (str): Training data index.
        training_date (str): Training date.

    Returns:
        tuple: The folder path and filename for saving the file.
    """
    save_filename = f'{training_date}_eval_{eval_id}-{eval_episode}.pdf'
    # Get the directory of the current Python file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the subfolder for saving the images
    figure_save_subfolder = os.path.join(current_dir, f"images/{training_data_index}")
    # Ensure the subfolder exists
    os.makedirs(figure_save_subfolder, exist_ok=True)
    return figure_save_subfolder, save_filename


def process_evaluation(training_data_index, training_date, eval_id, evaluations, eval_configs, colors):
    """
    Process a single evaluation.

    Args:
        training_data_index (str): Training data index.
        training_date (str): Training date.
        eval_id (int): Evaluation ID.
        evaluations (str): Evaluation filename.
        eval_configs (dict): Evaluation configuration.
        colors (list): Color configuration.
    """
    # Get the project root directory, which is the parent directory of process_evaluation.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dynamically generate the evaluation path
    _dir = os.path.join(project_root, str(training_data_index), str(training_date), "seed_9")
    os.makedirs(_dir, exist_ok=True)  # Ensure the directory exists

    logger.info(f"Plotting all evaluations in {_dir}")

    # Generate the evaluation file path
    eval_file_path = os.path.join(_dir, evaluations)
    logger.info(f"Directory: {eval_file_path}")

    for eval_episode in range(60):
        ev = initialize_env_visualizer(seed=231)
        figure_save_subfolder, save_filename = create_save_filename(eval_episode, eval_id, training_data_index,
                                                                    training_date)

        load_and_visualize_evaluation(
            ev=ev,
            config_file=os.path.join(_dir, eval_configs),
            eval_file=eval_file_path,
            eval_id=eval_id,
            eval_episode=eval_episode,
            colors=colors,
            filename=save_filename,
            save_path=figure_save_subfolder
        )


def main():
    """
    Main function to handle command-line arguments and execute corresponding operations.
    """
    parser = argparse.ArgumentParser(description="Manage and use model list")
    parser.add_argument("--add", help="Add a new model name")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--use", type=int, help="Select a model to use by index")
    parser.add_argument("--process-all", action="store_true", help="Process all models")
    args = parser.parse_args()

    # Load existing model list
    model_list = load_model_list()

    # Handle logic for adding a new model
    if args.add:
        model_list = add_new_model(model_list, args.add)
        print(f"Model '{args.add}' has been added to the list.")

    # Update handling of --use parameter
    if args.use is not None:
        if args.add:
            # If a new model was just added, use it directly
            model_name = args.add
            print(f"Using newly added model: {model_name}")
        elif 1 <= args.use <= len(model_list):
            model_name = model_list[args.use - 1]
            print(f"Using model: {model_name}")
        else:
            print("Invalid model index. Please use --list to view available models.")
            return

        process_single_model(model_name)

    # Handle other parameters
    elif args.list:
        list_models(model_list)
    elif args.process_all:
        process_all_models(model_list)
    elif not args.add:  # If no operation specified (except adding a model)
        parser.print_help()


def process_single_model(model_name):
    """
    Process a single model.

    Args:
        model_name (str): Name of the model to process.
    """
    training_data_set = [
        "training_data0",
        "training_data1",
        "training_data2",
    ]
    eval_configs = "eval_configs.json"
    evaluations = "evaluations.npz"
    colors = [
        "r", "lime", "cyan", "orange", "tab:olive", "white", "chocolate",
        "blue", "magenta", "yellow", "purple", "gold", "pink", "brown", "grey"
    ]
    eval_id = -1

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_evaluation, training_data_index, model_name, eval_id, evaluations, eval_configs,
                            colors)
            for training_data_index in training_data_set]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


def process_all_models(model_list):
    """
    Process all models using multiprocessing.

    Args:
        model_list (list): List of model names to process.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_model, model_name) for model_name in model_list]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred while processing a model: {e}")


if __name__ == "__main__":
    main()
