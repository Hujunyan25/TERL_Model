from visualization.visualize_evaluation_episode import env_visualizer
import os



def initialize_env_visualizer(seed):
    """
    Initialize the environment visualizer.

    Args:
        seed (int): The random seed.

    Returns:
        EnvVisualizer: An initialized instance of the environment visualizer.
    """
    return env_visualizer.EnvVisualizer(seed=seed)

#只加载TERL模型
def process_single_model():
    colors = [
        "r", "lime", "cyan", "orange", "tab:olive", "white", "chocolate",
        "blue", "magenta", "yellow", "purple", "gold", "pink", "brown", "grey"
    ]
    eval_configs = "eval_configs.json"
    try:
        process_evaluation(eval_configs,colors)
    except Exception as e:
        print(f"An error occurred: {e}")

def process_evaluation(eval_configs, colors):
    #获取脚本路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    #生成目录
    _dir = os.path.join(project_root)
    os.makedirs(_dir, exists_ok=True)

    

    