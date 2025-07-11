import copy
import colorsys
import os,sys
from environment.env import MarineEnv
from policy.agent import Agent
from thirdparty.APF import ApfAgent
from visualization import env_visualizer
import cv2
import shutil
from datetime import datetime


class generate_video():
    def __init__(self, num_pursuers, num_evaders, num_cores, num_obs):
        self.seed = 10
        self.eval_env = MarineEnv(self.seed)
        self.eval_env.num_pursuers = int(num_pursuers)
        self.eval_env.num_evaders = int(num_evaders)
        self.eval_env.num_cores = int(num_cores)
        self.eval_env.num_obs = int(num_obs)
        pursuer_state, _ = self.eval_env.reset()
        self.state, _ = pursuer_state
        self.evader_state, _ = self.eval_env.get_evaders_observation()
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        save_dir = f"TrainedModels/TERL"
        model_dir = os.path.join(self.project_root, save_dir)
        self.TERL_agent = Agent(device="cpu",model_name="TERL")
        self.TERL_agent.load_model(model_dir, "cpu")
        self.evader_agent = ApfAgent(self.eval_env.evaders[0].a, self.eval_env.evaders[0].w)

    def evaluation(self, pursuer_state, evader_state, agent, evader_agent, eval_env:MarineEnv, use_rl=True, use_iqn=True, act_adaptive=True):
        """Evaluate performance of the agent.
        """
        end_episode = False
        length = 0
        while not end_episode:
            action = []
            for i, rob in enumerate(eval_env.pursuers):
                if rob.deactivated:
                    action.append(None)
                    continue
                assert rob.robot_type == 'pursuer', "Every robot must be pursuer!"
                if use_rl:
                    if use_iqn:
                        if act_adaptive:
                            a, _, _, _ = agent.act_adaptive(pursuer_state[i])
                        else:
                            a, _, _ = agent.act(pursuer_state[i])
                    else:
                        a, _ = agent.act_dqn(pursuer_state[i])
                else:
                    a = agent.act(pursuer_state[i])
                action.append(a)
            evaders_action = []
            for j, evader in enumerate(eval_env.evaders):
                if evader.deactivated:
                    evaders_action.append(None)
                    continue
                assert evader.robot_type == 'evader', "Every robot must be evader!"
                a = evader_agent.act(evader_state[j])
                evaders_action.append(a)
            # Execute actions in the training environment
            pursuer_state, reward, done, info = eval_env.step((action, evaders_action))
            evader_state, _ = eval_env.get_evaders_observation()
            for i, rob in enumerate(eval_env.pursuers):
                if rob.deactivated:
                    continue
                assert rob.robot_type == 'pursuer', "Every robot must be pursuer!"
                if rob.collision:
                    rob.deactivated = True
            end_episode = (length >= 1000) or len([pursuer for pursuer in eval_env.pursuers if
                                                not pursuer.deactivated]) < 3 or eval_env.check_all_evader_is_captured()
            length += 1
        trajectories = [copy.deepcopy(rob.trajectory) for rob in eval_env.pursuers + eval_env.evaders]
        return trajectories
    

    def run_experiment(self):
        eavl_configs = self.eval_env.episode_data()
        trajectories = self.evaluation(self.state, self.evader_state, self.TERL_agent, self.evader_agent, self.eval_env)
        dt = datetime.now()
        create_timestamp = dt.strftime("%H-%M-%S")
        save_dir = os.path.join(self.project_root, "photos")
        # 清除 save_dir 路径下的所有非 .mp4 文件和文件夹
        if os.path.exists(save_dir):
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        if not filename.lower().endswith('.mp4'):
                            os.unlink(file_path)  # 删除非 .mp4 文件或链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除文件夹
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        os.makedirs(save_dir, exist_ok=True)
        ev = env_visualizer.EnvVisualizer(video_plots=True)
        ev.agent_name = "TERL"
        ev.env.reset_with_eval_config(eavl_configs)
        ev.init_visualize()
        colors = self.generate_hsv_colors(120)
        ev.draw_video_plots(trajectories=trajectories,colors=colors, save_dir=save_dir)

        #获取图片列表
        images = [img for img in os.listdir(save_dir) if img.endswith(".png")]
        images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # 按步骤序号排序

        if images:
            frame = cv2.imread(os.path.join(save_dir,images[0]))
            height, width, layers = frame.shape

            #定义视频编码器并创建VideoWriter对象
            video_path = os.path.join(save_dir, f"{create_timestamp}_TERL.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需要选择不同的编码器
            video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))  # 10为帧率，可以根据需要调整
            
            #将图片写入视频
            for image in images:
                video.write(cv2.imread(os.path.join(save_dir,image)))
            video.release() #释放资源
            return video_path


    def generate_hsv_colors(self, num_colors):
        return [colorsys.hsv_to_rgb(i / num_colors, 0.8, 0.9) for i in range(num_colors)]









