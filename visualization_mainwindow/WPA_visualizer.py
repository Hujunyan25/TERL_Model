import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, random
import numpy as np
from config import Config_all as Config
from entities.pursuer import Pursuer
from entities.escaper import Target
import matplotlib.patches as patches
import datetime
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'WenQuanYi Zen Hei', 'Microsoft YaHei']


class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(Config.MAP_X_RANGE[0], Config.MAP_X_RANGE[1])
        self.ax.set_ylim(Config.MAP_Y_RANGE[0], Config.MAP_Y_RANGE[1])
        self.ax.set_xlabel("X/m", fontsize=12)
        self.ax.set_ylabel("Y/m", fontsize=12)
        self.ax.set_title("无人机集群围捕", fontsize=14)
        self.itx = 0  # 初始化迭代次数
        
        # 绘制目标点
        self.target_position = self.ax.scatter(Config.TARGET_POS[0], Config.TARGET_POS[1], color='red', s=10, marker='*')
        # 初始化雁群散点
        self.goose_scat = self.ax.scatter([], [], color='blue', s=10)
        
        # self.ax.legend(fontsize=10)
        self.target = Target(Config)
        target_position = self.target.get_pos()  # 获取当前目标位置
        self.goose_swarm = Pursuer(Config, target_position)

        # ---------------------- 新增：距离数据记录初始化 ----------------------
        self.frame_list = []  # 存储记录的帧数
        # 存储每个无人机的距离序列，索引对应无人机编号（0=领航者）
        self.uav_distances = [[] for _ in range(Config.UAV_NUM)]
        self.mean_uav_distances = [] # 存储雁群中心点与目标的距离
        self.is_record = True  # 记录开关：包围后停止记录
        self.start_record = False #记录狼群包围的开始
        self.encircle_circle = patches.Circle(
            xy=(0, 0), 
            radius=6, 
            fill=False, 
            edgecolor='red', 
            linewidth=2, 
            linestyle='--', 
            alpha=0.6,
            label='Encircle Radius (6)'
        )
        self.ax.add_patch(self.encircle_circle)
        # 2. 遍历每个无人机，计算半径并创建圆
        # self.circles = []  # 存储所有圆对象，方便后续更新（如动态仿真）
        # colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']  # 区分不同无人机的圆
        # for idx in range(Config.UAV_NUM):
        #     # 计算当前无人机的动态半径
        #     r_i = self.goose_swarm.Ri_list[idx]
        #     # 初始化圆：圆心=target_position，半径=r_i
        #     circle = patches.Circle(
        #         xy=target_position,    # 固定圆心为目标位置
        #         radius=r_i,            # 动态计算的半径
        #         edgecolor=colors[idx],  # 边框色（循环取色）
        #         facecolor='none',      # 无填充（只显示边框，方便看多层圆）
        #         linewidth=2,           # 边框宽度
        #         alpha=0.7,             # 透明度（避免重叠遮挡）
        #         label=f'无人机{idx}的围捕圆'
        #     )
        #     # 将圆添加到坐标轴
        #     self.ax.add_patch(circle)
        #     self.circles.append(circle)  # 保存圆对象，后续可修改半径/位置
        

    def update_frame(self, frame):
        """动画更新函数，包围之后就立即停止更新"""
        
        #2.如果没有包围成功：正常执行所有更新逻辑
        # 更新目标位置
        self.target.update_position(frame, self.goose_swarm.get_positions())
        target_position = self.target.get_pos()
        # 更新雁群状态
        self.itx += 1
        self.goose_swarm.update_all_uavs(target_position) #当使用Pursuer这个类时才使用
        positions = self.goose_swarm.get_positions()
        min_dist = self.goose_swarm.get_all_agents_nearest_distances(positions[1:]) # 计算无人机之间的最近邻距离
        distance_to_target = np.linalg.norm(positions - target_position, axis=1)
        #1.检查是否已经包围成功，如果成功就返回当前元素
        target_position = self.target.get_pos()  # 获取当前目标位置
        if self.goose_swarm.check_encircle(target_position):
            # 包围成功：更新标题提示+返回当前所有动态元素（停止更新，固定画面）
            self.ax.set_title(
                f"无人机集群围捕 | 迭代步骤：{frame} | 包围成功！",
                fontsize=14, color='green', fontweight='bold'
            )
            self.is_record = False  # 停止记录距离数据
            # 返回所有动态元素（保持画面固定，不触发新更新）
            return self.goose_scat, self.target_position
        # 打印当前帧的最小距离

        # 记录当前帧的距离数据
        if self.goose_swarm.current_phase == 'wolves':
            self.start_record = True  # 开始记录狼群阶段的距离数据
        if self.is_record and self.start_record:
            self.frame_list.append(frame)
            # 记录雁群中心点与目标的距离
            total_distance = 0
            for uav_id in range(Config.UAV_NUM):
                uav_pos = positions[uav_id + 1]
                dist = np.linalg.norm(uav_pos - target_position)
                total_distance += dist
            average_distance = total_distance / Config.UAV_NUM - Config.MIN_ENCIRCLE_RADIUS
            
            self.mean_uav_distances.append(average_distance)
            # 记录每个无人机与目标的距离
            for uav_id in range(Config.UAV_NUM):
                uav_pos = positions[uav_id + 1]
                # 计算欧氏距离
                dist = np.linalg.norm(uav_pos - target_position)
                self.uav_distances[uav_id].append(dist)
        
        # 更新散点位置
        self.goose_scat.set_offsets(positions[1:])  # 普通雁只
        # 更新目标的位置
        self.target_position.set_offsets(target_position)
        self.encircle_circle.set_center(target_position)
        # if self.goose_swarm.current_phase == 'lions':
        #     for i in range(Config.UAV_NUM):
        #         self.circles[i].set_center(target_position)
        #         self.circles[i].set_radius(self.goose_swarm.Ri_list[i])
        
        # 更新标题（显示迭代次数）
        self.ax.set_title(f"无人机集群围捕模拟 | 迭代步骤：{frame}", fontsize=14)
        
        return self.goose_scat, self.target_position, self.encircle_circle
    
    def plot_distance_curve(self):
        """包围完成后，统一绘制距离-帧数关系曲线"""
        if len(self.frame_list) == 0:
            print("无距离数据可绘制！")
            return
        
        # time_step = Config.TIME_STEP
        # # 计算每秒的时间步数（0.2秒/帧 → 5帧/秒）
        # steps_per_second = int(1 / time_step)
        # total_steps = len(self.frame_list)  # 总时间步数（帧数）

        # # 初始化：存储每个无人机每秒的平均距离
        # uav_second_avg = {uav_id: [] for uav_id in range(Config.UAV_NUM)}
        # second_labels = []  # 横坐标：整数秒（1,2,3...）
        # # 遍历每一秒，计算该秒内所有时间步的平均距离
        # for second in range(1, total_steps // steps_per_second + 1):
        #     # 计算当前秒对应的时间步范围（整数索引，避免切片错误）
        #     start_idx = int((second - 1) * steps_per_second)
        #     end_idx = int(second * steps_per_second)
            
        #     # 处理最后一组不足5个时间步的情况
        #     if end_idx > total_steps:
        #         end_idx = total_steps
            
        #     # 遍历每个无人机，计算当前秒的平均距离
        #     for uav_id in range(Config.UAV_NUM):
        #         # 取出当前秒内该无人机的所有距离数据
        #         current_distances = self.uav_distances[uav_id][start_idx:end_idx]
        #         # 计算均值（跳过空数据）
        #         if len(current_distances) > 0:
        #             avg_dist = np.mean(current_distances)
        #             uav_second_avg[uav_id].append(avg_dist)
            
        #     # 记录横坐标（整数秒）
        #     second_labels.append(second)

        # 创建新的绘图窗口
        time = np.array(self.frame_list) * Config.TIME_STEP  # 核心：帧数→时间（秒）
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        # 遍历每个无人机绘制曲线
        for uav_id in range(Config.UAV_NUM):
            # 普通无人机用蓝色系，区分不同编号
            # ax_dist.plot(second_labels, uav_second_avg[uav_id], 
            #                 color=f'C{uav_id}', linewidth=1.5, label=f'无人机{uav_id+1}')
            
            ax_dist.plot(self.frame_list, self.uav_distances[uav_id], 
                            color=f'C{uav_id}', linewidth=1.5, label=f'无人机{uav_id+1}')
        
        ax_dist.set_xlabel("迭代时间(秒)", fontsize=12)
        ax_dist.set_ylabel("无人机-目标距离", fontsize=12)
        ax_dist.set_xlim(0,2000)
        ax_dist.set_title("各追捕者与目标的距离随时间变化曲线", fontsize=14)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend(loc='best', fontsize=10)
        # 保存图片（可选）
        if not os.path.exists('results'):
            os.makedirs('results')
        fig_dist.savefig('results/距离-帧数变化曲线.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_mean_distance(self):
        """绘制平均距离曲线"""
        if len(self.frame_list) == 0:
            print("无距离数据可绘制！")
            return
        
        total_frames = len(self.mean_uav_distances)  # 总更新次数
        second_average_distances = []
        second_labels = []
        interval_frames = int(1/Config.TIME_STEP)  # 每5帧为一个时间段（1秒）
        # 遍历每一秒（按5帧为一组）
        for second in range(1, total_frames // interval_frames + 1):
            # 计算当前秒对应的帧范围：比如第1秒对应0~4帧，第2秒对应5~9帧
            start_idx = int((second - 1) * Config.TIME_STEP)
            end_idx = int(second * Config.TIME_STEP)
            
            # 处理最后一组可能不足5帧的情况
            if end_idx > total_frames:
                end_idx = total_frames
            
            # 取出当前秒内的所有距离数据
            current_distances = self.mean_uav_distances[start_idx:end_idx]
            
            # 计算当前秒的平均距离（跳过空数据）
            if len(current_distances) > 0:
                avg_dist = np.mean(current_distances)
                second_average_distances.append(avg_dist)
                second_labels.append(second)  # 横坐标为整数秒（1,2,3...）
        # 创建新的绘图窗口
        fig_mean, ax_mean = plt.subplots(figsize=(10, 6))
        # ax_mean.plot(second_labels, second_average_distances, color='blue', linewidth=2)
        ax_mean.plot(self.frame_list, self.mean_uav_distances, color='blue', linewidth=2)
        ax_mean.set_xlabel("迭代时间(秒)", fontsize=12)
        ax_mean.set_ylabel("平均距离", fontsize=12)
        ax_mean.set_xlim(0, 2000)
        ax_mean.set_ylim(0, 120)
        ax_mean.set_title("平均距离随时间变化曲线", fontsize=14)
        # ax_mean.grid(True, alpha=0.3)
        # 保存图片（可选）
        if not os.path.exists('results'):
            os.makedirs('results')
        fig_mean.savefig('results/平均距离-时间变化曲线.png', dpi=150, bbox_inches='tight')
        plt.show()

    def show(self):
        """展示动画 + 围捕完成立即停止 + 视频保存不卡顿（Mac专用）"""
        # ===================== 自定义路径 + 时间戳文件名 =====================
        # 保存目录
        save_dir = "results"
        # 创建时间戳（和你CV2格式一致）
        create_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 拼接完整视频路径：时间戳_TERL.mp4
        video_path = os.path.join(save_dir, f"{create_timestamp}_WPA.mp4")
        # 自动创建文件夹
        os.makedirs(save_dir, exist_ok=True)
        # ==================================================================

        # 核心：动态生成帧数，检测到围捕完成就立刻停止
        def frame_generator():
            frame = 0
            # 最大限制20000帧，防止死循环
            while frame < 20000:
                # 实时检测是否围捕完成
                yield frame
                if self.goose_swarm.check_encircle(self.target.get_pos()):
                    break
                frame += 1

        # 创建动画：动态帧数，围捕完成自动终止
        ani = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=frame_generator,  # 关键：动态停帧
            interval=20,
            blit=False,
            repeat=False
        )

        print(f"🎥 开始录制视频，围捕完成后自动停止...")
        try:
            ani.save(
                filename=video_path,
                writer="ffmpeg",
                fps=30,
                dpi=150,
                # Mac双兼容编码，解决无法播放+卡顿
                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
            )
        finally:
            # 核心修复：强制关闭画布，释放进程，解决卡死
            plt.close(self.fig)

        print(f"✅ 视频保存完成！路径：{video_path}")
        # 绘制距离曲线
        self.plot_distance_curve()
        self.plot_mean_distance()
        
        # ===================== 返回视频路径（给Qt界面播放用）=====================
        return video_path
