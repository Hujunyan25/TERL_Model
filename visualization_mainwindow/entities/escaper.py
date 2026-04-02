# entities/escaper.py
import numpy as np
from config import Config_all as Config 

# ---------------------- 目标类（文档3.2.2逃逸策略） ----------------------
class Target:
    def __init__(self, config):
        self.config = config
        self.speed = Config.TARGET_Velocity  # 目标速度（m/s）
        self.position = Config.TARGET_POS.copy()  # 目标初始位置
        self.dt = Config.TIME_STEP  # 每帧时间间隔（s）
        self.x_min = Config.x_min
        self.x_max = Config.x_max
        self.y_min = Config.y_min
        self.y_max = Config.y_max
        self.boundary_k = 10.0 
        self.k_rep_boundary = 8000.0  # 边界斥力系数（可调节）
        self.d0_uav = 80.0
        self.d0_boundary = 70.0
        self.k_rep_uav = 2.5
        # 每帧实际移动弧长
        # self.per_frame_arc_length = self.speed * self.dt
        # 初始运动方向
        self.direction = np.array([np.random.randn(), np.random.randn()])
        self.direction /= np.linalg.norm(self.direction)

    def boundary_force(self):
        """和你原APF完全一样的边界斥力"""
        force = np.zeros(2)

        def calc(d, positive):
            if d < self.d0_boundary:
                mag = self.k_rep_boundary * ((1/abs(d)) - (1/self.d0_boundary)) / (abs(d)**2)
                return mag if positive else -mag
            return 0

        force[0] += calc(self.position[0] - self.x_min, True)
        force[0] += calc(self.x_max - self.position[0], False)
        force[1] += calc(self.position[1] - self.y_min, True)
        force[1] += calc(self.y_max - self.position[1], False)
        return force * self.boundary_k
    
    
    def compute_target_escape_force(self, uav_positions):
        '''
        人工势场法：目标用来躲避无人机
        '''
        escape_force = np.array([0.0, 0.0])

        #对每架无人机计算斥力
        for pos in uav_positions:
            dx = pos[0] - self.position[0]
            dy = pos[1] - self.position[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            # 当距离小于阈值的时候产生斥力
            if dist < self.d0_uav:
                # 人工势场斥力公式
                rep_force = self.k_rep_uav * (1 / dist - 1 / self.d0_uav) / (dist ** 2)
                # 方向：从追捕无人机指向目标
                escape_force -= np.array([dx, dy]) / dist * rep_force
        return escape_force
    
    def compute_escape_direction(self, uav_positions):
        # 1. 追捕无人机的斥力
        rep_force = self.compute_target_escape_force(uav_positions)
        # 2. 边界斥力
        boundary_force = self.boundary_force()
        # 3. 合力
        total_force = rep_force + boundary_force
        
        if np.linalg.norm(total_force) > 1e-3:
            # 有斥力 → 改变方向躲避
            new_dir = total_force / np.linalg.norm(total_force)
            # 平滑转向（不突变）
            self.direction = 0.6 * self.direction + 0.4 * new_dir
            self.direction /= np.linalg.norm(self.direction)

        return self.direction.copy()

    
    # def update_position(self, frame, pursuer_velocities):
    #     """
    #     沿直线匀速更新位置（保证实际速度=20m/s）
    #     :param frame: 动画迭代帧（仅用于区分帧，不参与计算）
    #     :return: 更新后的目标位置 (x, y)
    #     """
    #     # ---------------- 核心配置：直线运动方向（可按需修改） ----------------
    #     # 方式2：沿指定角度运动（比如45°，需将角度转为单位向量）
    #     move_angle = 0  # 运动方向与x轴夹角（弧度，45°，可自定义）
    #     direction = np.array([np.cos(move_angle), np.sin(move_angle)], dtype=np.float64)

    #     # ---------------- 直线位移计算（核心简化） ----------------
    #     # 直线运动时，per_frame_arc_length 直接等于每帧的直线位移距离
    #     # 单位方向向量 × 每帧位移距离 = x/y方向的增量
    #     delta_x = direction[0] * self.per_frame_arc_length 
    #     delta_y = direction[1] * self.per_frame_arc_length 

    #     total_theta = [0, 0]
    #     for vel in pursuer_velocities:
    #         total_theta += vel
    #     if total_theta[0] > 0:
    #         total_theta = np.arctan2(total_theta[1], total_theta[0])
    #     else:
    #         total_theta = np.arctan2(total_theta[1], total_theta[0]) + np.pi
    #     direction = np.array([np.cos(total_theta), np.sin(total_theta)])
    #     velocity = direction * self.speed
    #     self.position += velocity * self.dt

    #     return self.position.copy()


    def update_position(self, frame, uavs_positions):
        '''目标无人机移动'''
        direction = self.compute_escape_direction(uavs_positions)
        self.position += direction * self.speed * self.dt
    
    def get_pos(self):
        return self.position.copy()