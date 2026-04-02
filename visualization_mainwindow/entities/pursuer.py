# entities/pursuer.py
import numpy as np
# 第一步：补全Python的搜索路径，让解释器找到上级目录的config.py
import sys
import os,logging
from collections import defaultdict
import random

logging.basicConfig(
    level=logging.INFO,  # 日志级别（INFO 级别会记录所有 INFO/WARNING/ERROR）
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式（时间-级别-内容）
    handlers=[
        # logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('uav_radius_log.log', encoding='utf-8')  # 保存到文件（utf-8避免中文乱码）
    ]
)
logger = logging.getLogger(__name__)

# 获取当前脚本（pursuer.py）的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前脚本所在的目录（entities/）
current_dir = os.path.dirname(current_file)
# 获取父目录（WPA/），也就是config.py所在的目录
parent_dir = os.path.dirname(current_dir)
# 将父目录加入Python的搜索路径
sys.path.append(parent_dir)

# 第二步：正常导入config模块（此时Python能找到WPA/下的config.py）
from config import Config_all as Config

class Pursuer:
    """追捕者群体，封装单个追捕者的属性和行为"""
    def __init__(self, config, target_position):
        self.config = config
        self.positions = self._init_v_formation() #初始化雁群位置 self.generate_points_in_circle(num_points=Config.UAV_NUM, min_distance=20,center=(50, 50), radius=80)  
        self.velocities = self._init_velocity(target_position) #初始化个体速度大小     
        self.current_phase = ["geese"] * Config.UAV_NUM  # 当前阶段：初始为“雁群抵近”
        self.dt = Config.TIME_STEP
        self.Ri_list = [0 for _ in range(Config.UAV_NUM)]
        self.step = 0
        self.using_strategies = np.random.randint(0, 2, size=Config.UAV_NUM) # np.random.randint(0, 2, size=Config.UAV_NUM) ,self.init_strategies()

    def init_strategies(self):
        """
        初始化策略数组：一半1（合作）、一半0（背叛），随机打乱
        - 若UAV数量为奇数：多1个0（或1，可自选）
        """
        # 1. 计算1和0的数量
        uav_num = Config.UAV_NUM
        num_ones = uav_num // 2  # 1的数量（整除，保证一半）
        num_zeros = uav_num - num_ones  # 0的数量（偶数时等于num_ones，奇数时多1个）
        # 2. 构造基础数组（先0后1，顺序无关）
        strategies = np.concatenate([
            np.zeros(num_zeros, dtype=int),  # 生成指定数量的0
            np.ones(num_ones, dtype=int)     # 生成指定数量的1
        ])
        # 3. 随机打乱数组（核心：保证1和0随机分布）
        np.random.shuffle(strategies)
        return strategies
    
    def generate_points_in_circle(self, num_points=8, min_distance=20, center=(0, 0), radius=80):
        """
        在圆内生成指定数量的点，满足两两距离≥最小距离
        :param num_points: 点的数量（默认8）
        :param min_distance: 点之间的最小距离（默认20）
        :param center: 圆心坐标（默认(0,0)）
        :param radius: 圆半径（默认80，需足够大以容纳所有点）
        :return: 满足条件的点列表（每个点为np.array([x,y])）
        """
        points = []  # 存储最终满足条件的点
        points.append(center)
        max_attempts = 10000  # 最大尝试次数，避免死循环
        attempts = 0

        while len(points) < num_points + 1 and attempts < max_attempts:
            attempts += 1
            
            # 步骤1：生成圆内随机点（极坐标法，避免点集中在圆心）
            r = radius * np.sqrt(random.random())
            theta = 2 * np.pi * random.random()
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            candidate = np.array([x, y], dtype=np.float64)

            # 步骤2：检查候选点与所有已有点的距离≥min_distance
            valid = True
            for p in points:
                dist = np.linalg.norm(candidate - p)
                if dist < min_distance:
                    valid = False
                    break

            # 步骤3：满足条件则加入列表
            if valid:
                points.append(candidate)


        return np.array(points)
    
    def _init_v_formation(self):
        """初始化V字形编队位置"""
        positions = []
        # 虚拟领航者（V字顶点）初始位置
        leader_pos = np.array([150, 150])
        positions.append(leader_pos)  # 第0只为领航者
        
        # 两侧雁只对称分布
        for idx in range(1, Config.UAV_NUM + 1):
            # 核心规则1：奇数→左侧(-1)，偶数→右侧(1)
            side = -1 if idx % 2 == 1 else 1 
            # 核心规则2：计算所在排数（1,1,2,2,3,3... 左右对称）
            row = (idx + 1) // 2  
            # 3. 标准V形偏移计算
            # x偏移：向后退（远离领航者）
            x_offset = Config.ADJACENT_DIST * row * np.cos(Config.V_FORM_ANGLE / 2)
            # y偏移：左右展开（左负右正）
            y_offset = side * Config.ADJACENT_DIST * row * np.sin(Config.V_FORM_ANGLE / 2)
            # 计算x/y偏移（基于V字顶角）
            pos = leader_pos - np.array([x_offset, y_offset])
            positions.append(pos)
        
        return np.array(positions)
    
    def _init_velocity(self, target_position):
        velocities = []
        for i in range(Config.UAV_NUM):
            direction = target_position - self.positions[i + 1]
            direction_unit = direction / np.linalg.norm(direction)
            velocity = direction_unit * Config.GOOSE_SPEED
            velocities.append(velocity)
        return velocities
    
    def get_positions(self):
        return self.positions

    def get_velocities(self):
        return self.velocities
    
    
    def align(self):
        ave_velocity = [0.0, 0.0]
        for i in range(Config.UAV_NUM):
            ave_velocity += self.velocities[i]
        ave_velocity /= Config.UAV_NUM
        return ave_velocity
    
    def update_lead_position(self, target_position):
        to_target = target_position - self.positions[0]
        dist_to_target = np.linalg.norm(to_target)
        direction = to_target / dist_to_target
        self.positions[0] += direction * Config.GOOSE_SPEED * self.dt
    
    def update_uav_using_geese(self, target_position, i):
        to_target = target_position - self.positions[0]
        dist_to_target = np.linalg.norm(to_target)
        direction_target = to_target / dist_to_target
        #垂直于目标方向的向量
        row = (i + 1) // 2
        if (i + 1) % 2 == 1: #说明是奇数，位于左边
            dir_chuizhi = np.array([-direction_target[1], direction_target[0]])
            side_offset = row * Config.ADJACENT_DIST * np.sin(Config.V_FORM_ANGLE / 2) * dir_chuizhi 
        else:
            dir_chuizhi = np.array([direction_target[1], -direction_target[0]])
            side_offset = row * Config.ADJACENT_DIST * np.sin(Config.V_FORM_ANGLE / 2) * dir_chuizhi 
        forward_offset = - row * Config.ADJACENT_DIST * np.cos(Config.V_FORM_ANGLE / 2) * direction_target
        desired_position = self.positions[0] + forward_offset + side_offset
        
        
        # 规则1：飞向V形期望位置（聚合）
        cohesion = desired_position - self.positions[i + 1]
        if np.linalg.norm(cohesion) > 1e-3:
            cohesion = cohesion / np.linalg.norm(cohesion)  

        # 规则2：与邻居速度对齐（保证编队同向）
        alignment = np.zeros(2)
        neighbor_cnt = 0
        for j in range(Config.UAV_NUM):
            if j == i: continue
            d = np.linalg.norm(self.positions[j] - self.positions[i + 1])
            if d < Config.R_observation:  # 局部感知范围
                alignment += self.velocities[j]
                neighbor_cnt += 1
        if neighbor_cnt > 0:
            alignment = alignment / neighbor_cnt
            if np.linalg.norm(alignment) > 1e-3:
                alignment = alignment / np.linalg.norm(alignment)

        # 规则3：避免碰撞（分离）
        separation = np.zeros(2)
        for j in range(Config.UAV_NUM):
            if j == i: continue
            diff = self.positions[i + 1] - self.positions[j]
            d = np.linalg.norm(diff)
            if 0 < d < Config.R_SEPARATE:
                separation += diff / (d ** 2)
        
        # 权重可调
        desired_dir = 0.8 * cohesion + 0.1 * alignment + 0.1 * separation
        if np.linalg.norm(desired_dir) > 1e-3:
            desired_dir = desired_dir / np.linalg.norm(desired_dir)

        # 更新位置和速度（保持匀速）
        self.positions[i + 1] += desired_dir * Config.GOOSE_SPEED *self.dt
        self.velocities[i] = desired_dir * Config.GOOSE_SPEED

    def get_uav_mean_positions(self):
        '''用来获取狮群的中心位置'''
        return np.mean(self.positions, axis = 0) if len(self.positions) > 0 else np.array([0.0, 0.0])
    
    def calculate_Ri(self, distance_to_target, r_min, idx, k1 = 0.6, k2 = 0.2):
        '''计算单架无人机的动态围捕半径
        : param distance_to_target: 无人机到目标的距离，
        : param r_min: 最小包围半径，
        : param idx: 无人机编号(从1开始)，
        : param N: 无人机总数，
        '''
        def f_function(distance_to_target, alpha = 0.4):
            '''计算距离适配函数f(x),适配距离安全'''
            x = distance_to_target - r_min
            if x < 0:
                return 0.0
            D = Config.R_ENCIRCLE - Config.MIN_ENCIRCLE_RADIUS
            return 1 - np.exp(- alpha * x / D)

        def g_function(idx, N):
            '''平缓的编号适配函数：输出从-1到1连续渐变，无突变'''
            # 方案1：线性渐变（最简单，推荐）
            # 编号从1→N，输出从-1→1平滑过渡
            normalized_idx = idx / N  # 归一化到0~1
            return 2 * normalized_idx - 1         # 映射到-1~1

        f_x = f_function(distance_to_target)
        g_i = g_function(idx, Config.UAV_NUM)
        return Config.MIN_ENCIRCLE_RADIUS + (k1 * f_x + k2 * g_i) * (distance_to_target - Config.MIN_ENCIRCLE_RADIUS)
    

    def get_tangent_directions(self, uav_pos, target_pos, R_i, uav_id):
        """
        计算无人机到圆的两条严格几何切线方向，并根据编号分配（奇偶分工）
        :param uav_pos: 无人机位置 [x, y]
        :param target_pos: 目标位置 [x, y]
        :param R_i: 动态围捕半径（圆半径）
        :param uav_id: 无人机编号（从1开始）（用于分配切线方向）
        :return: 
            - selected_dir: 该无人机的飞行方向（27m/s）
            - tangent1: 逆时针切线方向（归一化）
            - tangent2: 顺时针切线方向（归一化）
        """
        # 1. 提取坐标
        Ox, Oy = target_pos
        Px, Py = uav_pos
        
        # 2. 计算OP向量和距离
        dx_OP = Px - Ox
        dy_OP = Py - Oy
        d_OP = np.hypot(dx_OP, dy_OP)
        
        # 4. 圆内情况：远离目标，无切线
        if d_OP <= R_i:
            dir_vec = np.array([dx_OP/d_OP, dy_OP/d_OP])
            return dir_vec
        
        # 5. 圆外情况：计算两条切线方向
        k = R_i / d_OP
        sqrt_term = np.sqrt(1 - k**2)  # 根号(1 - (Ri/d_OP)²)
        
        # ---------------- 5.1 计算逆时针切线（T1）方向 ----------------
        # 切点T1坐标
        T1_x = Ox + k * (dx_OP - dy_OP * sqrt_term)
        T1_y = Oy + k * (dy_OP + dx_OP * sqrt_term)
        # 无人机P到T1的向量 + 归一化
        dx_PT1 = T1_x - Px
        dy_PT1 = T1_y - Py
        d_PT1 = np.hypot(dx_PT1, dy_PT1)
        tangent1 = np.array([dx_PT1/d_PT1, dy_PT1/d_PT1])  # 归一化方向
        
        # ---------------- 5.2 计算顺时针切线（T2）方向 ----------------
        # 切点T2坐标
        T2_x = Ox + k * (dx_OP + dy_OP * sqrt_term)
        T2_y = Oy + k * (dy_OP - dx_OP * sqrt_term)
        # 无人机P到T2的向量 + 归一化
        dx_PT2 = T2_x - Px
        dy_PT2 = T2_y - Py
        d_PT2 = np.hypot(dx_PT2, dy_PT2)
        tangent2 = np.array([dx_PT2/d_PT2, dy_PT2/d_PT2])  # 归一化方向
        
        # ---------------- 5.3 仿生分工：分配切线方向 ----------------
        # 偶数号无人机 → 逆时针切线（右侧包抄）
        # 奇数号无人机 → 顺时针切线（左侧包抄）
        if uav_id % 2 == 0:
            selected_dir = tangent1
        else:
            selected_dir = tangent2
        
        return selected_dir
    
    def check_lion_complete(self, target_position, i):
        """
        判定狮群包围是否完成
        :param target_pos: 目标位置 (2,)
        :return: True/False（是否完成）
        """
        distance = np.linalg.norm(self.positions[i + 1] - target_position)
        if distance < Config.MIN_ENCIRCLE_RADIUS:
            return True
        else:
            return False



    def separation(self, i):
        range_in_separation = []
        direction = [0, 0]
        self_position = self.positions[i + 1]
        for j in range(Config.UAV_NUM):
            if j == i:
                continue
            distance = np.linalg.norm(self_position - self.positions[j + 1])
            if distance < Config.R_SEPARATE: #如果在分离半径范围内
                range_in_separation.append(j)
                direction += (self_position - self.positions[j + 1]) / distance
        if np.linalg.norm(direction) < 1e-6:
            return direction
        else:
            direction_unit = direction / np.linalg.norm(direction)
            return direction_unit
    
    def update_uav_using_lions(self, target_position, i):
        '''使用狮群算法更新uav的位置'''
        distance_to_target = np.linalg.norm(self.positions[i + 1] - target_position)
        Ri = self.calculate_Ri(distance_to_target, Config.MIN_ENCIRCLE_RADIUS, i + 1)
        self.Ri_list[i] = Ri
        # logger.info(f"无人机编号：{i + 1}，动态围捕半径：{Ri:.2f}") 
        direction_kaojin = self.get_tangent_directions(self.positions[i + 1], target_position, Ri, i + 1)
        direction_separation = self.separation(i)
        direction_total = direction_separation + direction_kaojin
        direction_total_unit = direction_total / np.linalg.norm(direction_total)
        self.velocities[i] = direction_total_unit * Config.LION_SPEED   # 根据编号调整速度（靠近目标的无人机飞得更慢）
        self.positions[i + 1] += self.velocities[i] * self.dt



        
    
    def update_uav_using_wolves(self, target_position, i):
        '''使用狼群策略进行围捕
        :param target_position: 目标位置
        :param assigned_points: 分配的围捕点列表
        '''

        
        for j in range(Config.UAV_NUM):
            if j == i:  # 跳过自身
                continue
            pos_j = np.array(self.positions[j + 1], dtype=np.float64)
            

        # --- 步骤1：找到离当前无人机最近的其他无人机 U_near ---
        pos_i = self.positions[i + 1]
        min_dist = np.inf
        near_pos = None
        for j in range(Config.UAV_NUM):
            if i == j:
                continue
            pos_j = self.positions[j + 1]
            dist = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
            if dist < min_dist:
                min_dist = dist
                near_pos = pos_j

        # --- 步骤2：计算当前无人机到目标的距离 dis ---
        # print(f"near_pos: {near_pos}")
        dx = pos_i[0] - target_position[0]
        dy = pos_i[1] - target_position[1]
        dis = np.hypot(dx, dy)

        
        # --- 步骤3：根据 dis 选择控制策略 ---
        if dis <= Config.MIN_ENCIRCLE_RADIUS:
            # 策略二：在攻击半径内，远离目标（保证安全）
            # 方向 = 从目标指向当前无人机的单位向量
            dir_vec_away = np.array([dx / dis, dy / dis])
            #然后我要给一个切向的方向，这个方向是远离最近切点的方向
            dir_vec_ni_unit = np.array([-dy / dis, dx / dis])
            dir_vec_shun_unit = np.array([dy / dis, -dx / dis])
            near_uav = near_pos - pos_i
            near_uav_unit = near_uav / np.linalg.norm(near_uav)
            #计算点积
            dot1 = np.dot(dir_vec_ni_unit, near_uav_unit)
            dot2 = np.dot(dir_vec_shun_unit, near_uav_unit)

            #选择远离友机的方向(选择点积小的那个)
            if dot1 > dot2:
                dir_vec_qiexiang = dir_vec_shun_unit
            elif dot2 > dot1:
                dir_vec_qiexiang = dir_vec_ni_unit
            else:
                dir_vec_qiexiang = random.choice([dir_vec_ni_unit, dir_vec_shun_unit])
            dir_vec = dir_vec_away + dir_vec_qiexiang
            dir_vec_norm = dir_vec / np.linalg.norm(dir_vec)
        else:
            
            # 策略一：在攻击半径外，计算两条切线，选远离最近友机的方向
            # 1. 计算从目标到当前无人机的向量 OP
            OP = np.array([dx, dy])
            d_OP = dis
            # 2. 计算两条切线方向（修正后的严格几何切线）
            r = Config.MIN_ENCIRCLE_RADIUS
            sinθ = r / d_OP
            cosθ = np.sqrt(1 - sinθ**2) if d_OP > r else 0  # 确保圆外点（d_OP > r）

            # 单位化OP向量
            u_OP = OP / d_OP if d_OP > 1e-6 else np.array([1, 0])

            # 逆时针切线切点T1（以圆心为基准判断是逆时针）
            OT1_x = r * (u_OP[0] * cosθ - u_OP[1] * sinθ)
            OT1_y = r * (u_OP[0] * sinθ + u_OP[1] * cosθ)
            T1_x = target_position[0] + OT1_x
            T1_y = target_position[1] + OT1_y
            v1 = np.array([T1_x - pos_i[0], T1_y - pos_i[1]])
            v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-6 else v1

            # 顺时针切线切点T2（以圆心为基准判断是顺时针）
            OT2_x = r * (u_OP[0] * cosθ + u_OP[1] * sinθ)
            OT2_y = r * (-u_OP[0] * sinθ + u_OP[1] * cosθ)
            T2_x = target_position[0] + OT2_x
            T2_y = target_position[1] + OT2_y
            v2 = np.array([T2_x - pos_i[0], T2_y - pos_i[1]])
            v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-6 else v2

            # 3. 选择远离最近友机 near_pos 的方向
            # 计算两个切线方向到 near_pos 的距离
            #计算无人机指向最近邻居的方向
            # print(f"无人机{i + 1}：最近友机位置={near_pos}")
            PU = near_pos - pos_i
            PU = PU / np.linalg.norm(PU)
            #计算点积
            dot1 = np.dot(v1, PU)
            dot2 = np.dot(v2, PU)


            #选择远离友机的方向
            if dot1 > dot2:
                dir_vec_norm = v2
            elif dot2 > dot1:
                dir_vec_norm = v1
        
        separate_direction = self.separation(i)
        if dis > Config.MIN_ENCIRCLE_RADIUS * 1.01:
            direction_to_target = target_position - self.positions[i + 1]
            direction_to_target_unit = direction_to_target / np.linalg.norm(direction_to_target)
            dir_total = separate_direction + dir_vec_norm + 0.3 * direction_to_target_unit
            dir_total_unit = dir_total / np.linalg.norm(dir_total)
        else:
            dir_total = separate_direction + dir_vec_norm
            dir_total_unit = dir_total / np.linalg.norm(dir_total) 
        self.velocities[i] = dir_total_unit * Config.WOLF_SPEED
        self.positions[i + 1] += self.velocities[i] * self.dt  



    def detect_uavs_in_observation(self, uav_positions, uav_wolves_indices, observation_radius):
        """
        检测每架无人机观测半径内的友机，记录编号
        :param uav_positions: 所有位于狼群围捕阶段的无人机位置数组，shape=(len(uav_positions), 2)，每行=[x,y]
        :param uav_wolves_indices: 位于狼群围捕阶段的无人机编号列表
        :param observation_radius: 观测半径（米）
        :return: observed_uavs: 字典，格式 {本机编号: [友机编号]}
        """
        observed_uavs = {}  # 存储最终结果
        
        # 步骤1：计算无人机间的距离矩阵（复用你之前的高效矩阵逻辑）
        # 向量化计算，避免嵌套循环，效率更高
        pos_expand = np.array(uav_positions)[:, np.newaxis, :]  # 扩展维度：(len(uav_positions),1,2)
        diff = pos_expand - np.array(uav_positions)             # 计算位置差：(len(uav_positions), len(uav_positions), 2)
        dist_matrix = np.linalg.norm(diff, axis=2)    # 距离矩阵：(len(uav_positions), len(uav_positions))

        # 步骤2：逐机筛选观测范围内的友机
        for i, self_idx in enumerate(uav_wolves_indices):
            # 本机到所有友机的距离（行）
            self_distances = dist_matrix[i]
            # 存储当前无人机观测到的友机id
            observed_friends_idx_list = []
            
            # 遍历所有友机（排除自身）
            for j, friend_idx in enumerate(uav_wolves_indices):
                if self_idx == friend_idx:
                    continue  # 跳过自己
                
                distance = self_distances[j]
                
                # 筛选：距离小于观测半径
                if distance < observation_radius:
                    observed_friends_idx_list.append(friend_idx)
            
            # 存储当前无人机的观测结果
            observed_uavs[self_idx] = observed_friends_idx_list
        
        return observed_uavs

    def update_uav_using_wolves_using_observations(self, target_position, i):
        '''基于更全的观测信息进行判断下一步的决策'''

        def count_friends(self_pos, dir_shun, dir_ni, friend_list, near_pos):
            """
            统计指定区域内的同伴数量（核心判断逻辑）
            :param self_pos: 当前无人机位置，如pos_i → 图中的U1坐标
            :param target_dir: 切线方向单位向量，如v1/v2 → 图中的v1、v2
            :param friend_positions: 所有同伴的编号列表（已筛选感知半径内）
            :return: count: 该扇形区域内的同伴数量
            """
            count_ni = 0  # 初始化逆时针方向计数
            count_shun = 0 # 初始化顺时针方向计数
            
            # 遍历每个同伴
            for idx in friend_list:
                # 步骤1：计算「当前无人机→同伴」的向量，并归一化
                friend_vec = self.positions[idx] - self_pos  # 向量：U1→U2/U3/U4/U5
                friend_dist = np.linalg.norm(friend_vec)
                
                # 容错：排除距离为0的情况（避免除以0，理论上已筛选过）
                if friend_dist < 1e-6:
                    continue
                friend_dir = friend_vec / friend_dist  # 归一化为单位向量
                
                dot_product_shun = np.dot(dir_shun, friend_dir)
                dot_product_ni = np.dot(dir_ni, friend_dir)

                # 步骤3：核心判断：两个切线夹角进行比较
                if dot_product_shun < dot_product_ni:
                    count_ni += 1
                else:
                    count_shun += 1
            
            if count_ni > count_shun:
                return dir_shun
            elif count_ni < count_shun:
                return dir_ni
            else:#如果两边的邻居数量相等，那么就找最近的邻居，远离最近的邻居
                PU = np.array([0.0, 0.0])
                if near_pos is not None:
                    PU = near_pos - pos_i
                    PU = PU / np.linalg.norm(PU)
                #计算点积
                dot_shun = np.dot(dir_shun, PU)
                dot_ni = np.dot(dir_ni, PU)

                #选择远离友机的方向
                if dot_shun > dot_ni:
                    dir_vec_next = dir_ni
                elif dot_ni > dot_shun:
                    dir_vec_next = dir_shun
                else:
                    dir_vec_next = random.choice([v1, v2])
                return dir_vec_next
        
        
        uav_positions_wolves_phase_positions = []
        uav_positions_wolves_phase_idx = []
        for phase_i in range(Config.UAV_NUM):
            if self.current_phase[phase_i] == 'wolves':
                uav_positions_wolves_phase_positions.append(self.positions[phase_i + 1])
                uav_positions_wolves_phase_idx.append(phase_i + 1)
        #寻找所有观测半径范围内的友机组合
        observed_uavs_using_wolves = self.detect_uavs_in_observation(uav_positions_wolves_phase_positions, uav_positions_wolves_phase_idx, Config.R_observation)#得到的是一个字典，键代表编号，值也是一个字典（其中键代表观测半径范围内的id，值代表距离）

        
        # --- 步骤1：找到离当前无人机最近的其他无人机 U_near ---
        pos_i = self.positions[i + 1]
        min_dist = np.inf
        near_pos = None
        for j in uav_positions_wolves_phase_idx:
            if i == j - 1:  # 跳过自身（注意编号和索引的关系）
                continue
            pos_j = self.positions[j]
            dist = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
            if dist < min_dist:
                min_dist = dist
                near_pos = pos_j

        dx = self.positions[i + 1][0] - target_position[0]
        dy = self.positions[i + 1][1] - target_position[1]
        dis = np.hypot(dx, dy)

        if dis <= Config.MIN_ENCIRCLE_RADIUS:
            # 策略二：在攻击半径内，远离目标（保证安全）
            # 方向 = 从目标指向当前无人机的单位向量
            dir_vec_away = np.array([dx / dis, dy / dis])
            #然后我要给一个切向的方向，这个方向是远离最近切点的方向
            dir_vec_ni_unit = np.array([-dy / dis, dx / dis])
            dir_vec_shun_unit = np.array([dy / dis, -dx / dis])
            # --- 步骤1：找到离当前无人机最近的其他无人机 U_near ---
            min_dist = np.inf
            near_pos = None
            for uav_j in range(Config.UAV_NUM):
                if i == uav_j:  # 跳过自身（注意编号和索引的关系）
                    continue
                pos_j = self.positions[uav_j + 1]
                dist = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
                if dist < min_dist:
                    min_dist = dist
                    near_pos = pos_j

            near_uav = near_pos - pos_i
            near_uav_unit = near_uav / np.linalg.norm(near_uav)
            #计算点积
            dot1 = np.dot(dir_vec_ni_unit, near_uav_unit)
            dot2 = np.dot(dir_vec_shun_unit, near_uav_unit)

            #选择远离友机的方向(选择点积小的那个)
            if dot1 > dot2:
                dir_vec_qiexiang = dir_vec_shun_unit
            elif dot2 > dot1:
                dir_vec_qiexiang = dir_vec_ni_unit
            else:
                dir_vec_qiexiang = random.choice([dir_vec_ni_unit, dir_vec_shun_unit])

            dir_vec_next = dir_vec_away + dir_vec_qiexiang
            dir_vec_next = dir_vec_next / np.linalg.norm(dir_vec_next)

        else:
            pos_i = self.positions[i + 1]
            # 策略一：在攻击半径外，计算两条切线，选远离最近友机的方向
            # 1. 计算从目标到当前无人机的向量 OP
            OP = np.array([dx, dy])
            d_OP = dis
            # 2. 计算两条切线方向（修正后的严格几何切线）
            r = Config.MIN_ENCIRCLE_RADIUS
            sinθ = r / d_OP
            cosθ = np.sqrt(1 - sinθ**2) if d_OP > r else 0  # 确保圆外点（d_OP > r）

            # 单位化OP向量
            u_OP = OP / d_OP if d_OP > 1e-6 else np.array([1, 0])

            # 逆时针切线切点T1（以圆心为基准判断是逆时针）
            OT1_x = r * (u_OP[0] * cosθ - u_OP[1] * sinθ)
            OT1_y = r * (u_OP[0] * sinθ + u_OP[1] * cosθ)
            T1_x = target_position[0] + OT1_x
            T1_y = target_position[1] + OT1_y
            v1 = np.array([T1_x - pos_i[0], T1_y - pos_i[1]])
            v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 1e-6 else v1

            # 顺时针切线切点T2（以圆心为基准判断是顺时针）
            OT2_x = r * (u_OP[0] * cosθ + u_OP[1] * sinθ)
            OT2_y = r * (-u_OP[0] * sinθ + u_OP[1] * cosθ)
            T2_x = target_position[0] + OT2_x
            T2_y = target_position[1] + OT2_y
            v2 = np.array([T2_x - pos_i[0], T2_y - pos_i[1]])
            v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 1e-6 else v2
        
            if len(observed_uavs_using_wolves[i + 1]) != 0: #i + 1周围有邻居
                dir_vec_next = count_friends(pos_i, v2, v1, observed_uavs_using_wolves[i + 1], near_pos)
            else: #id周围没有邻居，就找最近邻的邻居，远离那个最近的邻居
                PU = np.array([0.0, 0.0])
                if near_pos is not None:
                    PU = near_pos - pos_i
                    PU = PU / np.linalg.norm(PU)
                #计算点积
                dot1 = np.dot(v1, PU)
                dot2 = np.dot(v2, PU)

                #选择远离友机的方向
                if dot1 > dot2:
                    dir_vec_next = v2
                elif dot2 > dot1:
                    dir_vec_next = v1
                else:
                    dir_vec_next =random.choice([v1, v2])
            
        separation_direction = self.separation(i)
        dir_vec = dir_vec_next + separation_direction
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        if dis > Config.MIN_ENCIRCLE_RADIUS * 1.01:
            direction_to_target = target_position - self.positions[i + 1]
            direction_to_target_unit = direction_to_target / np.linalg.norm(direction_to_target)
            dir_total = separation_direction + dir_vec + 0.3 * direction_to_target_unit
            dir_total_unit = dir_total / np.linalg.norm(dir_total)
        else:
            dir_total = separation_direction + dir_vec
            dir_total_unit = dir_total / np.linalg.norm(dir_total) 

        self.velocities[i] = dir_total_unit * Config.WOLF_SPEED 
        self.positions[i + 1] += self.velocities[i] * Config.TIME_STEP

            


    def update_all_uavs(self, target_position):
        distances = np.linalg.norm(self.positions - target_position, axis=1)
        self.update_lead_position(target_position)  # 更新领头无人机位置
        for i in range(Config.UAV_NUM):
            if self.current_phase[i] == 'geese':
                #执行雁阵靠近
                self.update_uav_using_geese(target_position, i)
                #判定是否进入狮群阶段的条件：所有无人机都进入包围半径
                if distances[i + 1] <= Config.R_ENCIRCLE:
                    self.current_phase[i] = 'lions'
                    print("【阶段切换】所有无人机进入包围半径，切换到狮群阶段")
            
            #阶段2:狮群包围
            elif self.current_phase[i] == 'lions':
                #判定：狮群包围是否完成
                if not self.check_lion_complete(target_position, i):
                    self.update_uav_using_lions(target_position, i)
                #判定：狮群完成包围
                else:
                    self.current_phase[i] = 'wolves'
                    print("【阶段切换】狮群包围完成，切换到狼群阶段")
            #阶段3: 狼群围攻
            elif self.current_phase[i] == 'wolves':
                uavs_strategies = self.using_strategies.copy()

                uav_positions_wolves_phase_positions = []
                uav_positions_wolves_phase_idx = []
                for j in range(Config.UAV_NUM):
                    if self.current_phase[j] == 'wolves':
                        uav_positions_wolves_phase_positions.append(self.positions[j + 1])
                        uav_positions_wolves_phase_idx.append(j + 1)

                observation_network_using_wolves = self.detect_uavs_in_observation(uav_positions_wolves_phase_positions, uav_positions_wolves_phase_idx, Config.R_observation)#得到达到狼群阶段的观测网络

                sum_friend = 0
                for _, observed_friends in observation_network_using_wolves.items():
                    sum_friend += len(observed_friends)
                average_friend = sum_friend / len(uav_positions_wolves_phase_idx)
                eta = 0.3
                r = eta * (average_friend + 1)
                total_profit_list = [0 for _ in range(len(uav_positions_wolves_phase_idx))]
                #开始计算所有节点的收益
                for idx, observed_friends in observation_network_using_wolves.items():
                    self_idx_index = uav_positions_wolves_phase_idx.index(idx)
                    total_profit = 0
                    if len(observed_friends) != 0:
                            #开始计算奖池，如果选择策略1，就往奖池中加1，如果选择策略0，就不加。
                            if uavs_strategies[idx - 1] == 1:
                                total_profit = 1/(len(observed_friends) + 1)
                            for friend_idx in observed_friends:
                                if uavs_strategies[friend_idx - 1] == 1:
                                    total_profit += 1/ (len(observed_friends) + 1)
                            total_profit *= r #这里就计算好了每个无人机集群中的奖池了
                            #接下来开始计算每个无人机的收益了
                            total_profit_list[self_idx_index] += total_profit /(len(observed_friends) + 1) - 1 * uavs_strategies[idx - 1] #计算自己的收益
                            # for friend_idx in observed_friends:
                            #     friend_idx_index = uav_positions_wolves_phase_idx.index(friend_idx)
                            #     total_profit_list[friend_idx_index] += total_profit /(len(observed_friends) + 1) - 1 * uavs_strategies[friend_idx - 1] #计算朋友的收益
                    #以上就计算完成了所有节点的收益
                #接下来开始根据收益更新每个节点的策略了
                # ave = 0
                # for profit in total_profit_list:
                #     ave += profit
                # ave /= len(total_profit_list)
                # print(f"无人机的策略列表：{uavs_strategies}")
                # print(f"无人机所有的收益列表：{total_profit_list}")
                # print(f"所有无人机的平均收益为:{ave}")
                # for idx, observed_friends in observation_network_using_wolves.items():
                    # if len(observed_friends) != 0:
                    #     max_profit = -np.inf
                    #     max_idx = -1
                    #     for friend_idx in observed_friends:
                    #         if max_profit < total_profit_list[friend_idx - 1]:
                    #             max_profit = total_profit_list[friend_idx - 1]
                    #             max_idx = friend_idx
                                
                    #     if total_profit_list[idx - 1] < max_profit: #如果朋友的收益更高，就模仿朋友的策略
                    #         self.using_strategies[idx - 1] = uavs_strategies[max_idx - 1]
                Aspiration = 0.7
                if random.random() < 1/ (1 + np.exp((total_profit_list[self_idx_index] - Aspiration * len(observed_friends)) / 0.1)):
                    self.using_strategies[i] = 1 - self.using_strategies[i]
                #策略选择结束，就开始更新动作了
                self.using_strategies[0] = 0
                if self.using_strategies[i] == 0:
                    self.update_uav_using_wolves(target_position, i)
                else:
                    self.update_uav_using_wolves_using_observations(target_position, i)
                

        self.step += 1
        print(f"self.step的值为:{self.step}")




    def get_all_agents_nearest_distances(self, all_agents_pos, eps: float = 1e-8) -> np.ndarray:
        """
        批量计算每个智能体到其最近邻智能体的欧氏距离
        :param all_agents_pos: 所有智能体的位置数组，shape=(N, 2)，N为智能体总数，2为x/y坐标
        :param eps: 浮点精度极小值，用于过滤自身（避免自身距离为0的干扰）
        :return: 最近邻距离数组，shape=(N,)，第i个元素为第i个智能体的最近邻距离
                若仅1个智能体，返回[np.inf]
        """
        # 类型转换与维度校验，确保输入合法（适配列表/数组输入，避免报错）
        if all_agents_pos.ndim != 2 or all_agents_pos.shape[1] != 2:
            raise ValueError(f"输入必须是二维数组，shape=(N, 2)，当前输入shape={all_agents_pos.shape}")
        n_agents = all_agents_pos.shape[0]
        
        # 边界情况：仅1个智能体，无最近邻，返回无穷大
        if n_agents == 1:
            return np.array([np.inf])
        
        # 向量化计算所有智能体间的欧氏距离矩阵（shape=(N, N)）
        # dist_matrix[i,j] 表示第i个智能体到第j个智能体的距离
        dist_matrix = np.linalg.norm(all_agents_pos[:, np.newaxis] - all_agents_pos, axis=2)
        
        # 过滤自身距离：将对角线（i=j）的距离设为无穷大（避免选到自己）
        # 同时处理浮点精度误差（距离<eps判定为自身）
        dist_matrix[dist_matrix < eps] = np.inf
        # 对每一行取最小值，即每个智能体的最近邻距离
        nearest_distances = np.min(dist_matrix, axis=1)
        # print(f"nearest_distance的内容为:{nearest_distances}")
        # print(f"最近邻距离分布：min={nearest_distances.min():.2f}, max={nearest_distances.max():.2f}, mean={nearest_distances.mean():.2f}")
        
        return nearest_distances

        
    def check_encircle(self, target_pos):
        """
        改进版包围判断：
        1. 平均值与理想值的差值 ≤ 平均值阈值
        2. 所有单夹角与理想值的偏差 ≤ 单夹角阈值
        两者同时满足才判定为包围成功
        """
        target_pos = np.array(target_pos, dtype=np.float64)
        valid_uavs = []
        
        # 步骤1：筛选有效无人机
        for uav_pos in self.positions:
            uav_pos = np.array(uav_pos, dtype=np.float64)
            dist = np.linalg.norm(uav_pos - target_pos)
            if dist < 1e-6:  # 跳过与目标重合的无人机
                continue
            if dist < Config.MIN_ENCIRCLE_RADIUS + Config.check_encircle_threshold and dist > Config.MIN_ENCIRCLE_RADIUS - Config.check_encircle_threshold:
                valid_uavs.append(uav_pos)
        
        uav_count = len(valid_uavs)
        if uav_count != Config.UAV_NUM:
            return False
        
        # 步骤2：计算极角并排序（0~360度）
        polar_angles = []
        for uav in valid_uavs:
            vec = uav - target_pos
            angle_rad = np.arctan2(vec[1], vec[0])
            angle_deg = np.rad2deg(angle_rad)
            angle_deg = angle_deg if angle_deg >= 0 else angle_deg + 360.0
            polar_angles.append(angle_deg)
        
        sorted_indices = np.argsort(polar_angles)
        sorted_angles = [polar_angles[i] for i in sorted_indices]
        
        # 步骤3：计算相邻夹角（闭环）
        adjacent_angles = []
        ideal_angle = 360.0 / uav_count  # 理想均匀夹角
        for i in range(uav_count):
            curr = sorted_angles[i]
            next_ang = sorted_angles[(i + 1) % uav_count]
            diff = next_ang - curr
            if diff < 0:
                diff += 360.0
            adjacent_angles.append(diff)
        
        # 步骤4：双重判断（核心修复）
        avg_adjacent = np.mean(adjacent_angles)
        # 4.1 平均值偏差判断
        avg_diff = abs(avg_adjacent - ideal_angle)
        avg_ok = avg_diff <= Config.ENCIRCLE_AVG_THRESHOLD
        
        # 4.2 单夹角偏差判断（每个夹角与理想值的偏差≤阈值）
        single_diffs = [abs(ang - ideal_angle) for ang in adjacent_angles]
        max_single_diff = max(single_diffs)
        single_ok = max_single_diff <= Config.ENCIRCLE_SINGLE_THRESHOLD
        
        # 最终判断：两者都满足才成功
        is_encircled = avg_ok and single_ok
        
        return is_encircled
