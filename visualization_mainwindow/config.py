# config.py
import numpy as np

# ---------------------- 配置类（贴合文档参数定义） ----------------------
class Config_all:
    """全局配置，参考文档3.4实验参数"""

    MAP_X_RANGE = (0, 550)  # 地图范围（x轴）
    MAP_Y_RANGE = (0, 550) # 地图范围（y轴）
    x_min = 0.0
    x_max = 500.0
    y_min = 0.0
    y_max = 500.0

    TARGET_POS = np.array([200, 200], dtype=np.float64)
    TARGET_Velocity = 2 # 目标速度（m/s）
    # TARGET_SIN_AMPLITUDE = 50.0  # 目标正弦运动幅度（m）
    # TARGET_SIN_FREQUENCY = 0.005  # 目标正弦运动频率（Hz）

    # 无人机集群参数
    UAV_NUM = 4  # 围捕无人机数量
    #雁群的参数
    R_SEPARATE = 1 # 分离半径（确保无人机能在0.33s左右能够响应）
    # R_ALIGN = 16 # 调整半径（大约是R_SEPARATE的2倍）
    # R_COHERENT = 24 # 聚合半径（大约是R_ALIGN的1.5倍）
    GOOSE_SPEED = 3.0 # 雁群最大速度（m/s）
    # SAFE_DIST = 8  # 雁只间安全距离（避免碰撞）
    ADJACENT_DIST = 5  # 相邻雁只的期望距离
    V_FORM_ANGLE = np.deg2rad(110) # V字顶角（110°，可调整开合程度）
    R_ENCIRCLE = 25 # 狮群围捕半径（m）(在pursuer这个类中设置为250)
    # SPEED_DECEL_THRESH = 20 # 减速阈值距离（m）,<=20的时候该值开始线性减速
    # SPEED_DAMP_THRESH = 1.0 #阻尼阈值距离（m）,<=1.0的时候置为目标速度收尾

    # 狮群参数
    LION_SPEED = 2.7  # 狮子速度（m/s）
    # LION_ATTACK = 100.0  # 狼群攻击距离（m）

    # 狼群参数
    R_observation = 5 # 狼群个体的观测半径
    WOLF_SPEED = 2.7  # 狼群速度（m/s）
    # ATTACK_RANGE = 10  # 猛狼攻击距离（m）
    # P_RECALL = 0.3  # 召回概率
    # P_ATTACK = 0.4  # 攻击概率
    # ENCIRCLE_WOLF_NUM = 4 #包围数量
    # ENCIRCLE_QUADRANT_NUM = 4 #包围象限数量

    # 最小围捕半径
    MIN_ENCIRCLE_RADIUS = 6.0 # 最小围捕半径（m）
    MAX_ENCIRCLE_ANGLE = np.pi * 2/9   # 最大围捕角度（rad）
    #围捕角度阈值变化范围
    ENCIRCLE_ANGLE_THRESHOLD = np.pi * 1/9 # 最大围捕角度阈值（rad）

    # Distance_Threshold = 5 # 距离常数

    check_encircle_threshold = 0.5 # 包围检查距离阈值（m），当无人机与目标的距离小于该值时认为包围成功

    # MIN_UAV_DISTANCE = 10.0 # 无人机间最小安全距离（m）
    # 椭圆的长轴半长
    # a = MIN_ENCIRCLE_RADIUS - 30 
    #椭圆的短轴半长
    # b = MIN_ENCIRCLE_RADIUS - 15

    TIME_STEP = 0.1 # 时间步长（s）(根据无人机速度和安全距离调整，确保每步移动不会过大导致碰撞或过小导致过多迭代)
    ENCIRCLE_AVG_THRESHOLD = 360/UAV_NUM # 围捕平均角度阈值（度），当所有无人机与目标的平均角度小于该值时认为包围完成
    ENCIRCLE_SINGLE_THRESHOLD = 10  # 围捕单个角度阈值（度），当每个无人机与目标的角度小于该值时认为包围完成