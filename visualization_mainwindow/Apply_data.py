import sys,os,argparse
# 获取 model_eval.py 所在目录的绝对路径
model_eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将该目录添加到 sys.path
sys.path.insert(0, model_eval_dir)
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
# 导入转换后的 ui 类
from uav_mainwindow import Ui_MainWindow 
from model_eval import generate_video
from visualization_mainwindow.WPA_visualizer import Visualizer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from pyqtgraph import mkPen




class Apply(QMainWindow, Ui_MainWindow):
    def __init__(self, pursuer_num=None, env_length=None, env_width=None, obstacle_num=None):
        super().__init__()
        self.setupUi(self)
        # 1. 初始化媒体播放器
        self.media_player_traditional_control = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.video_widget_traditional_control = QVideoWidget(self)
        # self.video_widget_traditional_control.setParent(self.traditional_control_display)
        # self.video_widget_traditional_control.resize(self.traditional_control_display.size())
        self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget(self)  # 视频显示控件
        self.energy_pursuer_graph_widget.setBackground('w')
        self.imgDisplayLabel.setStyleSheet("background: white;")  # 背景设为白色
        self.video_widget.setParent(self.imgDisplayLabel)  # 放到 QLabel 里
        self.video_widget.resize(self.imgDisplayLabel.size())  # 适配大小

        # 连接 Apply 按钮的点击信号到自定义槽函数
        self.ApplyButton.clicked.connect(self.on_apply_clicked)
        # 关联播放器与显示控件
        self.media_player.setVideoOutput(self.video_widget)
        #界面切换按钮
        self.simulation_panel.clicked.connect(self.show_page1)
        self.action_statistics_panel.clicked.connect(self.show_page2)
        self.baseline_compare_panel.clicked.connect(self.show_page3)
        self.ablation_experiment_panel.clicked.connect(self.show_page4)
        self.baseline_collision_button.clicked.connect(self.show_baseline_collision)
        self.baseline_time_button.clicked.connect(self.show_baseline_time)
        self.baseline_success_button.clicked.connect(self.show_baseline_success)
        self.ablation_collision_button.clicked.connect(self.show_ablation_collision)
        self.ablation_time_button.clicked.connect(self.show_ablation_time)
        self.ablation_success_button.clicked.connect(self.show_ablation_success)

        self.Stop_Button.clicked.connect(self.media_player.pause)
        self.Continue_Button.clicked.connect(self.media_player.play)
        # ========== 滚动区域：装所有饼图 ==========
        self.scroll_container = QWidget()
        self.pie_layout = QGridLayout(self.scroll_container)
        self.scrollarea_pies.setWidget(self.scroll_container)
        self.scrollarea_pies.setWidgetResizable(True)
        self.all_pie_items = []

        # 如果提供了命令行参数，自动填充并运行
        if all([pursuer_num, obstacle_num, env_length, env_width]):
            self.Pursuer_Num_LineEdit.setText(str(pursuer_num))
            self.Obs_Num_LineEdit.setText(str(obstacle_num))
            self.env_length_LineEdit.setText(str(env_length))
            self.env_width_LineEdit.setText(str(env_width))
            # 使用定时器延迟调用，确保UI完全初始化
            QTimer.singleShot(100, self.on_apply_clicked)


    def on_apply_baseline(self):
        #1.读取输入框的内容
        experiment_num = self.exp_num_baseline_lineedit.text()
        pursuer_num = self.pursuer_num_lineedit.text()
        min_dist = self.init_min_dist_baseline_lineedit.text()
        obstacle_num = self.obs_num_baseline_lineedit.text()
        env_length = self.env_length_baseline_lineedit.text()
        env_width = self.env_width_baseline_lineedit.text()


    def w_str_to_num(self, w_str):
        import math
        pi = math.pi
        # 匹配你的角速度字符串（π/6、π/3等）
        convert_dict = {
            "π/6": pi/6,
            "π/3": pi/3,
            "π/2": pi/2,
            "π/4": pi/4,
            "0": 0
        }
        return convert_dict.get(w_str, 0)
    
    # 2. 创建1张饼图（返回做好的饼图控件）
    def create_pie_widget(self, data, title):
        # 创建画布
        plot = pg.PlotWidget()
        plot.setFixedSize(260, 240)
        plot.setAspectLocked()
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        # 白色背景核心
        plot.setBackground('white')
        # 黑色标题（适配白底）
        plot.setTitle(title, color='black', size='11pt')

        # 统计数据
        unique_vals, counts = np.unique(data, return_counts=True)
        total = sum(counts)
        if total == 0:
            return plot

        # 配色（鲜艳适配白底）
        colors = [(255,69,0), (30,144,255), (255,215,0), (0,255,127), (255,105,180)]
        start_angle = 0

        # 绘制饼图扇形
        for i, cnt in enumerate(counts):
            span_angle = 360 * cnt / total
            path = pg.QtGui.QPainterPath()
            path.moveTo(0, 0)
            path.arcTo(-1, -1, 2, 2, start_angle, span_angle)
            path.closeSubpath()

            sector = pg.QtWidgets.QGraphicsPathItem(path)
            sector.setBrush(pg.mkBrush(colors[i % len(colors)]))
            sector.setPen(pg.mkPen('white', width=1.5))
            plot.addItem(sector)
            start_angle += span_angle

        # ===================== 添加图例（数值+占比） =====================
        legend = pg.LegendItem(offset=(70, 10))
        legend.setParentItem(plot.graphicsItem())
        for i, (val, cnt) in enumerate(zip(unique_vals, counts)):
            percent = f"{cnt/total*100:.1f}%"
            # 图例显示：数值 | 百分比
            legend.addItem(pg.PlotDataItem(pen=colors[i % len(colors)]), f"{val} | {percent}")

        return plot
    

    def on_apply_clicked(self):
        # 1. 读取输入框内容
        pursuer_num = self.Pursuer_Num_LineEdit.text()
        obstacle_num = self.Obs_Num_LineEdit.text()
        gv = generate_video(pursuer_num, obstacle_num)
        visual = Visualizer()
        video_path,energies,times,execution_time, pursuer_captured_Id,pursuer_a_list, w_list = gv.run_experiment()
        for i in pursuer_a_list:
            if i==[]:
                continue
            for j in range(len(i)):  
                i[j] = float(i[j]) 

        for j in w_list:
            if j==[]:
                continue
            for l in range(len(j)):
                j[l] = float(j[l])
                
        pi_over_6 = np.pi / 6
        # 创建替换后的新列表
        pursuer_w_list = []
        for i in w_list:
            append_list = []
            for num in i:
                if np.isclose(num, -pi_over_6, atol=1e-8):  # 检查是否接近 -π/6
                    append_list.append('-π/6')
                elif np.isclose(num, pi_over_6, atol=1e-8):  # 检查是否接近 π/6
                    append_list.append('π/6')
                else:
                    append_list.append('0')  # 保持其他数值不变
            pursuer_w_list.append(append_list)
        
        self.show_time_edit.setText(f"{execution_time:.2f}s")
        self.show_encircle_rate_edit.setText(f"100%")

        for i in reversed(range(self.pie_layout.count())):
            self.pie_layout.itemAt(i).widget().setParent(None)

        self.all_pie_items.clear()

        # 遍历所有个体 → 批量生成饼图（你的原有表格逻辑不变）
        for row_idx, row_data in enumerate(pursuer_captured_Id):

            # ===================== 批量生成：每个个体2张饼图 =====================
            # 1. 加速度饼图
            a_pie = self.create_pie_widget(
                data=pursuer_a_list[row_idx],
                title=f"追捕无人机{row_idx} | 加速度a"
            )
            # 2. 角速度饼图
            w_data = [self.w_str_to_num(s) for s in pursuer_w_list[row_idx]]
            w_pie = self.create_pie_widget(
                data=w_data,
                title=f"追捕无人机{row_idx} | 角速度w"
            )
            # 网格布局：每行放 【a饼图】【w饼图】
            self.pie_layout.addWidget(a_pie, row_idx, 0)
            self.pie_layout.addWidget(w_pie, row_idx, 1)

        if video_path:
            # 3. 使用媒体播放器播放视频
            #第一张
            self.energy_pursuer_graph_widget.clear()
            self.energy_pursuer_graph_widget.showAxis('right')
            left_axis = self.energy_pursuer_graph_widget.getAxis("left")
            right_axis = self.energy_pursuer_graph_widget.getAxis("right")
            left_axis.setLabel(text="Energy",color='red')
            right_axis.setLabel(text="Time",color='blue')
            x_axis = self.energy_pursuer_graph_widget.getAxis("bottom")
            x_axis.setLabel(text="UAV_id")
            x_axis.setTicks([[(i, str(int(i))) for i in np.arange(len(energies))]])
            # 强制只显示整数刻度，不自动生成小数
            x_axis.setTickSpacing(1, 1)  # 主刻度步长=1，只显示整数
            self.energy_pursuer_graph_widget.setAutoVisible(y=True)
            
            # 1. 绘制 能量 红色柱状图（左Y轴）
            x_energy = np.arange(len(energies))
            width = 0.2  # 柱子宽度
            bg_energy = pg.BarGraphItem(x=x_energy - width/2, height=energies, width=width, brush='r')
            self.energy_pursuer_graph_widget.addItem(bg_energy)

            # 2. 绘制 时间 蓝色柱状图（右Y轴）
            x_time = np.arange(len(times))
            bg_time = pg.BarGraphItem(x=x_time + width/2, height=times, width=width, brush='blue')
            self.energy_pursuer_graph_widget.addItem(bg_time)

            # self.energy_pursuer_graph_widget.plot(np.arange(len(energies)), energies, pen='r')
            # self.energy_pursuer_graph_widget.setLabels(left='energy', bottom='UAV_id')
            # self.energy_pursuer_graph_widget.plot(np.arange(len(times)),times,pen='blue')
            # self.energy_pursuer_graph_widget.setLabels(right='times', bottom='UAV_id')
            self.energy_pursuer_graph_widget.addLegend()

            media = QMediaContent(QUrl.fromLocalFile(video_path))
            self.media_player.setMedia(media)
            self.media_player.play()

            video_path = visual.show()
            self.play_video_in_qt(video_path)

    def play_video_in_qt(self, video_path):
        """将保存好的视频播放到 Qt 视频组件"""
        # 1. 把本地路径转成 Qt 能识别的 URL
        video_url = QUrl.fromLocalFile(video_path)

        # 2. 设置媒体内容
        self.media_player_traditional_control.setMedia(QMediaContent(video_url))

        # 3. 设置输出画面到你的 QVideoWidget
        self.media_player_traditional_control.setVideoOutput(self.video_widget_traditional_control)

        # 4. 开始播放
        self.media_player_traditional_control.play()

        print("▶️ 视频已在 Qt 界面播放")
    
    def show_page1(self):
        self.stackedWidget.setCurrentIndex(0)
    
    def show_page2(self):
        self.stackedWidget.setCurrentIndex(1)

    def show_page3(self):
        self.stackedWidget.setCurrentIndex(2)

    def show_page4(self):
        self.stackedWidget.setCurrentIndex(3)

    def show_baseline_collision(self):
        self.stackedWidget_2.setCurrentIndex(0)

    def show_baseline_time(self):
        self.stackedWidget_2.setCurrentIndex(1)

    def show_baseline_success(self):
        self.stackedWidget_2.setCurrentIndex(2)

    def show_ablation_collision(self):
        self.stackedWidget_3.setCurrentIndex(0)
    
    def show_ablation_time(self):
        self.stackedWidget_3.setCurrentIndex(1)

    def show_ablation_success(self):
        self.stackedWidget_3.setCurrentIndex(2)

def parse_args():
    parser = argparse.ArgumentParser(description='UAV Pursuit-Evasion Visualization Tool')
    parser.add_argument('--pursuer-num', type=int, help='Number of pursuers')
    parser.add_argument('--env_width', type=int, help='width of environment')
    parser.add_argument('--env_length', type=int, help='length of environment')
    parser.add_argument('--obstacle-num', type=int, help='Number of obstacles')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    window = Apply(
        pursuer_num=args.pursuer_num,
        env_width=args.env_width,
        env_length=args.env_length,
        obstacle_num=args.obstacle_num
    )
    window.show()
    sys.exit(app.exec_())