import sys,os,argparse
# 获取 model_eval.py 所在目录的绝对路径
model_eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将该目录添加到 sys.path
sys.path.insert(0, model_eval_dir)
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
# 导入转换后的 ui 类
from uav_mainwindow import Ui_MainWindow 
from model_eval import generate_video
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np
from pyqtgraph import mkPen




class Apply(QMainWindow, Ui_MainWindow):
    def __init__(self, pursuer_num=None, evader_num=None, pursuer_perception=None, obstacle_num=None):
        super().__init__()
        self.setupUi(self)
        # 1. 初始化媒体播放器
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
        self.switch_button.clicked.connect(self.switch_page)

        self.Stop_Button.clicked.connect(self.media_player.pause)
        self.Continue_Button.clicked.connect(self.media_player.play)
        
        # 如果提供了命令行参数，自动填充并运行
        if all([pursuer_num, evader_num, pursuer_perception, obstacle_num]):
            self.Pursuer_Num_LineEdit.setText(str(pursuer_num))
            self.Evader_Num_LineEdit.setText(str(evader_num))
            self.R_Perception_lineEdit.setText(str(pursuer_perception))
            self.Obstacle_Num_LineEdit.setText(str(obstacle_num))
            # 使用定时器延迟调用，确保UI完全初始化
            QTimer.singleShot(100, self.on_apply_clicked)


        
    def on_apply_clicked(self):
        # 1. 读取输入框内容
        pursuer_num = self.Pursuer_Num_LineEdit.text()
        evader_num = self.Evader_Num_LineEdit.text()
        pursuer_perception = self.R_Perception_lineEdit.text()
        obstacle_num = self.Obstacle_Num_LineEdit.text()
        gv = generate_video(pursuer_num, evader_num, pursuer_perception, obstacle_num)
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
        self.tableWidget.setRowCount(0)  
        
        for row_idx, row_data in enumerate(pursuer_captured_Id):
            if row_data == []:
                continue  
            row_position = self.tableWidget.rowCount() 
            #插入新行
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(str(row_idx)))
            item_evader_set = QTableWidgetItem(str(row_data))
            self.tableWidget.setItem(row_position, 1, item_evader_set)
            item_a_set = QTableWidgetItem(str(pursuer_a_list[row_idx]))
            self.tableWidget.setItem(row_position, 2, QTableWidgetItem(item_a_set))
            item_w_set = QTableWidgetItem(str(pursuer_w_list[row_idx]))
            self.tableWidget.setItem(row_position, 3, QTableWidgetItem(item_w_set))

        if video_path:
            # 3. 使用媒体播放器播放视频
            #第一张
            self.energy_pursuer_graph_widget.clear()
            left_axis = self.energy_pursuer_graph_widget.getAxis("left")
            right_axis = self.energy_pursuer_graph_widget.getAxis("right")
            left_axis.setLabel(text="Energy",color='red')
            right_axis.setLabel(text="Time",color='blue')
            x_axis = self.energy_pursuer_graph_widget.getAxis("bottom")
            x_axis.setLabel(text="UAV_id")
            self.energy_pursuer_graph_widget.plot(np.arange(len(energies)), energies, pen='r')
            self.energy_pursuer_graph_widget.setLabels(left='energy', bottom='UAV_id')
            self.energy_pursuer_graph_widget.plot(np.arange(len(times)),times,pen='blue')
            self.energy_pursuer_graph_widget.setLabels(right='times', bottom='UAV_id')
            self.energy_pursuer_graph_widget.addLegend()

            media = QMediaContent(QUrl.fromLocalFile(video_path))
            self.media_player.setMedia(media)
            self.media_player.play()

    def switch_page(self):
        current_index = self.stackedWidget.currentIndex()
        if current_index == 0:
            self.stackedWidget.setCurrentIndex(1)
        else:
            self.stackedWidget.setCurrentIndex(0)
            self.media_player.pause()


def parse_args():
    parser = argparse.ArgumentParser(description='UAV Pursuit-Evasion Visualization Tool')
    parser.add_argument('--pursuer-num', type=int, help='Number of pursuers')
    parser.add_argument('--evader-num', type=int, help='Number of evaders')
    parser.add_argument('--perception', type=int, help='Pursuer perception range')
    parser.add_argument('--obstacle-num', type=int, help='Number of obstacles')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    window = Apply(
        pursuer_num=args.pursuer_num,
        evader_num=args.evader_num,
        pursuer_perception=args.perception,
        obstacle_num=args.obstacle_num
    )
    window.show()
    sys.exit(app.exec_())