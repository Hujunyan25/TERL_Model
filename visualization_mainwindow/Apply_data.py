import sys,os
# 获取 model_eval.py 所在目录的绝对路径
model_eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将该目录添加到 sys.path
sys.path.insert(0, model_eval_dir)
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
# 导入转换后的 ui 类
from uav_mainwindow import Ui_MainWindow 
from model_eval import generate_video
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl




class Apply(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 连接 Apply 按钮的点击信号到自定义槽函数
        self.ApplyButton.clicked.connect(self.on_apply_clicked)
        # 关联播放器与显示控件
        self.media_player.setVideoOutput(self.video_widget)
        # 4. 绑定按钮事件
        # self.Play_Button.clicked.connect(self.load_video)

        self.Stop_Button.clicked.connect(self.media_player.pause)
        self.Continue_Button.clicked.connect(self.media_player.play)
        
    def on_apply_clicked(self):
        # 1. 读取输入框内容
        pursuer_num = self.Pursuer_Num_LineEdit.text()
        evader_num = self.Evader_Num_LineEdit.text()
        pursuer_perception = self.R_Perception_lineEdit.text()
        obstacle_num = self.Obstacle_Num_LineEdit.text()
        gv = generate_video(pursuer_num, evader_num, pursuer_perception, obstacle_num)
        video_path = gv.run_experiment()
        if video_path:
            # 3. 使用媒体播放器播放视频
            media = QMediaContent(QUrl.fromLocalFile(video_path))
            self.media_player.setMedia(media)
            self.media_player.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Apply()
    window.show()
    sys.exit(app.exec_())