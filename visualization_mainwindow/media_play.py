from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
# 替换成你用 pyuic5 转换后的 ui 文件（如 mainwindow_ui.py）
from uav_mainwindow import Ui_MainWindow 

class MediaPlay(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 1. 初始化媒体播放器
        self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget(self)  # 视频显示控件
        
        self.imgDisplayLabel.setStyleSheet("background: white;")  # 背景设为黑色
        self.video_widget.setParent(self.imgDisplayLabel)  # 放到 QLabel 里

        self.video_widget.resize(self.imgDisplayLabel.size())  # 适配大小
        
        # 3. 关联播放器与显示控件
        self.media_player.setVideoOutput(self.video_widget)
        
        # 4. 绑定按钮事件
        self.Play_Button.clicked.connect(self.load_video)

        self.Stop_Button.clicked.connect(self.media_player.pause)
        self.Continue_Button.clicked.connect(self.media_player.play)

    def load_video(self):
        """选择视频文件并播放"""
        # 打开文件对话框选视频
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            # 设置媒体内容并播放
            media = QMediaContent(QUrl.fromLocalFile(file_path))
            self.media_player.setMedia(media)
            self.media_player.play()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())