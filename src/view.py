import logging

from PyQt5.QtCore import Qt, QFile, QSize
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QIcon, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QDesktopWidget, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSlider, QStyle, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget, QListWidget, QMenuBar, QFileDialog, QLineEdit,
                             QSpacerItem, QSizePolicy, QMainWindow, QAction, QToolBar, QMenu, QApplication)

import pandas as pd


class VideoFrameViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.logger = logging.getLogger(__name__)
        self.is_drawing = False
        self.is_selecting = False
        self.pt1 = self.pt2 = None
        self.select_pt1 = self.select_pt2 = None

        # case: draw config
        self.draw_color = QColor(0, 0, 0)
        self.draw_thickness = 1
        self.draw_style = Qt.SolidLine

        # case: select config
        self.select_color = QColor(0, 0, 0)
        self.select_thickness = 2
        self.select_style = Qt.SolidLine

        self.events_list = []

    def revise_coor(self, pt1: tuple, pt2: tuple):
        revise_pt1 = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
        revise_pt2 = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))
        return (revise_pt1, revise_pt2)

    def _draw_rect(self, pt1: tuple, pt2: tuple, pen: QPen):
        painter = QPainter()
        painter.begin(self)
        painter.setPen(pen)
        pt1_x, pt1_y, pt2_x, pt2_y = pt1[0], pt1[1], pt2[0], pt2[1]
        width, height = (pt2_x - pt1_x), (pt2_y - pt1_y)
        painter.drawRect(pt1_x, pt1_y, width, height)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.is_drawing and self.pt1 and self.pt2:
            pen = QPen(self.draw_color, self.draw_thickness, self.draw_style)
            pt1, pt2 = self.revise_coor(self.pt1, self.pt2)
            self._draw_rect(pt1, pt2, pen)

        elif not self.is_drawing and self.select_pt1 and self.select_pt2:
            pen = QPen(self.select_color, self.select_thickness, self.select_style)
            pt1, pt2 = self.revise_coor(self.select_pt1, self.select_pt2)
            self._draw_rect(pt1, pt2, pen)


class VideoAppViewer(QWidget):
    def __init__(self, title='PyQt5 video labeling viewer'):
        """init

        Arguments:
            QWidget {[type]} -- default qt widget

        Keyword Arguments:
            title {str} -- window title (default: {'PyQt5 video labeling viewer'})
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.title = title
        self.desktop = QDesktopWidget()
        self.font_header = QFont()
        self.font_header.setBold(True)
        self.item_selected = None
        # set geometry
        # self.top = 100
        # self.left = 100
        # self.width = 500
        # self.height = 300
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.screen = self.desktop.availableGeometry()

        # init window - init and set default config about window
        self.setWindowTitle(self.title)

        # grid: root layout
        self.grid_root = QGridLayout()
        self.setLayout(self.grid_root)
        vbox_trim = QVBoxLayout()
        vbox_panels = QVBoxLayout()
        vbox_option = QVBoxLayout()
        self.grid_root.addLayout(vbox_trim, 0, 0)
        self.grid_root.addLayout(vbox_panels, 0, 1)

        # info about video section to cut
        self.trim_video = QGroupBox('Video trim and save')
        sub_trim_grid = QGridLayout()
        self.trim_video.setLayout(sub_trim_grid)
        self.init_trim = QPushButton('init trim')
        self.init_trim.setDisabled(True)
        self.init_trim_value = QLineEdit(self)
        self.init_trim_value.setFixedWidth(100)

        self.select_event = QPushButton('select event')
        self.select_event.setDisabled(True)
        self.select_event_value = QLineEdit(self)

        self.begin_celebration = QPushButton('begin celebration')
        self.begin_celebration.setDisabled(True)
        self.begin_celebration_value = QLineEdit(self)

        self.stop_trim = QPushButton('stop trim')
        self.stop_trim.setDisabled(True)
        self.stop_trim_value = QLineEdit(self)

        sub_trim_grid.addWidget(self.init_trim, 0, 0)
        sub_trim_grid.addWidget(self.init_trim_value, 0, 1)
        sub_trim_grid.addWidget(self.select_event, 1, 0)
        sub_trim_grid.addWidget(self.select_event_value, 1, 1)
        sub_trim_grid.addWidget(self.begin_celebration, 2, 0)
        sub_trim_grid.addWidget(self.begin_celebration_value, 2, 1)
        sub_trim_grid.addWidget(self.stop_trim, 3, 0)
        sub_trim_grid.addWidget(self.stop_trim_value, 3, 1)

        self.trim_video.contentsMargins()
        self.trim_video.setAlignment(Qt.AlignTop)
        vbox_trim.addWidget(self.trim_video)

        self.timestamp_video = QGroupBox('Video timestamps')
        sub_timestamp_grid = QGridLayout()
        self.timestamp_video.setLayout(sub_timestamp_grid)
        self.init_timestamp = QLineEdit(self)
        self.stop_timestamp = QLineEdit(self)
        label_init_timestamp = self._get_header_label('Init time')
        label_stop_timestamp = self._get_header_label('Stop time')
        sub_timestamp_grid.addWidget(label_init_timestamp, 0, 0)
        sub_timestamp_grid.addWidget(self.init_timestamp, 0, 1)
        sub_timestamp_grid.addWidget(label_stop_timestamp, 1, 0)
        sub_timestamp_grid.addWidget(self.stop_timestamp, 1, 1)
        self.timestamp_video.contentsMargins()
        self.timestamp_video.setAlignment(Qt.AlignTop)
        vbox_trim.addWidget(self.timestamp_video)

        table_vbox = QVBoxLayout()
        table_toolbar = QToolBar('table_toolbar')
        table_toolbar.setIconSize(QSize(25, 25))

        self.add_record = QAction(QIcon('icons/table_row_add.png'), 'add item')
        self.add_record.setDisabled(True)
        table_toolbar.addAction(self.add_record)

        self.delete_record = QAction(QIcon('icons/table_row_delete.png'), 'delete item')
        self.delete_record.setDisabled(True)

        table_toolbar.addAction(self.delete_record)

        self.cut_video = QAction(QIcon('icons/video_cut.png'), 'cut video')
        self.cut_video.setDisabled(True)
        table_toolbar.addAction(self.cut_video)

        # table with cut info
        self.table_trim = self._get_trim_preview_table(self)
        table_vbox.addWidget(table_toolbar)
        table_vbox.addWidget(self.table_trim)
        vbox_trim.addLayout(table_vbox)
        table_vbox.insertStretch(-1, 1)

        # vbox_panel/label_video_status: show frame index or exception msg
        self.label_video_status = QLabel()
        self.label_video_status.setAlignment(Qt.AlignJustify)
        vbox_panels.addWidget(self.label_video_status)

        # vbox_panel/label_frame: show frame image
        self.label_frame = VideoFrameViewer(self)
        self.label_frame.setAlignment(Qt.AlignCenter)
        self.label_frame.setContentsMargins(0,0,0,0)
        self.label_frame.setMargin(0)
        self.label_frame.setMouseTracking(True)
        vbox_panels.addWidget(self.label_frame)

        # vbox_panel/hbox_video: show process about video

        hbox_video_slider_plot = QGridLayout()
        self.btn_play_video = QPushButton()
        self.btn_play_video.setEnabled(True)
        self.btn_play_video.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.setRange(0, 0)
        hbox_video_slider_plot.addWidget(self.btn_play_video, 0, 0)
        hbox_video_slider_plot.addWidget(self.slider_video, 0, 1)
        vbox_panels.addLayout(hbox_video_slider_plot)

        hbox_jump_frames = QHBoxLayout()
        hbox_jump_frames.alignment()
        hbox_jump_frames.addStretch()
        self.btn_3_previous_sec = QPushButton('<< 3 secs')
        self.btn_3_previous_sec.setFixedSize(120, 30)
        self.btn_10_previous_frames = QPushButton('<< 10 Frames')
        self.btn_10_previous_frames.setFixedSize(130, 30)
        self.btn_previous_frame = QPushButton('<< Previous Frame')
        self.btn_previous_frame.setFixedSize(150, 30)
        self.btn_next_frame = QPushButton('Next Frame >>')
        self.btn_next_frame.setFixedSize(150, 30)
        self.btn_10_next_frames = QPushButton('10 Frames >>')
        self.btn_10_next_frames.setFixedSize(130, 30)
        self.btn_3_next_sec = QPushButton('3 secs >>')
        self.btn_3_next_sec.setFixedSize(120, 30)

        hbox_jump_frames.addWidget(self.btn_3_previous_sec)
        hbox_jump_frames.addWidget(self.btn_10_previous_frames)
        hbox_jump_frames.addWidget(self.btn_previous_frame)

        hbox_jump_frames.addWidget(self.btn_next_frame)
        hbox_jump_frames.addWidget(self.btn_10_next_frames)
        hbox_jump_frames.addWidget(self.btn_3_next_sec)
        hbox_jump_frames.addStretch()
        vbox_panels.addLayout(hbox_jump_frames)

        # vbox_option/group_video_info: show video static info
        self.group_video_info = QGroupBox('Video Information')
        sub_grid = QGridLayout()
        label_path = self._get_header_label('Path')
        label_shape = self._get_header_label('Shape')
        label_fps = self._get_header_label('FPS')
        self.label_video_path = QLabel()
        self.label_video_path.setAlignment(Qt.AlignLeft)
        self.label_video_path.setWordWrap(True)
        self.label_video_shape = QLabel()
        self.label_video_shape.setAlignment(Qt.AlignLeft)
        self.label_video_fps = QLabel()
        self.label_video_fps.setAlignment(Qt.AlignLeft)
        sub_grid.addWidget(label_path, 0, 0)
        sub_grid.addWidget(self.label_video_path, 0, 1)
        sub_grid.addWidget(label_shape, 1, 0)
        sub_grid.addWidget(self.label_video_shape, 1, 1)
        sub_grid.addWidget(label_fps, 2, 0)
        sub_grid.addWidget(self.label_video_fps, 2, 1)
        self.group_video_info.setLayout(sub_grid)
        self.group_video_info.contentsMargins()
        self.group_video_info.setAlignment(Qt.AlignTop)
        vbox_option.addWidget(self.group_video_info)

        # Frame moving buttons
        # vbox_option/hbox_jump_records: jump to next or previous record
        hbox_jump_records = QHBoxLayout()
        self.btn_previous_record = QPushButton('<< Previous Record')
        self.btn_next_record = QPushButton('Next Record >>')
        hbox_jump_records.addWidget(self.btn_previous_record)
        hbox_jump_records.addWidget(self.btn_next_record)
        vbox_option.addLayout(hbox_jump_records)

        # vbox_option/btn_export: export records
        self.btn_export_records = QPushButton('Export')
        vbox_option.addWidget(self.btn_export_records)

        # events list
        self.events_list = []
        self.label_frame.events_list = self.events_list
        self.events_list_frames = []
        self.events_ids = []

    def open_file_name_dialog(self):
        """file opener window for playlist selection"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        return file_name

    def get_frame_from_event(self, event):
        """get frame from single event informations"""
        min = event['_min']
        print("min", min)
        sec = event['sec']
        print("sec", sec)
        seconds = min * 60 + sec
        print("seconds", seconds)
        initial = self.start_time_min * 60 + self.start_time_sec
        frame = (seconds - initial) * self.video_fps
        return frame

    def _get_header_label(self, text: str = ''):
        label = QLabel(text)
        label.setFont(self.font_header)
        label.setAlignment(Qt.AlignLeft)
        return label

    def get_event_name(self, event):
        df = pd.read_csv('appendix1.csv', sep=';', header=None)
        df.columns = ['event_id', 'name', 'description']
        result = df.loc[event['type_id'], "name"]
        return result

    def get_event_description(self, event):
        df = pd.read_csv('appendix1.csv', sep=';', header=None)
        df.columns = ['event_id', 'name', 'description']
        result = df.loc[event['type_id'], "description"]
        return result

    def _get_trim_preview_table(self, parent):
        table = QTableWidget(parent=parent)
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ['init frame', 'stop frame', 'event frame', 'begin celebration',  'match timestamp init', 'match timestamp stop'])
        table.setSortingEnabled(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        return table


class VideoAppMain(QMainWindow):
    def __init__(self, videoApp: VideoAppViewer):
        super().__init__()
        self.setCentralWidget(videoApp)
        self.centralWidget().setDisabled(True)
        self.select_path = QAction("&Select video...", self)
        self.select_path.setShortcut("Ctrl+O")
        self.statusBar()

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(self.select_path)
