from collections import OrderedDict
from datetime import datetime, timedelta
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QStyle, QWidget, QTableWidgetItem, QErrorMessage, QFileDialog

from .view import VideoAppViewer
from .view import VideoAppMain
from .text_recognition import recognizer

from pathlib import Path
import os.path
import csv


class MyMainApp(VideoAppMain):
    def __init__(self, **config):
        self.videoApp = VideoApp(**config)
        super().__init__(self.videoApp)

        self.select_path.triggered.connect(self.select_video_path)

        self.show()

    @pyqtSlot()
    def select_video_path(self):
        file_tuple = QFileDialog.getOpenFileName(self, "Open Video", "~", "Video Files (*.mp4);; Video Files ts (*.ts)")
        self.videoApp.videopath = str(file_tuple[0])
        self.centralWidget().setDisabled(False)


class VideoApp(VideoAppViewer):
    def __init__(self, **config):
        self.config = config
        self.title = self.config.get('title', 'PyQt5 video labeling viewer')
        super().__init__(title=self.title)

        # draw config
        if self.config.get('draw') and isinstance(self.config['draw'], dict):
            draw_config = self.config['draw']
            self.label_frame.draw_color = draw_config.get('color', QColor(0, 0, 0))
            self.label_frame.draw_thickness = draw_config.get('thickness', 2)
            self.label_frame.draw_style = draw_config.get('style', Qt.SolidLine)
        if self.config.get('select') and isinstance(self.config['select'], dict):
            select_config = self.config['select']
            self.label_frame.select_color = select_config.get('color', QColor(0, 0, 0))
            self.label_frame.select_thickness = select_config.get('thickness', 3)
            self.label_frame.select_style = select_config.get('style', Qt.SolidLine)

        # record config
        check_label = self.config.get('label')
        label_color = self.config['label'].get('color', (0, 0, 0)) if check_label else None
        label_thickness = self.config['label'].get('thickness', 2) if check_label else None
        self.label_color = label_color
        self.label_thickness = label_thickness
        self.limit_nlabel = self.config.get('limit_nlabel', None)
        self.records = []

        self.msg_with_guide = {'first': 'Draw a window around the time (the squarer, the better)',
                               'second': 'Indicate timestamps of the highlights',
                               'third': 'Add to the table all the highlights, then cut them!'}
        self.status = 'first'

        # self.show()

    @property
    def videopath(self):
        return self._videopath

    @videopath.setter
    def videopath(self, value):
        self._videopath = value
        self.read_video()
        self.bind_widgets()

    @property
    def frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else None

    @property
    def frame_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else None

    @property
    def frame_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else None

    @property
    def video_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else None

    def read_video(self):
        # read video
        self.cap = cv2.VideoCapture(self._videopath)
        self.target_frame_idx = 0  # ready to update
        self.render_frame_idx = None  # redneded
        self.scale_height = self.scale_width = None
        self.is_playing_video = False
        self.is_force_update = False
        self._update_video_info()
        self._update_frame()

    def bind_widgets(self):
        # widget binding
        # general
        self.slider_video.setRange(0, self.frame_count - 1)
        self.slider_video.repaint()
        self.slider_video.sliderMoved.connect(self.on_slider_moved)
        self.slider_video.sliderReleased.connect(self.on_slider_released)
        # self.slider_video.valueChanged.connect(self.on_slider_moved)
        self.btn_play_video.clicked.connect(self.on_play_video_clicked)
        self.label_frame.mousePressEvent = self.event_frame_mouse_press
        self.label_frame.mouseMoveEvent = self.event_frame_mouse_move
        self.label_frame.mouseReleaseEvent = self.event_frame_mouse_release
        self.label_frame.mouse = self.event_frame_mouse_double_click
        self.btn_previous_record.clicked.connect(self._goto_previous_record)
        self.btn_next_record.clicked.connect(self._goto_next_record)
        self.btn_export_records.clicked.connect(self.save_file)
        self.btn_previous_frame.clicked.connect(self.dec_frame)
        self.btn_next_frame.clicked.connect(self.inc_frame)
        self.btn_3_previous_sec.clicked.connect(self.dec_3_sec)
        self.btn_3_next_sec.clicked.connect(self.inc_3_sec)
        self.btn_10_previous_frames.clicked.connect(self.dec_10_frames)
        self.btn_10_next_frames.clicked.connect(self.inc_10_frames)

        # video trimming
        self.init_trim.clicked.connect(self.set_init_trim_value)
        self.select_event.clicked.connect(self.set_event_value)
        self.begin_celebration.clicked.connect(self.set_celebration_value)
        self.stop_trim.clicked.connect(self.set_stop_trim_value)
        self.table_trim.itemClicked.connect(self.select_trim_from_table)

        # table toolbar
        self.delete_record.triggered.connect(self.delete_record_from_table)
        self.add_record.triggered.connect(self.add_trim_to_table)
        self.cut_video.triggered.connect(self.cut_videos)

    def _ndarray_to_qimage(self, image: np.ndarray):
        """convert cv2 image to pyqt5 image
        Arguments:
            image {np.ndarray} -- original RGB image

        Returns:
            {QImage} -- pyqt5 image format
        """
        return QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)

    def _frame_idx_to_hmsf(self, frame_idx: int):
        """convert to hmsf timestamp by given frame idx and fps"""
        assert self.video_fps
        base = datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
        delta = timedelta(seconds=frame_idx / self.video_fps)
        return (base + delta).strftime('%H:%M:%S.%f')

    def _frame_idx_to_hms(self, frame_idx: int):
        """convert to hms timestamp by given frame idx and fps"""
        assert self.video_fps
        base = datetime.strptime('00:00:00', '%H:%M:%S')
        delta = timedelta(seconds=frame_idx // self.video_fps)
        return (base + delta).strftime('%H:%M:%S')

    def _read_frame(self, frame_idx: int):
        """check frame idx and read frame status than return frame
        Arguments:
            frame_idx {int} -- frame index

        Returns:
            {np.ndarray} -- RGB image in (h, w, c)
        """
        if frame_idx >= self.frame_count:
            self.logger.exception('frame index %d should be less than %d', frame_idx, self.frame_count)
        else:
            self.target_frame_idx = frame_idx
            self.cap.set(1, frame_idx)
            read_success, frame = self.cap.read()
            if read_success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            self.logger.exception('read #%d frame failed', frame_idx)

    def _play_video(self):
        """play video when button clicked"""
        if self.is_playing_video and self.video_fps:
            frame_idx = min(self.render_frame_idx + 1, self.frame_count)
            if frame_idx == self.frame_count:
                self.on_play_video_clicked()
            else:
                self.target_frame_idx = frame_idx
        QTimer.singleShot(1 / self.video_fps, self._play_video)

    def _check_coor_in_frame(self, coor_x: int, coor_y: int):
        """check the coordinate in mouse event"""
        return 0 < coor_x < self.scale_width and 0 < coor_y < self.scale_height

    def _update_video_info(self):
        shape = str((self.frame_width, self.frame_height))
        self.label_video_path.setText(self.videopath)
        self.label_video_shape.setText(shape)
        self.label_video_fps.setText(str(self.video_fps))

    def _update_frame(self):
        """read and update image to label"""

        # disable useless buttons
        self.check_available_buttons()

        scale_factor = 1.5

        if self.target_frame_idx != self.render_frame_idx or self.is_force_update:
            self.is_force_update = False
            frame = self._read_frame(self.target_frame_idx)
            if frame is not None:
                # draw, convert, resize pixmap
                frame = self.draw_rects(self.target_frame_idx, frame)
                pixmap = QPixmap(self._ndarray_to_qimage(frame))
                # self.scale_width = int(min(pixmap.width(), self.screen.width()))
                # self.scale_height = int(pixmap.height() * (self.scale_width / pixmap.width()))
                self.scale_width = int(min(pixmap.width(), self.screen.width()))
                self.scale_height = int(pixmap.height() * (self.scale_width / pixmap.width()))
                pixmap = pixmap.scaled(self.scale_width / 1.5, self.scale_height / 1.5, Qt.KeepAspectRatio)
                # pixmap = pixmap.scaled(self.scale_width / scale_factor, self.scale_height / scale_factor, Qt.KeepAspectRatio)

                self.label_frame.setPixmap(pixmap)
                # self.label_frame.resize(self.scale_width, self.scale_height)

                # sync, update related information
                self._update_frame_status(self.target_frame_idx)
                self.render_frame_idx = self.target_frame_idx
                self.slider_video.setValue(self.render_frame_idx)
                self.slider_video.repaint()

        QTimer.singleShot(1000 / self.video_fps, self._update_frame)

    def check_available_buttons(self):
        "disable unreacheable slider moving buttons"
        # 3 sec shift buttons
        if self.slider_video.value() + 75 >= self.frame_count - 1:
            self.btn_3_next_sec.setDisabled(True)
        else:
            self.btn_3_next_sec.setDisabled(False)

        if self.slider_video.value() - 75 < 0:
            self.btn_3_previous_sec.setDisabled(True)
        else:
            self.btn_3_previous_sec.setDisabled(False)

        # 10 frames shift buttons
        if self.slider_video.value() + 10 >= self.frame_count - 1:
            self.btn_10_next_frames.setDisabled(True)
        else:
            self.btn_10_next_frames.setDisabled(False)

        if self.slider_video.value() - 10 < 0:
            self.btn_10_previous_frames.setDisabled(True)
        else:
            self.btn_10_previous_frames.setDisabled(False)

        # 1 frame shift button
        if self.slider_video.value() + 1 >= self.frame_count - 1:
            self.btn_next_frame.setDisabled(True)
        else:
            self.btn_next_frame.setDisabled(False)

        if self.slider_video.value() - 1 < 0:
            self.btn_previous_frame.setDisabled(True)
        else:
            self.btn_previous_frame.setDisabled(False)

    def draw_rects(self, frame_idx: int, frame: np.ndarray):
        # rest_records = list(filter(lambda x: x['frame_idx'] == frame_idx, self.records))
        if len(self.records) == 0:
            # if not rest_records:
            return frame
        # for record in rest_records:
        for record in self.records:
            pt1, pt2 = (record['x1'], record['y1']), (record['x2'], record['y2'])
            print('[draw_rects] Coordinates: (x1, {}), (y1, {}), (x2, {}), (y2, {})'.format(pt1[0], pt1[1], pt2[0],
                                                                                            pt2[1]))

            rect = cv2.rectangle(frame, pt1, pt2, self.label_color, self.label_thickness)
        return rect

    def _update_frame_status(self, frame_idx: int, err: str = ''):
        """update frame status
        Arguments:
            frame_idx {int} -- frame index

        Keyword Arguments:
            err {str} -- show status when exception (default: '')
        """
        msg = '#frame ({}/{})'.format(frame_idx, self.frame_count - 1)
        msg = msg + "     -      " + self.msg_with_guide[self.status]
        if err:
            msg += '\n{}'.format(err)
        self.label_video_status.setText(msg)

    def _get_records_by_frame_idx(self, frame_idx=None):
        """return specfic records by frame index (default: current frame)"""
        frame_idx = frame_idx or self.render_frame_idx
        return list(filter(lambda x: x['frame_idx'] == frame_idx, self.records))

    def _get_nrecord_in_current_frame(self):
        """get the number of records in current frame"""
        current_records = self._get_records_by_frame_idx()
        return len(current_records) if current_records else None

    def get_events(self, events_file_path, start_time_min, start_time_sec, end_time_min, end_time_sec):
        df = pd.read_csv(events_file_path, sep=',')
        my_events = df.loc[
            (df['_min'] >= start_time_min) & (df['sec'] >= start_time_sec) & (df['_min'] <= end_time_min) & (
                    df['sec'] <= end_time_sec)]
        self.events_list = my_events
        return my_events

    def _remove_record(self, frame_idx: int, pt1: tuple, pt2: tuple):
        """remove record by given value
        """
        current_records = self._get_records_by_frame_idx(frame_idx)
        target_record = None
        for record in current_records:
            src_pt1, src_pt2 = (record['x1'], record['y1']), (record['x2'], record['y2'])
            if src_pt1 == pt1 and src_pt2 == pt2:
                target_record = record
        if target_record:
            target_row_idx = self.records.index(target_record)
            self.records.remove(target_record)
            self.remove_record_from_preview(target_row_idx)

    @pyqtSlot()
    def cut_videos(self):
        print("@pyqtSlot() cut_videos")

        frame_to_add = self.video_fps * 30

        csv_dir = os.path.dirname(self.videopath) + '/cuts/'
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        csv_path = csv_dir + 'labels_info.csv'
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as writeFile:
                writer = csv.writer(writeFile)
                label_info = [
                    ['video_name', '#_highlight', 'starting_frame', 'goal_frame', 'ending_frame', 'start_celebration',
                     'starting_time', 'ending_time', 'added_frames_bf', 'fps']]
                writer.writerows(label_info)

        label_info = []
        all_rows = self.table_trim.rowCount()
        i = 1
        for row in range(0, all_rows):
            i_trim_frame = self.table_trim.item(row, 0)
            s_trim_frame = self.table_trim.item(row, 1)
            event_frame = self.table_trim.item(row, 2)
            celebration = self.table_trim.item(row, 3).text()
            i_time = self.table_trim.item(row, 4).text()
            s_time = self.table_trim.item(row, 5).text()

            # decimal values, they take account of the single cutting frame
            i_trim_sec = (int(i_trim_frame.text()) - frame_to_add)/ self.video_fps
            s_trim_sec = (int(s_trim_frame.text()) + frame_to_add) / self.video_fps

            file_name_output = self.videopath.replace('merged__res1', '/cuts/label' + '_' + str(i))

            if not os.path.exists(file_name_output):
                ffmpeg_extract_subclip(self.videopath, i_trim_sec, s_trim_sec,
                                       targetname=file_name_output)

            label_info.append(
                [file_name_output, i, i_trim_frame.text(), event_frame.text(), s_trim_frame.text(), celebration,
                 i_time, s_time, frame_to_add, self.video_fps])

            i = i + 1

        with open(csv_path, 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(label_info)

    @pyqtSlot()
    def inc_frame(self):
        self.slider_video.setValue(self.slider_video.value() + 1)
        self.slider_video.repaint()
        self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def dec_frame(self):
        self.slider_video.setValue(self.slider_video.value() - 1)
        self.slider_video.repaint()
        self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def inc_10_frames(self):
        self.slider_video.setValue(self.slider_video.value() + 10)
        self.slider_video.repaint()
        self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def dec_10_frames(self):
        self.slider_video.setValue(self.slider_video.value() - 10)
        self.slider_video.repaint()
        self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def inc_3_sec(self):
        if (self.slider_video.value() + 75 <= self.frame_count - 1):
            self.slider_video.setValue(self.slider_video.value() + 75)
            self.slider_video.repaint()
            self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def dec_3_sec(self):
        if (self.slider_video.value() - 75 >= 0):
            self.slider_video.setValue(self.slider_video.value() - 75)
            self.slider_video.repaint()
            self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def delete_record_from_table(self):
        if self.item_selected is not None:
            self.table_trim.removeRow(self.item_selected)
        else:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('No item selected!')
            error_dialog.exec_()

    @pyqtSlot()
    def set_init_trim_value(self):
        frame_selected = self.slider_video.value()
        ret, frame = self.cap.read()
        output_path_part = Path('.')
        output_path = output_path_part / 'OCR_frames' / 'init_frame.jpg'
        output_path_str = str(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path_str, frame[self.y1: self.y2, self.x1: self.x2])
        if (self.x1 is not None):
            recognized_text = recognizer(output_path_str, padding=0.08, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
            self.init_timestamp.setText(recognized_text)
            self.init_timestamp.repaint()
            # else:
            # recognizer(output_path_str, padding=0.08)

        if len(self.stop_trim_value.text()) > 0 and len(self.select_event_value.text()) > 0:
            if int(self.stop_trim_value.text()) < int(frame_selected) and int(self.select_event_value.text()) > int(
                    frame_selected):
                self.init_trim_value.setText(str(frame_selected))
                self.init_trim_value.repaint()

            else:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(
                    'Error! the starting frame should be BEFORE the event frame and BEFORE the ending frame')
                error_dialog.exec_()
        else:
            if len(self.stop_trim_value.text()) > 0 and int(self.stop_trim_value.text()) < int(frame_selected):
                error_dialog = QErrorMessage()
                error_dialog.showMessage('Error! the starting frame should be BEFORE the ending frame')
                error_dialog.exec_()
            else:
                if len(self.select_event_value.text()) > 0 and int(self.select_event_value.text()) < int(
                        frame_selected):
                    error_dialog = QErrorMessage()
                    error_dialog.showMessage('Error! the starting frame should be BEFORE the event frame')
                    error_dialog.exec_()
                else:
                    self.init_trim_value.setText(str(frame_selected))
                    self.init_trim_value.repaint()

        self.update()

    @pyqtSlot()
    def set_stop_trim_value(self):
        frame_selected = self.slider_video.value()
        self.status = 'third'

        ret, frame = self.cap.read()

        output_path_part = Path('.')
        output_path = output_path_part / 'OCR_frames' / 'stop_frame.jpg'
        output_path_str = str(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path_str, frame[self.y1: self.y2, self.x1: self.x2])

        if (self.x1 is not None):
            recognized_text = recognizer(output_path_str, padding=0.08, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)
            self.stop_timestamp.setText(recognized_text)
            self.stop_timestamp.repaint()

        if len(self.init_trim_value.text()) > 0 and len(self.select_event_value.text()) > 0:
            if int(self.init_trim_value.text()) < int(frame_selected) and int(self.select_event_value.text()) < int(
                    frame_selected):
                self.stop_trim_value.setText(str(frame_selected))
                self.stop_trim_value.repaint()
            else:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(
                    'Error! the ending frame should be AFTER the event frame and AFTER the starting frame')
                error_dialog.exec_()
        else:
            if len(self.init_trim_value.text()) > 0 and int(self.init_trim_value.text()) > int(frame_selected):
                error_dialog = QErrorMessage()
                error_dialog.showMessage('Error! the ending frame should be AFTER the starting frame')
                error_dialog.exec_()
            else:
                if len(self.select_event_value.text()) > 0 and int(self.select_event_value.text()) > int(
                        frame_selected):
                    error_dialog = QErrorMessage()
                    error_dialog.showMessage('Error! the ending frame should be AFTER the event frame')
                    error_dialog.exec_()
                else:
                    self.stop_trim_value.setText(str(frame_selected))
                    self.stop_trim_value.repaint()

        self.update()

    @pyqtSlot()
    def set_event_value(self):
        """ event value is set before or after selecting the starting and ending frame.
            Condition of inclusion between these two values should be satisfied"""
        frame_selected = self.slider_video.value()
        error_dialog = QErrorMessage()
        if len(self.stop_trim_value.text()) > 0 and len(self.init_trim_value.text()) > 0:
            # both init and end frame has been set
            if (int(frame_selected) > int(self.stop_trim_value.text()) or int(frame_selected) < int(
                    self.init_trim_value.text())):
                error_dialog.showMessage('Error! the event frame should be between the starting and ending frame')
                error_dialog.exec_()
            else:
                self.select_event_value.setText(str(frame_selected))
                self.select_event_value.repaint()
        else:
            if len(self.stop_trim_value.text()) > 0:
                if int(frame_selected) > int(self.stop_trim_value.text()):
                    error_dialog.showMessage('Error! the event frame should be before the ending frame')
                    error_dialog.exec_()
                else:
                    self.select_event_value.setText(str(frame_selected))
                    self.select_event_value.repaint()
            else:
                if len(self.init_trim_value.text()) > 0:
                    if int(frame_selected) < int(self.init_trim_value.text()):
                        error_dialog.showMessage('Error! the event frame should be after the starting frame')
                        error_dialog.exec_()
                    else:
                        self.select_event_value.setText(str(frame_selected))
                        self.select_event_value.repaint()
                else:
                    self.select_event_value.setText(str(frame_selected))
                    self.select_event_value.repaint()

        self.update()

    @pyqtSlot()
    def set_celebration_value(self):
        """ event value is set before or after selecting the starting and ending frame.
            Condition of inclusion between these two values should be satisfied"""
        frame_selected = self.slider_video.value()
        error_dialog = QErrorMessage()
        if len(self.stop_trim_value.text()) > 0 and len(self.init_trim_value.text()) > 0:
            # both init and end frame has been set
            if (int(frame_selected) > int(self.stop_trim_value.text()) or int(frame_selected) < int(
                    self.init_trim_value.text())):
                error_dialog.showMessage('Error! the event frame should be between the starting and ending frame')
                error_dialog.exec_()
            else:
                self.begin_celebration_value.setText(str(frame_selected))
                self.begin_celebration_value.repaint()
        else:
            if len(self.stop_trim_value.text()) > 0:
                if int(frame_selected) > int(self.stop_trim_value.text()):
                    error_dialog.showMessage('Error! the event frame should be before the ending frame')
                    error_dialog.exec_()
                else:
                    self.begin_celebration_value.setText(str(frame_selected))
                    self.begin_celebration_value.repaint()
            else:
                if len(self.init_trim_value.text()) > 0:
                    if int(frame_selected) < int(self.init_trim_value.text()):
                        error_dialog.showMessage('Error! the event frame should be after the starting frame')
                        error_dialog.exec_()
                    else:
                        self.begin_celebration_value.setText(str(frame_selected))
                        self.begin_celebration_value.repaint()
                else:
                    self.begin_celebration_value.setText(str(frame_selected))
                    self.begin_celebration_value.repaint()

        self.update()

    @pyqtSlot()
    def select_trim_from_table(self):
        self.item_selected = self.table_trim.currentRow()
        print("self.item_selected", self.item_selected)

    @pyqtSlot()
    def _goto_previous_record(self):
        rest_records = list(filter(lambda x: x['frame_idx'] < self.render_frame_idx, self.records))
        if not rest_records:
            QMessageBox.information(self, 'Info', 'no previous record', QMessageBox.Ok)
        else:
            self.target_frame_idx = rest_records[-1]['frame_idx']

    @pyqtSlot()
    def _goto_next_record(self):
        rest_records = list(filter(lambda x: x['frame_idx'] > self.render_frame_idx, self.records))
        if not rest_records:
            QMessageBox.information(self, 'Info', 'no next record', QMessageBox.Ok)
        else:
            self.target_frame_idx = rest_records[0]['frame_idx']

    @pyqtSlot()
    def on_slider_released(self):
        """update frame and frame status when the slider released"""
        self.target_frame_idx = self.slider_video.value()

    @pyqtSlot()
    def on_slider_moved(self):
        """update frame status only when the slider moved"""
        self._update_frame_status(frame_idx=self.slider_video.value())

    @pyqtSlot()
    def on_play_video_clicked(self):
        """control to play or pause the video"""
        self.is_playing_video = not self.is_playing_video
        if self.is_playing_video:
            self.btn_play_video.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.btn_play_video.repaint()
            self._play_video()
        else:
            self.btn_play_video.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.btn_play_video.repaint()

    @pyqtSlot()
    def event_frame_mouse_press(self, event):
        """start drawing rectangle"""
        if self._check_coor_in_frame(event.x(), event.y()) and not self.is_playing_video:
            if event.button() == Qt.LeftButton:
                self.label_frame.is_drawing = True
                self.label_frame.is_selecting = False
                self.logger.debug('press mouse at (%d, %d)', event.x(), event.y())
                self.label_frame.pt1 = (event.x(), event.y())
                print('event x', event.x())
                print('event y', event.y())
            elif event.button() == Qt.RightButton:
                print('right button clicked')

    @pyqtSlot()
    def event_frame_mouse_move(self, event):
        """drawing rectangle"""
        if self.label_frame.is_drawing:
            self.logger.debug('move mouse at (%d, %d)', event.x(), event.y())
            self.label_frame.pt2 = (event.x(), event.y())
            self.update()

    @pyqtSlot()
    def event_frame_mouse_release(self, event):
        """conclude drawing rectangle"""
        scale_factor = 1.5
        if self.label_frame.is_drawing:
            self.label_frame.is_drawing = False
            self.logger.debug('release mouse at (%d, %d)', event.x(), event.y())
            if self._check_coor_in_frame(event.x(), event.y()):
                self.label_frame.pt2 = (event.x(), event.y())

            pt1, pt2 = self.label_frame.revise_coor(self.label_frame.pt1, self.label_frame.pt2)
            record = OrderedDict([
                ('frame_idx', self.render_frame_idx), ('fps', self.video_fps),
                ('x1', int(pt1[0] * scale_factor)),
                ('y1', int(pt1[1] * scale_factor)),
                ('x2', int(pt2[0] * scale_factor)),
                ('y2', int(pt2[1] * scale_factor))
            ])
            self.records.clear()
            self.records.append(record)
            self.enable_buttons()
            self.status = 'second'

            print('Coordinates: (x1, {}), (y1, {}), (x2, {}), (y2, {})'.format(pt1[0], pt1[1], pt2[0], pt2[1]))
            self.x1 = int(pt1[0] * scale_factor)
            self.y1 = int(pt1[1] * scale_factor)
            self.x2 = int(pt2[0] * scale_factor)
            self.y2 = int(pt2[1] * scale_factor)

            self.is_force_update = True

            self.update()

    @pyqtSlot()
    def enable_buttons(self):
        self.init_trim.setDisabled(False)
        self.select_event.setDisabled(False)
        self.begin_celebration.setDisabled(False)
        self.stop_trim.setDisabled(False)
        self.add_record.setDisabled(False)
        self.delete_record.setDisabled(False)
        self.cut_video.setDisabled(False)

    @pyqtSlot()
    def event_frame_mouse_double_click(self, event):
        """allows to select instantly the value of the frame and to it into the 'select event' form value"""
        self.set_event_value()

    @pyqtSlot()
    def event_preview_clicked(self):
        self.event_selected = self.table_events_preview_records.currentRow()
        record = OrderedDict([
            ('type_id', int(self.table_events_preview_records.item(self.event_selected, 1).text())),
            ('left_frames', int(self.table_events_preview_records.item(self.event_selected, 0).text())),
            ('right_frames', self.frame_count - 1)
        ])
        self.records.append(record)
        self.records = sorted(self.records, key=lambda x: x['left_frames'])

        # frame_idx = int(self.table_events_preview_records.item(self.event_selected, 0).text())
        # self.target_frame_idx = frame_idx

    @pyqtSlot()
    def add_trim_to_table(self):
        self.table_trim.insertRow(0)
        self.table_trim.setItem(0, 0, QTableWidgetItem(str(self.init_trim_value.text())))
        self.table_trim.setItem(0, 1, QTableWidgetItem(str(self.stop_trim_value.text())))
        self.table_trim.setItem(0, 2, QTableWidgetItem(str(self.select_event_value.text())))
        self.table_trim.setItem(0, 3, QTableWidgetItem(str(self.begin_celebration_value.text())))
        self.table_trim.setItem(0, 4, QTableWidgetItem(str(self.init_timestamp.text())))
        self.table_trim.setItem(0, 5, QTableWidgetItem(str(self.stop_timestamp.text())))

    def save_file(self):
        """export records to default paths
        - click ok only close message box
        - click close to close PyQt program
        """
        exist_msg = 'File <b>{}</b> exist.<br/><br/>\
                         Do you want to replace?'.format(self.outpath)
        info_msg = 'Save at <b>{}</b><br/>\
                    total records: {}'.format(self.outpath, len(self.records))

        # check the file existense
        exist_reply = QMessageBox.No
        if Path(self.outpath).exists():
            exist_reply = QMessageBox.question(self, 'File Exist', exist_msg, \
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if not Path(self.outpath).exists() or exist_reply == QMessageBox.Yes:
            df_labels = pd.DataFrame().from_records(self.records)
            df_labels.to_csv(self.outpath, index=False)

        # check if the application is going to close
        reply = QMessageBox.about(self, 'Info', info_msg)
        self.close()

    def keyPressEvent(self, event):
        """global keyboard event"""
        if event.key() in [Qt.Key_Space, Qt.Key_P]:
            self.on_play_video_clicked()
        elif event.key() in [Qt.Key_Right, Qt.Key_D]:
            self.inc_frame()
            # self.target_frame_idx = min(self.target_frame_idx + self.video_fps, self.frame_count - 1)
        elif event.key() in [Qt.Key_Left, Qt.Key_A]:
            self.dec_frame()
            # self.target_frame_idx = max(0, self.target_frame_idx - self.video_fps)
        else:
            self.logger.debug('clicked %s but no related binding event', str(event.key()))
