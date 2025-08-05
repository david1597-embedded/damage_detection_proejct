import sys 
import csv
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt, QSize, QTimer
import tensorflow as tf
import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import re

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        #===============================================#
        #=================레이아웃관련초기화==============#
        # 전체 메인 레이아웃 수직
        main_layout = QVBoxLayout()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 상단 버튼 레이아웃 (레이아웃 전환용)
        button_layout = QHBoxLayout()
        self.btn_manual = QPushButton("수동 처리")
        self.btn_automatic = QPushButton("자동 처리")
        button_layout.addWidget(self.btn_manual)
        button_layout.addWidget(self.btn_automatic)
        button_layout.addStretch(1)
        button_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.addLayout(button_layout)

        #중첩 레이아웃 설정
        self.stacked_layout = QStackedLayout()
        main_layout.addLayout(self.stacked_layout)

        # 레이아웃 1 위젯 생성
        layout_manual_widget = self.create_layout_manual()
        self.stacked_layout.addWidget(layout_manual_widget)

        # 레이아웃 2 위젯 생성
        layout_automatic_widget = self.create_layout_automatic()
        self.stacked_layout.addWidget(layout_automatic_widget)

        # 버튼 클릭 시 레이아웃 전환
        self.btn_manual.clicked.connect(lambda: self.switch_to_manual())
        self.btn_automatic.clicked.connect(lambda: self.switch_to_automatic())

        #===============================================#
        #============딥러닝관련 변수들 초기화=============#
        self.valid_extensions = {".jpg", ".jpeg", ".png"}
        self.process_folder = None
        self.images_dir = None  # 작업을 진행할 폴더
        self.jpg_files = []  # 불러온 경로의 images 디렉토리의 이미지 파일들
        self.current_img_path = None
        self.current_index = -1
        self.ssd_model = None
        self.det_model = None
        self.ssd_model_path = "./models/ssdmobilenet/saved_model"
        self.det_model_path = "./models/efficientdet/saved_model"
        self.ssd_detect_fn = None  # 실제 추론 함수
        self.det_detect_fn = None
        self.label_map = {1: 'scratch', 2: 'damage'}
        self.is_work_folder_set = False # 작업 폴더 설정 여부 판단 변수
        self.min_index = None # 설정한 작업폴더의 이미지들 중 가장 작은 인덱스
        self.max_index = None # 설정한 작업폴더의 이미지들 중 가장 큰 인덱스
        self.score_threshold = 0.4
        self.is_working_auto = False
        self._start_time = None
        self._start_time_str = None
        self.effi_csv_rows = []
        self.ssd_csv_rows = []
        self.process_count = 0
        self.ssd_defection_count = 0
        self.det_defection_count = 0
        # 추가할 코드
        self.det_defective_images = []  # EfficientDet에서 불량으로 판정된 이미지 경로들
        self.ssd_defective_images = []  # SSD에서 불량으로 판정된 이미지 경로들
        self.det_good_images = []       # EfficientDet에서 양품으로 판정된 이미지 경로들
        self.ssd_good_images = []       # SSD에서 양품으로 판정된 이미지 경로들

        # 현재 활성 모드 추적
        self.current_mode = 'manual'  # 'manual' 또는 'automatic'

        # 모델 로드는 파일이 존재할 때만 실행
        if os.path.exists(self.ssd_model_path) and os.path.exists(self.det_model_path):
            self.load_model()
        else:
            print("모델 파일이 존재하지 않습니다. 모델 로딩을 건너뜁니다.")

    def switch_to_manual(self):
        if self.is_working_auto == False :
            self.current_mode = 'manual'
            self.stacked_layout.setCurrentIndex(0)
            self.update_current_layout()
            self.reset_to_default()
            self.manual_log_plaintext_left.clear()
        

    def switch_to_automatic(self):
        self.current_mode = 'automatic'
        self.stacked_layout.setCurrentIndex(1)
        self.update_current_layout()
        self.reset_to_default()
        self.manual_log_plaintext_right.clear()

    def update_current_layout(self):
        """현재 활성화된 레이아웃의 이미지와 텍스트를 업데이트"""
        if self.is_work_folder_set and self.current_index >= 0:
            # 현재 이미지 표시
            self.display_current_image()
            # 경로 텍스트 업데이트
            if hasattr(self, 'process_folder') and self.process_folder:
                if self.current_mode == 'manual':
                    self.manual_path_text.setText(self.process_folder)
                else:
                    self.auto_path_text.setText(self.process_folder)

    def display_current_image(self):
        """현재 이미지를 활성화된 레이아웃에 표시"""
        if not self.is_work_folder_set or self.current_index < 0:
            return
            
        self.current_img_path = str(self.jpg_files[self.current_index])
        pixmap = QPixmap(self.current_img_path)
        
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if self.current_mode == 'manual':
                self.manual_left_image.setPixmap(scaled_pixmap)
            else:
                self.auto_left_image.setPixmap(scaled_pixmap)
        else:
            print("이미지 로드 실패")

    def load_model(self):
        try:
            self.ssd_model = tf.saved_model.load(self.ssd_model_path)
            self.det_model = tf.saved_model.load(self.det_model_path) 

            # 추론 함수 설정 : Efficient Det D0
            if hasattr(self.det_model, 'signatures'):
                # 서명이 있는 경우
                if 'serving_default' in self.det_model.signatures:
                    self.det_detect_fn = self.det_model.signatures['serving_default']
                else:
                    # 첫 번째 서명 사용
                    det_signature_keys = list(self.det_model.signatures.keys())
                    if det_signature_keys:
                        self.det_detect_fn = self.det_model.signatures[det_signature_keys[0]]
            
            # 서명이 없는 경우 직접 호출 시도
            if self.det_detect_fn is None:
                if hasattr(self.det_model, '__call__'):
                    self.det_detect_fn = self.det_model
                elif hasattr(self.det_model, 'call'):
                    self.det_detect_fn = self.det_model.call
            
            if self.det_detect_fn is not None:
                print("EfficientDet 모델이 성공적으로 로드되었습니다.")
            else: 
                print("EfficientDet 모델이 로드되지 못했습니다.")
            
            # 추론 함수 설정 : SSD mobilenet V2
            if hasattr(self.ssd_model, 'signatures'):
                # 서명이 있는 경우
                if 'serving_default' in self.ssd_model.signatures:
                    self.ssd_detect_fn = self.ssd_model.signatures['serving_default']
                else:
                    # 첫 번째 서명 사용
                    ssd_signature_keys = list(self.ssd_model.signatures.keys())
                    if ssd_signature_keys:
                        self.ssd_detect_fn = self.ssd_model.signatures[ssd_signature_keys[0]]
            
            # 서명이 없는 경우 직접 호출 시도
            if self.ssd_detect_fn is None:
                if hasattr(self.ssd_model, '__call__'):
                    self.ssd_detect_fn = self.ssd_model
                elif hasattr(self.ssd_model, 'call'):
                    self.ssd_detect_fn = self.ssd_model.call
            
            if self.ssd_detect_fn is not None:
                print("SSD MobileNet 모델이 성공적으로 로드되었습니다.")
            else:  
                print("SSD MobileNet 모델이 로드되지 못했습니다.")
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
                
    def create_layout_manual(self):  
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 상단 수평 레이아웃
        top_layout = QHBoxLayout()
        top_origin_lbl = QLabel("검사 이미지")
        top_origin_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        top_det_lbl = QLabel("예측 : EfficientDet D0")
        top_det_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        
        top_ssd_lbl = QLabel("예측 : SSD Mobilenet V2")
        top_ssd_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        top_layout.addWidget(top_origin_lbl)
        top_layout.addWidget(top_det_lbl)
        top_layout.addWidget(top_ssd_lbl)
        top_layout.setSpacing(10)
        top_layout.setContentsMargins(20, 20, 20, 0)
        
        # 중간 수평 레이아웃
        middle_layout = QGridLayout()
        
        # 수동 모드 전용 위젯들
        self.manual_prev_button = QPushButton()
        self.manual_prev_button.setIcon(QIcon('./widgetimages/prev_arrow.png'))
        self.manual_prev_button.setIconSize(QSize(40, 40))
        self.manual_prev_button.setFixedSize(40, 40)
        self.manual_prev_button.clicked.connect(self.load_prev_image)

        self.manual_next_button = QPushButton()
        self.manual_next_button.setIcon(QIcon('./widgetimages/next_arrow.png'))
        self.manual_next_button.setIconSize(QSize(40, 40))
        self.manual_next_button.setFixedSize(40, 40)
        self.manual_next_button.clicked.connect(self.load_next_image)

        self.manual_left_image = QLabel()
        self.manual_left_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manual_left_image.setFixedSize(420, 420)
        self.set_default_image(self.manual_left_image)
        
        self.manual_middle_image = QLabel()
        self.manual_middle_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manual_middle_image.setFixedSize(420, 420)
        self.set_default_image(self.manual_middle_image)
        
        self.manual_right_image = QLabel()
        self.manual_right_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manual_right_image.setFixedSize(420, 420)
        self.set_default_image(self.manual_right_image)

        self.manual_path_text = QLabel("현재 이미지 경로 표시")
        self.manual_path_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 이미지 이동 버튼 추가
        middle_layout.addWidget(self.manual_prev_button, 0, 0)
        middle_layout.addWidget(self.manual_left_image, 0, 1)
        middle_layout.addWidget(self.manual_middle_image, 0, 2)
        middle_layout.addWidget(self.manual_right_image, 0, 3)
        middle_layout.addWidget(self.manual_next_button, 0, 4)
        middle_layout.addWidget(self.manual_path_text, 1, 0, 1, 2)
        
        middle_layout.setSpacing(10)
        middle_layout.setContentsMargins(10, 0, 10, 0)

        # 하단 수평 레이아웃
        bottom_layout = QHBoxLayout()
        
        # EfficientDet Log(왼쪽)
        effi_log_layout = QVBoxLayout()
        label_effi_log = QLabel("EfficientDet Log")
        label_effi_log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.manual_log_plaintext_left = QPlainTextEdit()
        self.manual_log_plaintext_left.setReadOnly(True)
        effi_log_layout.addWidget(label_effi_log)
        effi_log_layout.addWidget(self.manual_log_plaintext_left)

        # SSD Log(오른쪽)
        ssd_log_layout = QVBoxLayout()
        label_ssd_log = QLabel("SSD Log")
        label_ssd_log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.manual_log_plaintext_right = QPlainTextEdit()
        self.manual_log_plaintext_right.setReadOnly(True)
        ssd_log_layout.addWidget(label_ssd_log)
        ssd_log_layout.addWidget(self.manual_log_plaintext_right)

        # 두 로그 레이아웃을 수평으로 묶기
        logs_layout = QHBoxLayout()
        logs_layout.addLayout(effi_log_layout, 1)
        logs_layout.addLayout(ssd_log_layout, 1)

        middle_bottom_layout = QVBoxLayout()
        checkbox_texts = ['GrayScale Conversion', 'Gaussian blur', 'Canny edge', 'Specular reflection', 'Background Removal']
        self.checkboxes = [QCheckBox(text) for text in checkbox_texts]
        for checkbox in self.checkboxes:
            middle_bottom_layout.addWidget(checkbox)
        middle_bottom_layout.setSpacing(10)
        middle_bottom_layout.setContentsMargins(5, 5, 5, 5)
        middle_bottom_layout.addStretch(1)

        right_bottom_layout = QVBoxLayout()
        process_buttons_texts = ['작업 폴더 선택', '예측 시작', '선택된 영상 처리']
        self.manual_process_buttons = [QPushButton(text) for text in process_buttons_texts]
        for process_button in self.manual_process_buttons:
            right_bottom_layout.addWidget(process_button)
        self.manual_process_buttons[0].clicked.connect(self.select_folder)
        self.manual_process_buttons[1].clicked.connect(self.predict_and_compare)
        self.manual_process_buttons[2].clicked.connect(self.cv_process)
        right_bottom_layout.setSpacing(10)
        right_bottom_layout.setContentsMargins(5, 5, 5, 5)
        right_bottom_layout.addStretch(1)

        # 레이아웃 활성화
        bottom_layout.addLayout(logs_layout, 4)
        bottom_layout.addLayout(middle_bottom_layout, 1)
        bottom_layout.addLayout(right_bottom_layout, 1)
        
        layout.addLayout(top_layout, 0)
        layout.addLayout(middle_layout, 0)
        layout.addLayout(bottom_layout, 0)
        
        return widget
        
    def create_layout_automatic(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 상단 수평 레이아웃
        top_layout = QHBoxLayout()
        top_origin_lbl = QLabel("검사 이미지")
        top_origin_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        top_det_lbl = QLabel("예측 : EfficientDet D0")
        top_det_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        
        top_ssd_lbl = QLabel("예측 : SSD Mobilenet V2")
        top_ssd_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        top_layout.addWidget(top_origin_lbl)
        top_layout.addWidget(top_det_lbl)
        top_layout.addWidget(top_ssd_lbl)
        top_layout.setSpacing(10)
        top_layout.setContentsMargins(20, 20, 20, 0)
        
        # 중간 수평 레이아웃
        middle_layout = QGridLayout()
        
        # 자동 모드 전용 위젯들
        self.auto_prev_button = QPushButton()
        self.auto_prev_button.setIcon(QIcon('./widgetimages/prev_arrow.png'))
        self.auto_prev_button.setIconSize(QSize(40, 40))
        self.auto_prev_button.setFixedSize(40, 40)
        self.auto_prev_button.clicked.connect(self.load_prev_image)

        self.auto_next_button = QPushButton()
        self.auto_next_button.setIcon(QIcon('./widgetimages/next_arrow.png'))
        self.auto_next_button.setIconSize(QSize(40, 40))
        self.auto_next_button.setFixedSize(40, 40)
        self.auto_next_button.clicked.connect(self.load_next_image)

        self.auto_left_image = QLabel()
        self.auto_left_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.auto_left_image.setFixedSize(420, 420)
        self.set_default_image(self.auto_left_image)
        
        self.auto_middle_image = QLabel()
        self.auto_middle_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.auto_middle_image.setFixedSize(420, 420)
        self.set_default_image(self.auto_middle_image)
        
        self.auto_right_image = QLabel()
        self.auto_right_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.auto_right_image.setFixedSize(420, 420)
        self.set_default_image(self.auto_right_image)

        self.auto_path_text = QLabel("현재 이미지 경로 표시")
        self.auto_path_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 이미지 이동 버튼 추가
        middle_layout.addWidget(self.auto_prev_button, 0, 0)
        middle_layout.addWidget(self.auto_left_image, 0, 1)
        middle_layout.addWidget(self.auto_middle_image, 0, 2)
        middle_layout.addWidget(self.auto_right_image, 0, 3)
        middle_layout.addWidget(self.auto_next_button, 0, 4)
        middle_layout.addWidget(self.auto_path_text, 1, 0, 1, 2)
        
        middle_layout.setSpacing(10)
        middle_layout.setContentsMargins(10, 0, 10, 0)

        # 하단 수평 레이아웃
        bottom_layout = QHBoxLayout()
        
        # EfficientDet Log(왼쪽)
        effi_log_layout = QVBoxLayout()
        label_effi_log = QLabel("EfficientDet Log")
        label_effi_log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.auto_log_plaintext_left = QPlainTextEdit()
        self.auto_log_plaintext_left.setReadOnly(True)
        effi_log_layout.addWidget(label_effi_log)
        effi_log_layout.addWidget(self.auto_log_plaintext_left)

        # SSD Log(오른쪽)
        ssd_log_layout = QVBoxLayout()
        label_ssd_log = QLabel("SSD Log")
        label_ssd_log.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.auto_log_plaintext_right = QPlainTextEdit()
        self.auto_log_plaintext_right.setReadOnly(True)
        ssd_log_layout.addWidget(label_ssd_log)
        ssd_log_layout.addWidget(self.auto_log_plaintext_right)

        # 두 로그 레이아웃을 수평으로 묶기
        logs_layout = QHBoxLayout()
        logs_layout.addLayout(effi_log_layout, 1)
        logs_layout.addLayout(ssd_log_layout, 1)
    
        right_bottom_layout = QVBoxLayout()
        process_buttons_texts = ['작업 폴더 선택', '자동예측 시작', '작업 중지']
        self.auto_process_buttons = [QPushButton(text) for text in process_buttons_texts]
        for process_button in self.auto_process_buttons:
            right_bottom_layout.addWidget(process_button)
        self.auto_process_buttons[0].clicked.connect(self.select_folder)
        self.auto_process_buttons[1].clicked.connect(self.auto_predict_and_compare)
        self.auto_process_buttons[2].clicked.connect(self.abort_process)
        right_bottom_layout.setSpacing(10)
        right_bottom_layout.setContentsMargins(5, 5, 5, 5)
        right_bottom_layout.addStretch(1)

        # 레이아웃 활성화
        bottom_layout.addLayout(logs_layout, 6)
        bottom_layout.addLayout(right_bottom_layout, 1)
        
        layout.addLayout(top_layout, 0)
        layout.addLayout(middle_layout, 0)
        layout.addLayout(bottom_layout, 0)
        
        return widget

    def set_default_image(self, label):
        """기본 이미지 설정"""
        try:
            default_pixmap = QPixmap("./tempimages/temp.png")
            if default_pixmap.isNull():
                # temp.png가 없으면 빈 이미지 생성
                default_pixmap = QPixmap(420, 420)
                default_pixmap.fill(Qt.gray)
            scaled_pixmap = default_pixmap.scaled(420, 420, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"기본 이미지 설정 중 오류: {e}")
            # 오류 시 빈 이미지 생성
            default_pixmap = QPixmap(420, 420)
            default_pixmap.fill(Qt.gray)
            label.setPixmap(default_pixmap)

    # 이전 이미지 불러오는 함수
    def load_prev_image(self):
        print(f"현재 인덱스 : {self.current_index}")
        print(f"최소 인덱스 : {self.min_index}")
        print(f"최대 인덱스 : {self.max_index}")
        if self.is_work_folder_set == False:
            QMessageBox.warning(self, '폴더 선택 경고', '작업 폴더가 선택되지 않았습니다!')
            return
        if self.current_index == self.min_index:
            self.current_index = self.max_index
        else:
            self.current_index = self.current_index - 1
        
        self.display_current_image()
        
    # 다음 이미지 불러오는 함수
    def load_next_image(self):
        if self.is_working_auto and self.current_index > self.max_index:
            self.abort_process()
            return
        print(f"현재 인덱스 : {self.current_index}")
        print(f"최소 인덱스 : {self.min_index}")
        print(f"최대 인덱스 : {self.max_index}")
        if self.is_work_folder_set == False:
            QMessageBox.warning(self, '폴더 선택 경고', '작업 폴더가 선택되지 않았습니다!')
            return
        if self.current_index == self.max_index:
            self.current_index = self.min_index
            if self.current_mode == 'automatic':
                self.abort_process()
                self.reset_to_default()
                return
        if self.current_index < self.max_index:
            self.current_index = self.current_index + 1
        else:
            return False
        
        self.display_current_image()

        return True

    # 작업 폴더 버튼과 연결된 함수
    def select_folder(self):  
        folder_path = QFileDialog.getExistingDirectory(self, "작업 폴더 선택")
        if folder_path:
            self.process_folder = folder_path
            
            # 현재 활성화된 레이아웃의 텍스트 업데이트
            if self.current_mode == 'manual':
                self.manual_path_text.setText(folder_path)
            else:
                self.auto_path_text.setText(folder_path)
                
            self.images_dir = Path(folder_path) / "images"
            print(self.images_dir)
            
            if os.path.exists(self.images_dir):
                all_files = [f.name for f in self.images_dir.iterdir()]
                print(f"All files in {self.images_dir}: {all_files}")

                self.jpg_files = [f for f in self.images_dir.iterdir() if f.suffix.lower() in self.valid_extensions and "captured_image" in f.name and "204" in f.name]
                print(f"Found {len(self.jpg_files)} image files in {self.images_dir}")
                
                if self.jpg_files:
                    self.min_index = 0
                    self.max_index = len(self.jpg_files) - 1
                    self.is_work_folder_set = True
                    self.current_index = 0
                    print(f"최소 인덱스 : {self.min_index}")
                    print(f"최대 인덱스 : {self.max_index}")
                    
                    # 첫 번째 이미지 표시
                    self.display_current_image()
                else:
                    print("조건에 맞는 이미지 파일을 찾을 수 없습니다.")
                    self.reset_to_default()
            else:
                print("images 폴더가 존재하지 않습니다.")
                self.reset_to_default()
    #이미지 및 세팅을 기본 세팅으로 맞추는 함수
    def reset_to_default(self):
        """기본 상태로 리셋"""
        self.is_work_folder_set = False
        self.current_index = -1
        # 현재 활성화된 레이아웃의 이미지를 기본 이미지로 설정
        if self.current_mode == 'manual':
            self.set_default_image(self.manual_left_image)
            self.set_default_image(self.manual_middle_image)
            self.set_default_image(self.manual_right_image)
        else:
            self.set_default_image(self.auto_left_image)
            self.set_default_image(self.auto_middle_image)
            self.set_default_image(self.auto_right_image)

         # 추가할 코드
        self.det_defective_images = []
        self.ssd_defective_images = []
        self.det_good_images = []
        self.ssd_good_images = []
    #자동 분류 작업 중지 함수
    
    def load_ground_truth(self, processed_images=None):
        """
        실제 처리된 이미지들에 대해서만 ground truth 로드
        processed_images: 실제로 처리된 이미지 경로들의 리스트
        """
        self.ground_truth = {}
        
        # 처리된 이미지 목록이 없으면 전체 jpg_files 사용
        images_to_check = processed_images if processed_images else self.jpg_files
        
        if not images_to_check:
            print("체크할 이미지 목록이 비어 있습니다.")
            return

        for img_path in images_to_check:
            img_path_str = str(img_path)
            
            # 파일명에서 숫자 추출 (204_101_{number} 또는 204_102_{number} 형태)
            match = re.search(r'204_10[12]_(\d+)', img_path_str)
            
            if match:
                number = int(match.group(1))
                if number == 10:
                    quality = "양품"
                elif number == 20:
                    quality = "불량"
                else:
                    quality = "알 수 없음"
                    print(f"알 수 없는 레이블 숫자: {number} in {img_path_str}")
                    continue
                
                self.ground_truth[img_path_str] = quality
                print(f"파싱 성공: {img_path_str} -> {quality} (숫자: {number})")
            else:
                print(f"파일명에서 숫자를 추출하지 못했습니다: {img_path_str}")

        print(f"실제 처리된 이미지 중 라벨된 수: {len(self.ground_truth)}")
        
        # 양품/불량품 개수 확인
        good_count = sum(1 for label in self.ground_truth.values() if label == '양품')
        defective_count = sum(1 for label in self.ground_truth.values() if label == '불량')
        print(f"처리된 이미지 중 - 양품: {good_count}, 불량품: {defective_count}")

    def abort_process(self):
        if not self.is_working_auto:
            return

        processed_images = getattr(self, 'processed_images_list', self.jpg_files[:self.process_count])
        self.load_ground_truth(processed_images)

        end_time = datetime.now()
        if self._start_time:  # _start_time이 None이 아닌지 확인
            try:
                start_datetime = datetime.strptime(self._start_time, '%Y%m%d_%H%M%S')
                work_duration = (end_time - start_datetime).total_seconds()
            except (ValueError, TypeError) as e:
                print(f"시간 계산 오류: {e}")
                work_duration = 0
        else:
            print("시작 시간이 설정되지 않았습니다.")
            work_duration = 0   


        # det와 ssd 모델의 예측 정답률 계산
        det_correct = 0
        ssd_correct = 0
        total_labeled = len(self.ground_truth)
        
        # ground_truth에서 실제 불량품/양품 수 계산
        actual_defective = sum(1 for label in self.ground_truth.values() if label == '불량')
        actual_good = sum(1 for label in self.ground_truth.values() if label == '양품')
       
        # ground_truth와 모델 예측 비교
        for img_path, true_label in self.ground_truth.items():
            # det 모델 예측 (예: det_defection_count는 불량으로 예측한 수, 나머지는 양품으로 간주)
            det_predicted = '불량' if img_path in self.det_defective_images else '양품'  # det_defective_images는 불량으로 예측된 이미지 리스트라고 가정
            ssd_predicted = '불량' if img_path in self.ssd_defective_images else '양품' # ssd_defective_images는 불량으로 예측된 이미지 리스트라고 가정

            if det_predicted == true_label:
                det_correct += 1
            if ssd_predicted == true_label:
                ssd_correct += 1

        # 정답률 계산
        det_accuracy = det_correct / total_labeled if total_labeled > 0 else 0
        ssd_accuracy = ssd_correct / total_labeled if total_labeled > 0 else 0
       
        print(f"처리된 이미지 수: {self.process_count}")
        print(f"처리된 이미지 중 라벨 추출 가능한 수: {len(self.ground_truth)}")
        print(f"실제 불량품 수: {actual_defective}, 실제 양품 수: {actual_good}")

        # 모델 오차 계산 (처리된 이미지 수 기준)
        if self.process_count > 0:
            det_error = abs(self.det_defection_count - actual_defective) / self.process_count
            ssd_error = abs(self.ssd_defection_count - actual_defective) / self.process_count
        else:
            det_error = 0
            ssd_error = 0

        # 요약 CSV 파일 저장
        summary_log_dir = f'./results/model_result/summary_{self._start_time}.csv'
        os.makedirs(os.path.dirname(summary_log_dir), exist_ok=True)

        fieldnames = [
            '작업 폴더 경로', '작업 시간', '작업 이미지 수',
            'det 불량품 수', 'det 양품 수', 'ssd 불량품 수', 'ssd 양품 수',
            '실제 불량품 수', '실제 양품 수', 'efficient 모델오차', 'ssd 모델오차',
            'det 정답률', 'ssd 정답률'
        ]

        summary_row = {
            '작업 폴더 경로': self.process_folder if self.process_folder else 'N/A',
            '작업 시간': f"{work_duration / 60:.2f} 분",
            '작업 이미지 수': self.process_count,
            'det 불량품 수': self.det_defection_count,
            'det 양품 수': self.process_count - self.det_defection_count,
            'ssd 불량품 수': self.ssd_defection_count,
            'ssd 양품 수': self.process_count - self.ssd_defection_count,
            '실제 불량품 수': actual_defective,
            '실제 양품 수': actual_good,
            'efficient 모델오차': f"{det_error:.4f}",
            'ssd 모델오차': f"{ssd_error:.4f}",
            'det 정답률': f"{det_accuracy:.4f}",  # 정답률 추가
            'ssd 정답률': f"{ssd_accuracy:.4f}"   # 정답률 추가
        }

        try:
            with open(summary_log_dir, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(summary_row)
            print(f"요약 CSV 파일이 저장되었습니다: {summary_log_dir}")
            
            # CSV 내용 디버깅 출력
            with open(summary_log_dir, 'r', encoding='utf-8-sig') as f:
                print("CSV 내용:", f.read())
                
            QMessageBox.information(self, '작업 중단', f'자동 처리가 중단되고 요약 결과가 {summary_log_dir}에 저장되었습니다.')
        except Exception as e:
            print(f"요약 CSV 저장 중 오류 발생: {e}")
            QMessageBox.critical(self, '오류', f"CSV 저장 중 오류 발생: {e}")

        # 기본 설정으로 리셋
        self.reset_to_default()
        self.process_count = 0
        self.ssd_defection_count = 0
        self.det_defection_count = 0
        self.effi_csv_rows = []
        self.ssd_csv_rows = []
        self.det_csv_path = None
        self.ssd_csv_path = None
        self._start_time = None
        self._start_time_str = None
        self.ground_truth = {}

        # 로그 창 초기화
        if self.current_mode == 'manual':
            self.manual_log_plaintext_left.clear()
            self.manual_log_plaintext_right.clear()
        else:
            self.auto_log_plaintext_left.clear()
            self.auto_log_plaintext_right.clear() 
            self.is_working_auto = False

      
   
        #예측 및 분류 자동화 버튼 연계 함수
    def auto_predict_and_compare(self):
        print(self.is_work_folder_set)
    
        self._start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
        if self.is_work_folder_set == False:
            QMessageBox.warning(self, '폴더 선택 경고', '작업 폴더가 선택되지 않았습니다!')
            return
    
        self.is_working_auto = True
        self.current_index = self.min_index
        self.display_current_image()
    
        prefix = 'manual_' if self.current_mode == 'manual' else 'auto_'
    
        # 첫 번째 이미지 처리
        try:
            self.predict_and_compare()
        except Exception as e:
            left_log = getattr(self, f"{prefix}log_plaintext_left")
            right_log = getattr(self, f"{prefix}log_plaintext_right")
            left_log.appendPlainText(f"이미지 처리 에러 발생. {self.jpg_files[self.current_index]}: {str(e)}")
            right_log.appendPlainText(f"이미지 처리 에러 발생. {self.jpg_files[self.current_index]}: {str(e)}")
    
         # 다음 이미지들 처리 시작
        QTimer.singleShot(500, self.process_next_image_auto)
        
    #자동 예측및 분류 함수
    def process_next_image_auto(self):
        #"""자동 처리를 위한 별도 메서드"""
        if not self.is_working_auto:
            return  # 작업이 중단된 경우
        
        prefix = 'manual_' if self.current_mode == 'manual' else 'auto_'
        left_log = getattr(self, f"{prefix}log_plaintext_left")
        right_log = getattr(self, f"{prefix}log_plaintext_right")
        
        # 다음 이미지로 이동
        if not self.load_next_image():
            # 모든 이미지 처리 완료
            left_log.appendPlainText("전체 이미지 처리완료.")
            right_log.appendPlainText("전체 이미지 처리완료.")
            self.is_working_auto = False
            self.current_index = self.min_index  # 첫 번째 이미지로 리셋
            self.display_current_image()
            return
        
        # 현재 이미지 처리
        try:
            self.predict_and_compare()
            left_log.appendPlainText(f"처리 완료: {self.jpg_files[self.current_index]}")
            right_log.appendPlainText(f"처리 완료: {self.jpg_files[self.current_index]}")
        except Exception as e:
            left_log.appendPlainText(f"이미지 처리 에러 발생. {self.jpg_files[self.current_index]}: {str(e)}")
            right_log.appendPlainText(f"이미지 처리 에러 발생. {self.jpg_files[self.current_index]}: {str(e)}")
        
        # 다음 이미지 처리 예약 (재귀 호출 대신 타이머 사용)
        QTimer.singleShot(500, self.process_next_image_auto)
    #개별 예측 및 분류함수
    def predict_and_compare(self):
        
        if self.is_work_folder_set == False:
            return
        else:       
            # 추론 1단계 이미지 전처리 input_tensor 생성
            input_image = cv2.imread(self.current_img_path)
            if input_image is None:
                raise ValueError(f"이미지를 불러오지 못했습니다 : {self.current_img_path}")
            image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
            input_tensor = input_tensor[tf.newaxis, ...]
            
            # 추론 2단계 ssd_detections와 det_detections에 detections 반환
            ssd_detections = self.run_inference(input_tensor, self.ssd_detect_fn)
            det_detections = self.run_inference(input_tensor, self.det_detect_fn)
        if ssd_detections is None or det_detections is None:
            raise Exception("모델 추론결과가 없습니다")
        
        if self.current_mode == 'automatic':
            self.process_count = self.process_count + 1
                
        if isinstance(ssd_detections, dict) and isinstance(det_detections, dict):
            # 표준 Object Detection API 출력 형식
            if 'num_detections' in ssd_detections:
                ssd_num_detections = int(ssd_detections['num_detections'][0])
                ssd_detection_boxes = ssd_detections['detection_boxes'][0].numpy()
                ssd_detection_classes = ssd_detections['detection_classes'][0].numpy().astype(np.int32)
                ssd_detection_scores = ssd_detections['detection_scores'][0].numpy()
            else:
                # 다른 출력 형식일 경우
                print("감지된 출력 키:", list(ssd_detections.keys()))
                raise Exception("예상된 출력 키를 찾을 수 없습니다.")
            if 'num_detections' in det_detections:
                det_num_detections = int(det_detections['num_detections'][0])
                det_detection_boxes = det_detections['detection_boxes'][0].numpy()
                det_detection_classes = det_detections['detection_classes'][0].numpy().astype(np.int32)
                det_detection_scores = det_detections['detection_scores'][0].numpy()
            else:
                # 다른 출력 형식일 경우
                print("감지된 출력 키:", list(det_detections.keys()))
                raise Exception("예상된 출력 키를 찾을 수 없습니다.")
        else:
            # 텐서나 다른 형태의 출력
            print(f"예상과 다른 출력 형식: {type(detections)}")
            raise Exception("예상과 다른 모델 출력 형식입니다.")

        # 동적으로 prefix 설정
        prefix = 'manual_' if self.current_mode == 'manual' else 'auto_'

        #================EfficientDet D0 바운딩 박스 표시=========================#
        # 모델 별 detection을 활용해서 middle_image, right_image 업데이트하기.
        # 중앙에 이미지 표시
        self.current_img_path = str(self.jpg_files[self.current_index])
        print(self.current_img_path)
        pixmap = QPixmap(self.current_img_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            getattr(self, f"{prefix}middle_image").setPixmap(scaled_pixmap)
        print(f"\n=== 탐지 결과 디버깅 ({self.jpg_files[self.current_index]}) ===")
        print(f"이미지 크기: {image.shape}")
        print(f"num_detections: {det_num_detections}")
        print(f"상위 5개 탐지 결과:")

        #======Log 창 띄우기=======#
        # 신뢰도 0.4 이상인 모든 탐지 결과 선택
        det_valid_detections = []
        confidence_threshold = self.score_threshold  # 신뢰도 임계값을 0.4로 변경

        for i in range(min(5, det_num_detections)):
            det_score = det_detection_scores[i]
            if det_score > confidence_threshold:
                det_class_id = det_detection_classes[i]
                det_box = det_detection_boxes[i]
                det_valid_detections.append((i, det_score))
                print(f"  {i}: 클래스={det_class_id}, 신뢰도={det_score:.4f}, 박스={det_box}")

        det_valid_detections.sort(key=lambda x: x[1], reverse=True)

        getattr(self, f"{prefix}log_plaintext_left").appendPlainText(f"이미지 파일 : {self.jpg_files[self.current_index]} EfficientDet D1 예측 결과")
        getattr(self, f"{prefix}log_plaintext_left").appendPlainText("바운딩 박스 예측 결과")
 
        # EfficientDet CSV 데이터 준비
        det_csv_row = {
            '작업이미지파일 경로': self.current_img_path,
            '바운딩박스': '',
            '예측점수': 0.0,
            '불량여부판단': '',
            '불량개소': len(det_valid_detections)
        }

        for i, _ in det_valid_detections:
            log_det_score = det_detection_scores[i]
            log_det_class_id = det_detection_classes[i]
            log_det_box = det_detection_boxes[i]
            print(f"  {i}: 클래스={log_det_class_id}, 신뢰도={log_det_score:.4f}, 박스={log_det_box}")
            getattr(self, f"{prefix}log_plaintext_left").appendPlainText(f"  {i}: 클래스={log_det_class_id}, 신뢰도={log_det_score:.4f}, 박스={log_det_box}")
            if i == det_valid_detections[0][0]:  # 첫 번째 (최고 신뢰도)
                det_csv_row['바운딩박스'] = f"{log_det_box.tolist()}"
                det_csv_row['예측점수'] = f"{log_det_score:.4f}"
        
        
        
        if det_valid_detections:
            det_judge_log = "불량품"
            if self.current_mode == 'automatic':
                self.det_defection_count = self.det_defection_count + 1  # 추가할 코드
                self.det_defective_images.append(self.current_img_path)
        else:
            det_judge_log = "양품"
            self.det_good_images.append(self.current_img_path)
        det_csv_row['불량여부판단'] = det_judge_log
        getattr(self, f"{prefix}log_plaintext_left").appendPlainText(det_judge_log)
        # EfficientDet CSV 저장
        # SSD CSV 저장
        if self.current_mode == 'automatic':
            det_log_dir = f'./results/model_result/det_csv_{self._start_time}.csv'
            os.makedirs(os.path.dirname(det_log_dir), exist_ok=True)
            with open(det_log_dir, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['작업이미지파일 경로', '바운딩박스', '예측점수', '불량여부판단', '불량개소'])
                if f.tell() == 0:  # 파일이 비어 있으면 헤더 작성
                    writer.writeheader()
                writer.writerow(det_csv_row)


        # 바운딩 박스 그리기
        draw_image = input_image.copy()  # 원본 이미지 복사
        height, width = draw_image.shape[:2]
        for idx, _ in det_valid_detections:
            det_box = det_detection_boxes[idx]
            det_class_id = det_detection_classes[idx]
            det_score = det_detection_scores[idx]

            # 정규화된 박스 좌표를 이미지 크기에 맞게 변환
            ymin, xmin, ymax, xmax = det_box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # 박스 그리기 (녹색: damage, 파란색: scratch)
            color = (0, 255, 0) if det_class_id == 2 else (255, 0, 0)  # BGR 형식
            cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{self.label_map.get(det_class_id, 'unknown')} ({det_score:.2f})"
            cv2.putText(draw_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # OpenCV 이미지(BGR)를 RGB로 변환 후 QPixmap으로 변환
        draw_image_rgb = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
        height, width, channel = draw_image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(draw_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap_with_boxes = QPixmap.fromImage(q_image).scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        getattr(self, f"{prefix}middle_image").setPixmap(pixmap_with_boxes)

        #================SSD Mobilenet v2 바운딩 박스 표시=========================#
        # 오른쪽에 이미지 표시
        self.current_img_path = str(self.jpg_files[self.current_index])
        print(self.current_img_path)
        pixmap = QPixmap(self.current_img_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            getattr(self, f"{prefix}right_image").setPixmap(scaled_pixmap)

        print(f"\n=== 탐지 결과 디버깅 ({self.jpg_files[self.current_index]}) ===")
        print(f"이미지 크기: {image.shape}")
        print(f"num_detections: {ssd_num_detections}")
        print(f"상위 5개 탐지 결과:")

        ssd_valid_detections = []
        for i in range(min(5, ssd_num_detections)):
            ssd_score = ssd_detection_scores[i]
            if ssd_score > self.score_threshold:
                ssd_class_id = ssd_detection_classes[i]
                ssd_box = ssd_detection_boxes[i]
                ssd_valid_detections.append((i, ssd_score))
                print(f"  {i}: 클래스={ssd_class_id}, 신뢰도={ssd_score:.4f}, 박스={ssd_box}")

        ssd_valid_detections.sort(key=lambda x: x[1], reverse=True)

        getattr(self, f"{prefix}log_plaintext_right").appendPlainText(f"이미지 파일 : {self.jpg_files[self.current_index]} SSD mobilenet V2 예측 결과")
        getattr(self, f"{prefix}log_plaintext_right").appendPlainText("바운딩 박스 예측 결과")

        # SSD CSV 데이터 준비
        ssd_csv_row = {
            '작업이미지파일 경로': self.current_img_path,
            '바운딩박스': '',
            '예측점수': 0.0,
            '불량여부판단': '',
            '불량개소': len(ssd_valid_detections)
        }

        for i, _ in ssd_valid_detections:
            log_ssd_score = ssd_detection_scores[i]
            log_ssd_class_id = ssd_detection_classes[i]
            log_ssd_box = ssd_detection_boxes[i]
            print(f"  {i}: 클래스={log_ssd_class_id}, 신뢰도={log_ssd_score:.4f}, 박스={log_ssd_box}")
            getattr(self, f"{prefix}log_plaintext_right").appendPlainText(f"  {i}: 클래스={log_ssd_class_id}, 신뢰도={log_ssd_score:.4f}, 박스={log_ssd_box}")
            # 가장 높은 신뢰도의 탐지 결과 저장
            if i == ssd_valid_detections[0][0]:  # 첫 번째 (최고 신뢰도)
                ssd_csv_row['바운딩박스'] = f"{log_ssd_box.tolist()}"
                ssd_csv_row['예측점수'] = f"{log_ssd_score:.4f}"
        
        if ssd_valid_detections:
            ssd_judge_log = "불량품"
            if self.current_mode == 'automatic':
                self.ssd_defection_count = self.ssd_defection_count + 1
                 # 추가할 코드
                self.ssd_defective_images.append(self.current_img_path)
        else:
            ssd_judge_log = "양품"
            if self.current_mode == 'automatic':
                self.ssd_good_images.append(self.current_img_path)
        ssd_csv_row['불량여부판단'] = ssd_judge_log
        getattr(self, f"{prefix}log_plaintext_right").appendPlainText(ssd_judge_log)
        
       # SSD CSV 저장
        if self.current_mode == 'automatic':
            ssd_log_dir = f'./results/model_result/ssd_csv_{self._start_time}.csv'
            os.makedirs(os.path.dirname(ssd_log_dir), exist_ok=True)
            with open(ssd_log_dir, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['작업이미지파일 경로', '바운딩박스', '예측점수', '불량여부판단', '불량개소'])
                if f.tell() == 0:  # 파일이 비어 있으면 헤더 작성
                    writer.writeheader()
                writer.writerow(ssd_csv_row)


        # 바운딩 박스 그리기
        draw_image = input_image.copy()  # 원본 이미지 복사
        height, width = draw_image.shape[:2]
        for idx, _ in ssd_valid_detections:
            ssd_box = ssd_detection_boxes[idx]
            ssd_class_id = ssd_detection_classes[idx]
            ssd_score = ssd_detection_scores[idx]

            # 정규화된 박스 좌표를 이미지 크기에 맞게 변환
            ymin, xmin, ymax, xmax = ssd_box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # 박스 그리기 (녹색: damage, 파란색: scratch)
            color = (0, 255, 0) if ssd_class_id == 2 else (255, 0, 0)  # BGR 형식
            cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{self.label_map.get(ssd_class_id, 'unknown')} ({ssd_score:.2f})"
            cv2.putText(draw_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # OpenCV 이미지(BGR)를 RGB로 변환 후 QPixmap으로 변환
        draw_image_rgb = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
        height, width, channel = draw_image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(draw_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap_with_boxes = QPixmap.fromImage(q_image).scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        getattr(self, f"{prefix}right_image").setPixmap(pixmap_with_boxes)

    #추론함수 불러오기    
    def run_inference(self, input_tensor, detect_fn):
        try:
            if hasattr(detect_fn, 'structured_input_signature'):
                # 입력 키 확인
                input_keys = list(detect_fn.structured_input_signature[1].keys())
                if input_keys:
                    input_key = input_keys[0]  # 첫 번째 입력 키 사용
                    detections = detect_fn(**{input_key: input_tensor})
                else:
                    detections = detect_fn(input_tensor)
            else:
                # 방법 2: 직접 호출
                detections = detect_fn(input_tensor)
        except Exception as e:
            print(f"첫 번째 추론 방법 실패: {e}")
            # 방법 3: 다른 입력 형식 시도
            try:
                detections = detect_fn(image=input_tensor)
            except Exception as e2:
                print(f"두 번째 추론 방법 실패: {e2}")
                # 방법 4: input_tensor를 직접 전달
                detections = detect_fn(input_tensor)
        return detections
 
    # 체크박스에서 선택된 부분 영상처리
    def cv_process(self):
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    app.exec_()