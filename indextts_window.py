"""
IndexTTS For Chaos - 零样本情感语音合成窗口界面
直接在窗口内调用 Python API 进行语音合成
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTextEdit, QGroupBox, QMessageBox, 
    QComboBox, QLineEdit, QSlider, QSpinBox,
    QFileDialog, QDoubleSpinBox, QWidget
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# 确保当前目录在 sys.path 中（便携分发兼容）
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from indextts_engine import IndexTTSEngine
from indextts_worker import IndexTTSWorker


class IndexTTSWindow(QDialog):
    """IndexTTS For Chaos 窗口"""
    
    # 信号定义：音频生成完成时发送 (audio_path, text)
    audio_generated = pyqtSignal(str, str)
    
    def __init__(self, model_path=None, parent=None):
        super().__init__(parent)
        self.engine = None
        self.player = None
        self.audio_output = None
        self.worker = None  # 工作线程引用
        self._model_path = model_path  # 可外部传入模型路径
        
        # 设置窗口标志：添加最小化和最大化按钮
        self.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.WindowMinimizeButtonHint | 
            Qt.WindowType.WindowMaximizeButtonHint | 
            Qt.WindowType.WindowCloseButtonHint
        )
        
        self.setWindowTitle("IndexTTS For Chaos - 零样本情感语音合成")
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)
        
        self.init_ui()
        self.apply_white_style()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # === 标题 ===
        title_label = QLabel("IndexTTS For Chaos - 零样本情感语音合成")
        title_font = QFont("Microsoft YaHei", 14, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # === 状态信息 ===
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: gray; font-size: 12px;")
        main_layout.addWidget(self.status_label)
        
        # === 文本输入 ===
        text_group = QGroupBox("文本输入")
        text_layout = QVBoxLayout(text_group)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("请输入要合成的文本...")
        self.text_input.setMinimumHeight(100)
        text_layout.addWidget(self.text_input)
        
        main_layout.addWidget(text_group)
        
        # === 音色参考音频 ===
        speaker_group = QGroupBox("音色参考音频")
        speaker_layout = QHBoxLayout(speaker_group)
        
        self.speaker_audio_input = QLineEdit()
        self.speaker_audio_input.setPlaceholderText("选择音色参考音频文件 (.wav, .mp3)")
        speaker_layout.addWidget(self.speaker_audio_input)
        
        speaker_browse_btn = QPushButton("浏览...")
        speaker_browse_btn.setFixedWidth(80)
        speaker_browse_btn.clicked.connect(self.browse_speaker_audio)
        speaker_layout.addWidget(speaker_browse_btn)
        
        main_layout.addWidget(speaker_group)
        
        # === 情感控制 ===
        emo_group = QGroupBox("情感控制")
        emo_layout = QVBoxLayout(emo_group)
        
        # 情感控制方式选择
        emo_method_layout = QHBoxLayout()
        emo_method_layout.addWidget(QLabel("控制方式:"))
        
        self.emo_method_combo = QComboBox()
        self.emo_method_combo.addItems([
            "与音色参考音频相同",
            "使用情感参考音频",
            "使用情感向量控制",
            "使用情感描述文本控制"
        ])
        self.emo_method_combo.currentIndexChanged.connect(self.on_emo_method_changed)
        emo_method_layout.addWidget(self.emo_method_combo)
        emo_method_layout.addStretch()
        
        emo_layout.addLayout(emo_method_layout)
        
        # 情感参考音频 (模式1)
        self.emo_audio_widget = QWidget()
        emo_audio_layout = QHBoxLayout(self.emo_audio_widget)
        self.emo_audio_input = QLineEdit()
        self.emo_audio_input.setPlaceholderText("选择情感参考音频")
        emo_audio_layout.addWidget(self.emo_audio_input)
        
        emo_audio_browse = QPushButton("浏览...")
        emo_audio_browse.setFixedWidth(80)
        emo_audio_browse.clicked.connect(self.browse_emo_audio)
        emo_audio_layout.addWidget(emo_audio_browse)
        
        emo_layout.addWidget(self.emo_audio_widget)
        
        # 情感权重滑块 (模式1,2,3)
        self.emo_weight_widget = QWidget()
        emo_weight_layout = QHBoxLayout(self.emo_weight_widget)
        emo_weight_layout.addWidget(QLabel("情感权重:"))
        
        self.emo_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.emo_weight_slider.setMinimum(0)
        self.emo_weight_slider.setMaximum(100)
        self.emo_weight_slider.setValue(65)
        self.emo_weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.emo_weight_slider.setTickInterval(10)
        emo_weight_layout.addWidget(self.emo_weight_slider)
        
        self.emo_weight_label = QLabel("0.65")
        self.emo_weight_label.setFixedWidth(50)
        emo_weight_layout.addWidget(self.emo_weight_label)
        
        self.emo_weight_slider.valueChanged.connect(
            lambda v: self.emo_weight_label.setText(f"{v/100:.2f}")
        )
        
        emo_layout.addWidget(self.emo_weight_widget)
        
        # 情感向量滑块 (模式2)
        self.emo_vector_widget = QWidget()
        emo_vector_layout = QVBoxLayout(self.emo_vector_widget)
        
        emo_names = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]
        self.emo_sliders = []
        
        for name in emo_names:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{name}:"))
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(20)
            slider_layout.addWidget(slider)
            
            value_label = QLabel("0.0")
            value_label.setFixedWidth(40)
            slider_layout.addWidget(value_label)
            
            slider.valueChanged.connect(
                lambda v, lbl=value_label: lbl.setText(f"{v/100:.1f}")
            )
            
            self.emo_sliders.append((slider, value_label))
            emo_vector_layout.addLayout(slider_layout)
        
        emo_layout.addWidget(self.emo_vector_widget)
        
        # 情感描述文本 (模式3)
        self.emo_text_widget = QWidget()
        emo_text_layout = QVBoxLayout(self.emo_text_widget)
        emo_text_layout.addWidget(QLabel("情感描述:"))
        
        self.emo_text_input = QLineEdit()
        self.emo_text_input.setPlaceholderText("例如: 委屈巴巴、危险在悄悄逼近")
        emo_text_layout.addWidget(self.emo_text_input)
        
        emo_layout.addWidget(self.emo_text_widget)
        
        main_layout.addWidget(emo_group)
        
        # === 高级参数 ===
        advanced_group = QGroupBox("高级参数")
        advanced_layout = QVBoxLayout(advanced_group)
        
        params_layout = QHBoxLayout()
        
        # temperature
        temp_layout = QVBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setMinimum(0.1)
        self.temp_spin.setMaximum(2.0)
        self.temp_spin.setValue(0.8)
        self.temp_spin.setSingleStep(0.1)
        temp_layout.addWidget(self.temp_spin)
        params_layout.addLayout(temp_layout)
        
        # top_p
        top_p_layout = QVBoxLayout()
        top_p_layout.addWidget(QLabel("Top P:"))
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setMinimum(0.0)
        self.top_p_spin.setMaximum(1.0)
        self.top_p_spin.setValue(0.8)
        self.top_p_spin.setSingleStep(0.01)
        top_p_layout.addWidget(self.top_p_spin)
        params_layout.addLayout(top_p_layout)
        
        # top_k
        top_k_layout = QVBoxLayout()
        top_k_layout.addWidget(QLabel("Top K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setMinimum(0)
        self.top_k_spin.setMaximum(100)
        self.top_k_spin.setValue(30)
        self.top_k_spin.setSingleStep(1)  # 设置步长为1，与TopP一致
        top_k_layout.addWidget(self.top_k_spin)
        params_layout.addLayout(top_k_layout)
        
        # max tokens
        tokens_layout = QVBoxLayout()
        tokens_layout.addWidget(QLabel("最大Token数:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setMinimum(20)
        self.max_tokens_spin.setMaximum(500)
        self.max_tokens_spin.setValue(120)
        self.max_tokens_spin.setSingleStep(1)  # 设置步长为1，与TopP一致
        tokens_layout.addWidget(self.max_tokens_spin)
        params_layout.addLayout(tokens_layout)
        
        params_layout.addStretch()
        advanced_layout.addLayout(params_layout)
        
        main_layout.addWidget(advanced_group)
        
        # === 控制按钮 ===
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.generate_btn = QPushButton("▶ 生成语音")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_speech)
        btn_layout.addWidget(self.generate_btn)
        
        close_btn = QPushButton("❌ 关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 初始化显示状态
        self.on_emo_method_changed(0)
    
    def apply_white_style(self):
        """应用白色背景样式"""
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: black;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: black;
            }
            QLabel {
                background-color: transparent;
                color: black;
            }
            QTextEdit, QLineEdit {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #e0e0e0;
                selection-color: black;
                border: 1px solid #ccc;
                outline: 0px;
            }
            QComboBox::drop-down {
                border: none;
                background: transparent;
            }
            QComboBox QAbstractItemView::item {
                background-color: white;
                color: black;
                border: none;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #e0e0e0;
                color: black;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: #f0f0f0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #45a049;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 3px;
            }
        """)
    
    def get_indextts_model_path(self):
        """获取 IndexTTS 模型路径（优先级：构造传入 > 环境变量 > 自动检测）"""
        # 1. 构造时传入的路径
        if self._model_path:
            if os.path.exists(os.path.join(self._model_path, "gpt.pth")):
                return self._model_path

        # 2. 环境变量
        env_path = os.environ.get("INDEXTTS_MODEL_DIR")
        if env_path and os.path.exists(os.path.join(env_path, "gpt.pth")):
            return env_path

        # 3. 自动检测：程序所在目录下的 checkpoints/
        app_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(app_dir, "checkpoints"),
            os.path.join(app_dir, "..", "checkpoints"),
            os.path.join(app_dir, "source", "checkpoints"),
        ]
        for path in candidates:
            abs_path = os.path.normpath(path)
            if os.path.exists(os.path.join(abs_path, "gpt.pth")):
                return abs_path

        # 4. 旧的 config.json 方式（兼容）
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "config", "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    local_models = config.get("local_models", {})
                    model_paths = local_models.get("model_paths", [])
                    for path in model_paths:
                        if os.path.exists(os.path.join(path, "gpt.pth")):
                            return path
            except Exception as e:
                print(f"读取配置失败: {e}")

        return None
    
    def browse_speaker_audio(self):
        """浏览选择音色参考音频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音色参考音频",
            "",
            "音频文件 (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.speaker_audio_input.setText(file_path)
    
    def browse_emo_audio(self):
        """浏览选择情感参考音频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择情感参考音频",
            "",
            "音频文件 (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.emo_audio_input.setText(file_path)
    
    def on_emo_method_changed(self, index):
        """情感控制方式改变时的处理"""
        # 模式0: 与音色相同 - 只显示权重
        # 模式1: 情感参考音频 - 显示音频+权重
        # 模式2: 情感向量 - 显示向量+权重
        # 模式3: 情感文本 - 显示文本+权重
        
        self.emo_audio_widget.setVisible(index == 1)
        self.emo_vector_widget.setVisible(index == 2)
        self.emo_text_widget.setVisible(index == 3)
        self.emo_weight_widget.setVisible(index in [1, 2, 3])
    
    def collect_emotion_params(self):
        """收集情感参数"""
        emo_mode = self.emo_method_combo.currentIndex()
        
        params = {
            "emo_mode": emo_mode,
            "emo_weight": self.emo_weight_slider.value() / 100.0
        }
        
        if emo_mode == 1:
            params["emo_audio"] = self.emo_audio_input.text()
        elif emo_mode == 2:
            params["emo_vector"] = [
                slider.value() / 100.0 
                for slider, _ in self.emo_sliders
            ]
        elif emo_mode == 3:
            params["emo_text"] = self.emo_text_input.text()
        
        return params
    
    def generate_speech(self):
        """异步生成语音（使用工作线程）"""
        # 验证输入
        text = self.text_input.toPlainText().strip()
        speaker_audio = self.speaker_audio_input.text().strip()
        
        if not text:
            QMessageBox.warning(self, "警告", "请输入要合成的文本")
            return
        
        if not speaker_audio:
            QMessageBox.warning(self, "提示", "请选择音色参考音频")
            return
        
        if not os.path.exists(speaker_audio):
            QMessageBox.critical(self, "错误", f"音色参考音频文件不存在:\n{speaker_audio}")
            return
        
        # 禁用按钮，显示准备状态
        self.generate_btn.setEnabled(False)
        self.status_label.setText("准备生成...")
        self.status_label.setStyleSheet("color: blue; font-size: 12px;")
        
        # 获取模型路径
        model_path = self.get_indextts_model_path()
        if not model_path:
            QMessageBox.critical(self, "错误", "未找到模型路径\n\n请设置 INDEXTTS_MODEL_DIR 环境变量或放置模型到 checkpoints/ 目录")
            self.generate_btn.setEnabled(True)
            self.status_label.setText("就绪")
            self.status_label.setStyleSheet("color: gray; font-size: 12px;")
            return
        
        # 收集情感参数
        emo_params = self.collect_emotion_params()
        
        # 收集高级参数
        advanced_params = {
            "temperature": self.temp_spin.value(),
            "top_p": self.top_p_spin.value(),
            "top_k": self.top_k_spin.value(),
            "max_text_tokens_per_segment": self.max_tokens_spin.value()
        }
        
        # 创建工作线程
        self.worker = IndexTTSWorker(text, speaker_audio, emo_params, advanced_params, model_path)
        self.worker.progress_update.connect(self.update_status)
        self.worker.status_message.connect(self.send_status_to_extension)  # 发送状态到扩展界面
        self.worker.generation_finished.connect(self.on_generation_finished)
        self.worker.generation_error.connect(self.on_generation_error)
        self.worker.finished.connect(self.cleanup_worker)
        self.worker.start()
    
    def update_status(self, status_text):
        """更新状态标签（由工作线程信号调用）"""
        self.status_label.setText(status_text)
    
    def send_status_to_extension(self, message):
        """将状态消息发送到扩展界面的聊天区域"""
        try:
            # 查找父窗口（MainWindow）
            main_window = None
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'extension_panel') and parent.extension_panel:
                    main_window = parent
                    break
                parent = parent.parent()
            
            # 如果找到主窗口和扩展面板，则插入消息
            if main_window and hasattr(main_window, 'extension_panel') and main_window.extension_panel:
                # 使用扩展面板的 insert_status_message 方法
                main_window.extension_panel.insert_status_message(message)
        except Exception as e:
            print(f"发送状态到扩展界面失败: {e}")
    
    def on_generation_finished(self, audio_path):
        """生成完成回调（由工作线程信号调用）"""
        # 播放音频
        self.play_generated_audio(audio_path)
        self.status_label.setText("✅ 生成完成")
        self.status_label.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
        self.generate_btn.setEnabled(True)
        
        # 发送信号到扩展面板
        text = self.text_input.toPlainText().strip()[:100]  # 截取前100字符
        self.audio_generated.emit(audio_path, text)
    
    def on_generation_error(self, error_msg):
        """生成错误回调（由工作线程信号调用）"""
        # 记录错误到日志文件
        self._log_indextts_error(Exception(error_msg))
        
        # 显示错误信息
        self.status_label.setText(f"❌ 错误: {error_msg[:50]}...")
        self.status_label.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
        QMessageBox.warning(self, "IndexTTS For Chaos", f"生成失败:\n{error_msg}")
        self.generate_btn.setEnabled(True)
    
    def cleanup_worker(self):
        """清理工作线程"""
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def _log_indextts_error(self, error: Exception):
        """将错误记录到日志文件"""
        try:
            # 创建 logs 目录
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(base_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"indextts_error_{timestamp}.log")
            
            # 获取模型文件夹路径
            extend_models_path = os.path.join(base_dir, "extend_models")
            
            # 写入错误信息
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("IndexTTS For Chaos - 错误日志\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误类型: {type(error).__name__}\n")
                f.write(f"错误信息: {str(error)}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"模型文件夹路径: {extend_models_path}\n")
                f.write(f"当前模型目录: {self.engine.model_dir if hasattr(self, 'engine') and self.engine else '未加载'}\n\n")
                f.write("堆栈跟踪:\n")
                f.write(traceback.format_exc())
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"[IndexTTS For Chaos] 错误日志已保存到: {log_file}")
            
        except Exception as log_error:
            print(f"[IndexTTS For Chaos] 写入错误日志失败: {log_error}")
    
    def play_generated_audio(self, audio_path):
        """播放生成的音频"""
        if not os.path.exists(audio_path):
            QMessageBox.warning(self, "警告", f"音频文件不存在: {audio_path}")
            return
        
        try:
            # 初始化播放器
            if self.player is None:
                self.player = QMediaPlayer()
                self.audio_output = QAudioOutput()
                self.player.setAudioOutput(self.audio_output)
            
            # 设置音频源并播放
            self.player.setSource(QUrl.fromLocalFile(audio_path))
            self.player.play()
            
            QMessageBox.information(self, "成功", f"音频已生成并播放:\n{audio_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"播放失败: {str(e)}\n\n但音频已保存到:\n{audio_path}")
    
    def closeEvent(self, event):
        """窗口关闭时卸载模型"""
        if self.engine:
            self.engine.unload_model()
        
        if self.player:
            self.player.stop()
        
        event.accept()
