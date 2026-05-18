"""
IndexTTS For Chaos 工作线程
在后台线程中执行音频生成，避免界面卡顿
"""
from PyQt6.QtCore import QThread, pyqtSignal
import os
import sys

# 确保当前目录在 sys.path 中（便携分发兼容）
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from indextts_engine import IndexTTSEngine


class IndexTTSWorker(QThread):
    """IndexTTS For Chaos 音频生成工作线程"""
    
    # 信号定义
    generation_finished = pyqtSignal(str)  # 音频路径
    generation_error = pyqtSignal(str)     # 错误信息
    progress_update = pyqtSignal(str)      # 进度文本（用于窗口状态栏）
    status_message = pyqtSignal(str)       # 状态消息（用于扩展界面显示）
    
    def __init__(self, text, speaker_audio, emo_params, advanced_params, model_path):
        """
        初始化工作线程
        
        Args:
            text: 要合成的文本
            speaker_audio: 音色参考音频路径
            emo_params: 情感参数字典
            advanced_params: 高级参数字典
            model_path: 模型文件夹路径
        """
        super().__init__()
        self.text = text
        self.speaker_audio = speaker_audio
        self.emo_params = emo_params
        self.advanced_params = advanced_params
        self.model_path = model_path
    
    def run(self):
        """执行音频生成任务（在后台线程中运行）"""
        try:
            # 步骤1: 加载模型
            loading_msg = "IndexTTS For Chaos 正在加载模型"
            self.progress_update.emit("正在加载模型...")
            self.status_message.emit(loading_msg)
            engine = IndexTTSEngine(self.model_path)
            engine.load_model()
            
            # 步骤2: 生成音频
            generating_msg = f"🎵 正在生成语音: {self.text[:50]}{'...' if len(self.text) > 50 else ''}"
            self.progress_update.emit("正在生成音频...")
            self.status_message.emit(generating_msg)
            output_path = engine.infer(
                text=self.text,
                speaker_audio=self.speaker_audio,
                **self.emo_params,
                **self.advanced_params
            )
            
            # 步骤3: 发送完成信号
            success_msg = f"✅ 音频生成完成！\n文件: {os.path.basename(output_path)}"
            self.status_message.emit(success_msg)
            self.generation_finished.emit(output_path)
            
        except Exception as e:
            # 发送错误信号
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.generation_error.emit(error_msg)
