"""
IndexTTS For Chaos 推理引擎封装
提供文本转语音的核心功能
"""
import os
import sys
import torch
from typing import Optional, Dict, Any


class IndexTTSEngine:
    """IndexTTS For Chaos 推理引擎封装"""
    
    def __init__(self, model_dir: str, use_fp16: bool = True):
        """
        IndexTTS For Chaos 引擎

        Args:
            model_dir: 模型目录路径 (包含 gpt.pth, s2mel.pth, config.yaml 等)
            use_fp16: 是否使用 FP16 精度加速 (默认启用)
        """
        self.model_dir = model_dir
        # 如果检测到 GPU 可用，默认使用 FP16
        import torch
        if torch.cuda.is_available():
            self.use_fp16 = use_fp16
            print(f"[IndexTTS For Chaos] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[IndexTTS For Chaos] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.use_fp16 = False
            print("[IndexTTS For Chaos] No GPU, using CPU mode")
        self.tts_model = None
        self.is_loaded = False
        
        # 验证模型文件
        self._validate_model_files()
    
    def _validate_model_files(self):
        """验证必需的模型文件是否存在"""
        required_files = [
            "gpt.pth", "s2mel.pth", "config.yaml",
            "bpe.model", "wav2vec2bert_stats.pt"
        ]
        missing = []
        for f in required_files:
            if not os.path.exists(os.path.join(self.model_dir, f)):
                missing.append(f)
        
        if missing:
            raise FileNotFoundError(
                f"缺少必需的模型文件: {', '.join(missing)}"
            )
    
    def load_model(self):
        """加载 IndexTTS For Chaos 模型"""
        if self.is_loaded:
            return
        
        # 添加 source 目录到 sys.path，确保 indextts 包可被导入
        # indextts_engine.py 所在目录的 source/ 子目录包含 indextts 包
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(engine_dir, "source")
        
        if os.path.exists(source_dir):
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
                print(f"[IndexTTS For Chaos] Added source path: {source_dir}")
        else:
            # Fallback: try engine_dir itself (backward compat)
            if engine_dir not in sys.path:
                sys.path.insert(0, engine_dir)
                print(f"[IndexTTS For Chaos] Added fallback path: {engine_dir}")
        
        try:
            from indextts.infer_v2 import IndexTTS2
            
            self.tts_model = IndexTTS2(
                model_dir=self.model_dir,
                cfg_path=os.path.join(self.model_dir, "config.yaml"),
                use_fp16=self.use_fp16,
                use_deepspeed=False,
                use_cuda_kernel=True  # 启用 CUDA 加速
            )
            self.is_loaded = True
            
        except ImportError as e:
            # 获取项目根目录和模型文件夹路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(current_dir))
            extend_models_path = os.path.join(base_dir, "extend_models")
            
            raise RuntimeError(
                f"无法导入 indextts 模块。请确保依赖已安装:\n"
                f"错误详情: {str(e)}\n\n"
                f"模型文件夹路径: {extend_models_path}\n"
                f"当前源码路径: {current_dir}\n\n"
                f"解决方案:\n"
                f" 检查 extend_models 目录下是否有 IndexTTS 模型文件夹"
                f"2. 确保已安装依赖（运行 pip install -r requirements.txt）"
                f"3. 查看错误日志文件获取更多详细信息"
            )
        except FileNotFoundError as e:
            # 文件不存在，需要下载模型
            raise RuntimeError(
                f"模型文件缺失: {str(e)}\n\n"
                f" 自动从 HuggingFace 下载所需模型"
            )
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    def infer(
        self,
        text: str,
        speaker_audio: str,
        output_path: str = None,
        emo_mode: int = 0,
        emo_audio: str = None,
        emo_weight: float = 0.65,
        emo_vector: list = None,
        emo_text: str = None,
        **kwargs
    ) -> str:
        """
        执行文本转语音推理
        
        Args:
            text: 目标文本
            speaker_audio: 音色参考音频路径
            output_path: 输出音频路径 (可选,默认自动生成)
            emo_mode: 情感控制模式
                     0: 与音色参考音频相同
                     1: 使用情感参考音频
                     2: 使用情感向量控制
                     3: 使用情感描述文本控制
            emo_audio: 情感参考音频路径 (emo_mode=1 时使用)
            emo_weight: 情感权重 (0.0-1.0)
            emo_vector: 情感向量列表 [喜,怒,哀,惧,厌恶,低落,惊喜,平静]
            emo_text: 情感描述文本 (emo_mode=3 时使用)
            **kwargs: 其他生成参数 (top_p, top_k, temperature 等)
        
        Returns:
            生成的音频文件路径
        """
        if not self.is_loaded:
            self.load_model()
        
        if not output_path:
            import time
            output_path = os.path.join(
                self.model_dir, 
                "outputs", 
                f"tts_{int(time.time())}.wav"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 处理情感向量
        if emo_mode == 2 and emo_vector:
            emo_vector = self.tts_model.normalize_emo_vec(emo_vector, apply_bias=True)
        
        # 调用推理
        result = self.tts_model.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio if emo_mode == 1 else None,
            emo_alpha=emo_weight if emo_mode in [1, 2, 3] else 1.0,
            emo_vector=emo_vector if emo_mode == 2 else None,
            use_emo_text=(emo_mode == 3),
            emo_text=emo_text if emo_mode == 3 else None,
            verbose=False,
            **kwargs
        )
        
        return result
    
    def unload_model(self):
        """卸载模型释放显存"""
        if self.tts_model:
            del self.tts_model
            self.tts_model = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
