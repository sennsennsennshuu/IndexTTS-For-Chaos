#!/usr/bin/env python3
"""
IndexTTS For Chaos - Standalone Desktop Launcher
支持 PyQt6 桌面窗口 / Gradio WebUI 双模式
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 路径自适应
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将 source/ 加入 sys.path（indeexTTS 核心包在此）
SOURCE_DIR = os.path.join(SCRIPT_DIR, "source")
if os.path.exists(SOURCE_DIR) and SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

# 将当前目录加入 sys.path（indextts_engine/window/worker 在此）
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ============================================================
# 模型路径解析
# ============================================================
def find_model_dir():
    """按优先级查找模型目录"""
    # 1. 环境变量
    env = os.environ.get("INDEXTTS_MODEL_DIR")
    if env and os.path.exists(os.path.join(env, "gpt.pth")):
        return env

    # 2. 自动检测
    candidates = [
        os.path.join(SCRIPT_DIR, "checkpoints"),
        os.path.join(SCRIPT_DIR, "..", "checkpoints"),
        os.path.join(SOURCE_DIR, "checkpoints"),
    ]
    for p in candidates:
        abs_p = os.path.normpath(p)
        if os.path.exists(os.path.join(abs_p, "gpt.pth")):
            return abs_p
    return None


# ============================================================
# 桌面窗口模式 (PyQt6)
# ============================================================
def run_desktop(args):
    from PyQt6.QtWidgets import QApplication
    from indextts_window import IndexTTSWindow

    model_dir = args.model_dir or find_model_dir()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置应用图标
    icon_path = os.path.join(SCRIPT_DIR, "Chaos.ico")
    if os.path.exists(icon_path):
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))

    window = IndexTTSWindow(model_path=model_dir)
    window.show()

    sys.exit(app.exec())


# ============================================================
# WebUI 模式 (Gradio) - 精简版
# ============================================================
def run_webui(args):
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    model_dir = args.model_dir or find_model_dir()
    if not model_dir:
        print("[ERROR] Model directory not found!")
        print("  Set INDEXTTS_MODEL_DIR env var or place models in ./checkpoints/")
        sys.exit(1)

    os.environ['HF_HUB_CACHE'] = os.path.join(model_dir, 'hf_cache')

    from indextts.infer_v2 import IndexTTS2
    import time

    config_path = os.path.join(model_dir, "config.yaml")
    print(f"[INFO] Loading model from: {model_dir}")
    tts = IndexTTS2(
        model_dir=model_dir,
        cfg_path=config_path,
        use_fp16=args.fp16,
        use_deepspeed=False,
        use_cuda_kernel=not args.no_cuda_kernel,
    )
    print(f"[INFO] Model loaded. Starting WebUI...")

    def do_infer(prompt_audio, text):
        output_path = os.path.join(SCRIPT_DIR, "outputs", f"spk_{int(time.time())}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = tts.infer(
            spk_audio_prompt=prompt_audio,
            text=text,
            output_path=output_path,
        )
        return result

    with gr.Blocks(title="IndexTTS For Chaos", theme=gr.themes.Soft()) as demo:
        gr.HTML('<h2 style="color:#1a1a1a"><center>IndexTTS For Chaos - 零样本情感语音合成</center></h2>')
        with gr.Row():
            audio_in = gr.Audio(label="Voice Reference", type="filepath", sources=["upload", "microphone"])
            text_in = gr.Textbox(label="Text", placeholder="Enter text to synthesize...", lines=3)
        gen_btn = gr.Button("Generate Speech", variant="primary")
        audio_out = gr.Audio(label="Result")
        gen_btn.click(do_infer, inputs=[audio_in, text_in], outputs=[audio_out])

    demo.launch(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="IndexTTS For Chaos - Standalone Desktop App")
    parser.add_argument("--mode", choices=["desktop", "webui"], default="desktop",
                        help="Launch mode: desktop (PyQt6) or webui (Gradio)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (default: auto-detect)")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use FP16 half precision")
    parser.add_argument("--no_cuda_kernel", action="store_true", default=False,
                        help="Disable CUDA kernel")
    parser.add_argument("--port", type=int, default=7860,
                        help="WebUI port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="WebUI bind address (default: 127.0.0.1)")
    parser.add_argument("--no_browser", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    print("=" * 60)
    print("  IndexTTS For Chaos - Standalone Desktop App")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    if args.mode == "desktop":
        run_desktop(args)
    else:
        run_webui(args)


if __name__ == "__main__":
    main()
