#!/usr/bin/env python3
"""
IndexTTS For Chaos - Standalone Desktop Launcher
PyQt6 桌面窗口模式
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
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="IndexTTS For Chaos - Standalone Desktop App")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (default: auto-detect)")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use FP16 half precision")
    parser.add_argument("--no_cuda_kernel", action="store_true", default=False,
                        help="Disable CUDA kernel")
    args = parser.parse_args()

    print("=" * 60)
    print("  IndexTTS For Chaos - Standalone Desktop App")
    print("  Mode: desktop (PyQt6)")
    print("=" * 60)

    run_desktop(args)


if __name__ == "__main__":
    main()
