"""
main.py — Entry point for the manga translator.

Startup sequence:
    1. Show toolbar immediately
    2. Load models in background thread
    3. Once ready, enable Start button
    4. On Start: PipelineWorker captures region, runs pipeline, feeds overlay
    5. On Stop: worker stops
"""

import sys
import os
import time
import numpy as np
import cv2

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QThread, QRect, Signal, QObject, QTimer
from PySide6.QtGui import QImage

from toolbar import Toolbar
from overlay import OverlayWindow
from pipeline_worker import PipelineWorker


os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — runs in background thread
# ─────────────────────────────────────────────────────────────────────────────

class ModelLoader(QThread):
    finished   = Signal(object, object, object)  # detector, ocr, translator
    progress   = Signal(str)
    model_step = Signal(int, int)               # (loaded, total)
    failed     = Signal(str)

    def run(self):
        try:
            self.progress.emit("Loading bubble detector…")
            from bubble_detector_ct import BubbleDetectorCT
            detector = BubbleDetectorCT(device='cuda', confidence_threshold=0.3)
            detector.load()
            self.model_step.emit(1, 3)

            self.progress.emit("Loading OCR…")
            from ocr_onnx import OCRonnx
            ocr = OCRonnx(device='cuda', min_confidence=0.0)
            ocr.load()
            self.model_step.emit(2, 3)
############################################################
            self.progress.emit("Loading translator…")
            from PySide6.QtCore import QSettings
            settings = QSettings("MangaTranslator", "Translator")
            model_path = settings.value("model_path", "")
            print(f"[Loader] model_path = {model_path!r}")  # debug

            if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                from unsloth import FastLanguageModel
                import torch
                print("[Loader] Loading LoRA model...")
                trans_model, trans_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    device_map="auto",
                )
                print("[Loader] Model loaded, setting inference mode...")
                FastLanguageModel.for_inference(trans_model)
                print("[Loader] Translator ready!")

                def translator(text: str) -> str:
                    msgs = [{"role": "user", "content": text}]
                    prompt = trans_tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    inputs = trans_tokenizer(prompt, return_tensors="pt").to(trans_model.device)
                    with torch.inference_mode():
                        out = trans_model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=trans_tokenizer.eos_token_id,
                            repetition_penalty=1.3,
                        )
                    gen = out[0][inputs.input_ids.shape[-1]:]
                    return trans_tokenizer.decode(gen, skip_special_tokens=True).strip()

            else:
                from transformers import pipeline
                lang_map = {
                    "Chinese":  "Helsinki-NLP/opus-mt-zh-en",
                    "Japanese": "Helsinki-NLP/opus-mt-ja-en",
                    "Korean":   "Helsinki-NLP/opus-mt-ko-en",
                }
                from_lang = settings.value("from_lang", "Chinese")
                model_id = model_path if model_path else lang_map.get(from_lang, "Helsinki-NLP/opus-mt-zh-en")
                trans_pipe = pipeline("translation", model=model_id, device=-1)

                def translator(text: str) -> str:
                    return trans_pipe([text], max_length=512)[0]["translation_text"]

            self.model_step.emit(3, 3)
            self.finished.emit(detector, ocr, translator)

        except Exception as e:
            self.failed.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker — runs translation loop
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# App controller
# ─────────────────────────────────────────────────────────────────────────────

class App(QObject):
    def __init__(self):
        super().__init__()
        self._detector   = None
        self._ocr        = None
        self._translator = None
        self._worker     = None

        # Toolbar
        self._toolbar = Toolbar()
        self._toolbar.translate_start.connect(self._on_start)
        self._toolbar.translate_stop.connect(self._on_stop)
        # Overlay — covers full virtual desktop in logical coords
        self._overlay = OverlayWindow(QRect(0, 0, 1, 1))
        self._overlay.show()
        try:
            import ctypes
            ctypes.windll.user32.SetWindowPos(int(self._overlay.winId()), -1, 0, 0, 0, 0, 0x0002 | 0x0001)
        except Exception:
            pass
        self._toolbar.overlay_style_changed.connect(self._overlay.set_bg_style)
        # Position toolbar top-center
        screen = QApplication.primaryScreen().geometry()
        self._toolbar.adjustSize()
        self._toolbar.move(screen.center().x() - self._toolbar.width() // 2,
                           screen.y() + 40)
        self._toolbar.show()
        try:
            import ctypes
            hwnd = int(self._toolbar.winId())
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001)
        except Exception:
            pass

        # Disable start until models loaded
        self._toolbar._btn_start.setEnabled(False)
        self._toolbar._btn_start.setText("⏳ Loading…")

        # Load models
        self._loader = ModelLoader()
        self._loader.progress.connect(self._on_load_progress)
        self._loader.model_step.connect(self._toolbar.set_progress)
        self._loader.finished.connect(self._on_models_ready)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.start()

    # ── model loading ─────────────────────────────────────────────────────────

    def _on_load_progress(self, msg: str):
        print(f"[Loader] {msg}")

    def _on_models_ready(self, detector, ocr, translator):
        self._detector   = detector
        self._ocr        = ocr
        self._translator = translator
        print("[Loader] All models ready")
        self._toolbar._btn_start.setText("  ▶  Start")
        self._toolbar.mark_models_ready()
        self._toolbar.set_progress(0, 1)

    def _on_load_failed(self, error: str):
        print(f"[Loader] Failed: {error}")
        self._toolbar._btn_start.setText("  ✕  Error")

    # ── pipeline ──────────────────────────────────────────────────────────────

    def _on_start(self):
        region = self._toolbar._region
        if not region or not self._detector:
            return
        self._worker = PipelineWorker(region, self._detector, self._ocr, self._translator)
        self._worker.results_ready.connect(self._on_results)
        self._worker.overlay_clear.connect(lambda: self._overlay.update_bubbles([]))
        self._worker.overlay_clear.connect(lambda: self._toolbar.set_progress(0, 1))
        self._worker.translation_progress.connect(self._toolbar.set_progress)
        self._worker.status.connect(lambda s: print(f"[Pipeline] {s}"))
        self._worker.start()
        print(f"[App] Pipeline started on {region.width()}×{region.height()} region")

    def _on_stop(self):
        if self._worker:
            self._worker.finished.connect(self._on_worker_done)
            self._worker.stop()

    def _on_worker_done(self):
        if self._worker:
            self._worker = None
        self._overlay.update_bubbles([])
        self._toolbar.set_progress(0, 1)
        print("[App] Pipeline stopped")

    def _on_results(self, bubbles: list):
        if bubbles:
            # Resize overlay to cover just the selected region
            self._overlay.reposition(self._toolbar._region)
        self._overlay.update_bubbles(bubbles)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")

    controller = App()

    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
