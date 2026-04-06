"""
PipelineWorker — Smart page-change detection and translation pipeline.

States:
    IDLE     — checking every IDLE_INTERVAL seconds via phash
    MOVING   — page is changing, overlay hidden, burst checking every BURST_INTERVAL
    SETTLING — page looks stable, counting down STABLE_COUNT checks before translating
    RUNNING  — full pipeline executing (detect → OCR → translate → overlay)
"""

import time
import numpy as np
import cv2
from PIL import Image
import imagehash

from PySide6.QtCore import QThread, QRect, Signal
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor


IDLE_INTERVAL   = 2.0   # seconds between checks when stable
BURST_INTERVAL  = 0.5   # seconds between checks when moving
STABLE_COUNT    = 2     # consecutive stable checks before translating
HASH_THRESHOLD  = 8     # phash hamming distance to consider "changed"


def _phash(frame: np.ndarray) -> imagehash.ImageHash:
    return imagehash.phash(Image.fromarray(frame))


class PipelineWorker(QThread):
    results_ready        = Signal(list)   # list of (bubble_rect, text_rect, text)
    overlay_clear        = Signal()       # clear overlay — page is moving
    translation_progress = Signal(int, int)  # (translated, total)
    status               = Signal(str)

    def __init__(self, region: QRect, detector, ocr, translator, parent=None):
        super().__init__(parent)
        self._region     = region
        self._detector   = detector
        self._ocr        = ocr
        self._translator = translator
        self._running    = False

        # Load settings
        from PySide6.QtCore import QSettings
        s = QSettings("MangaTranslator", "Translator")
        self._idle_interval  = float(s.value("idle_interval",   5.0))
        self._burst_interval = float(s.value("burst_interval",  0.5))
        stable_duration      = float(s.value("stable_duration", 1.0))
        self._cur_stable   = max(1, int(stable_duration / self._burst_interval))

        # State
        self._prev_hash    = None
        self._cur_stable   = 0
        self._state        = "IDLE"

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            import mss
            with mss.mss() as sct:
                # Run pipeline immediately on first start
                self._state = "SETTLING"
                while self._running:
                    t0 = time.monotonic()

                    if self._state == "IDLE":
                        self._check(sct)
                        self._sleep_interruptible(self._idle_interval)

                    elif self._state == "MOVING":
                        changed = self._check(sct)
                        if not changed:
                            self._cur_stable += 1
                            if self._cur_stable >= self._cur_stable:
                                self._state = "SETTLING"
                                self.status.emit("Page settled, translating…")
                        else:
                            self._cur_stable = 0
                        self._sleep_interruptible(self._burst_interval)

                    elif self._state == "SETTLING":
                        self._run_pipeline(sct)
                        self._state = "IDLE"
                        self._cur_stable = 0

        except ImportError:
            self.status.emit("mss not installed")

    def _sleep_interruptible(self, duration: float):
        slept = 0.0
        while self._running and slept < duration:
            time.sleep(0.1)
            slept += 0.1

    def _capture(self, sct) -> np.ndarray:
        r     = self._region
        ratio = 1.0
        for screen in QApplication.screens():
            if screen.geometry().contains(r.topLeft()):
                ratio = screen.devicePixelRatio()
                break
        mon = {
            "left":   int(r.x()      * ratio),
            "top":    int(r.y()      * ratio),
            "width":  int(r.width()  * ratio),
            "height": int(r.height() * ratio),
        }
        shot = sct.grab(mon)
        return np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3), ratio

    def _check(self, sct) -> bool:
        """Capture frame, compute phash, compare to previous. Returns True if changed."""
        try:
            frame, _ = self._capture(sct)
            h = _phash(frame)
            if self._prev_hash is None:
                self._prev_hash = h
                return False
            dist = h - self._prev_hash
            changed = dist >= HASH_THRESHOLD
            if changed:
                self._prev_hash = h
                if self._state == "IDLE":
                    self._state = "MOVING"
                    self._cur_stable = 0
                    self.overlay_clear.emit()
                    self.status.emit("Page changing…")
            return changed
        except Exception as e:
            self.status.emit(f"Check error: {e}")
            return False

    def _run_pipeline(self, sct):
        """Full detect → OCR → translate → emit results."""
        try:
            frame, ratio = self._capture(sct)

            self.status.emit("Detecting…")
            det_results = self._detector.detect(frame)
            if not det_results:
                self.status.emit("No bubbles found")
                return

            bubbles = [(text_rect, np.array([
                [text_rect.x(),                   text_rect.y()                   ],
                [text_rect.x()+text_rect.width(),  text_rect.y()                   ],
                [text_rect.x()+text_rect.width(),  text_rect.y()+text_rect.height()],
                [text_rect.x(),                   text_rect.y()+text_rect.height()],
            ], dtype=np.float32)) for _, text_rect, _ in det_results]

            self.status.emit("Reading text…")
            ocr_results = self._ocr.read_bubbles(frame, bubbles)
            if not ocr_results:
                self.status.emit("No text found")
                return

            self.status.emit("Translating…")
            texts = [text for _, text in ocr_results]
            self.translation_progress.emit(0, len(texts))
            en_texts = []
            for i, text in enumerate(texts):
                en_texts.append(self._translator(text))
                self.translation_progress.emit(i + 1, len(texts))


            text_rect_to_bubble = {
                id(text_rect): bubble_rect
                for bubble_rect, text_rect, _ in det_results
            }

            overlay_bubbles = []
            for (ocr_text_rect, _), en_text in zip(ocr_results, en_texts):
                screen_text_rect = QRect(
                    int(ocr_text_rect.x()      / ratio),
                    int(ocr_text_rect.y()      / ratio),
                    int(ocr_text_rect.width()  / ratio),
                    int(ocr_text_rect.height() / ratio),
                )
                raw_bubble = text_rect_to_bubble.get(id(ocr_text_rect))
                if raw_bubble is not None:
                    screen_bubble_rect = QRect(
                        int(raw_bubble.x()      / ratio),
                        int(raw_bubble.y()      / ratio),
                        int(raw_bubble.width()  / ratio),
                        int(raw_bubble.height() / ratio),
                    )
                else:
                    screen_bubble_rect = screen_text_rect
                overlay_bubbles.append((screen_bubble_rect, screen_text_rect, en_text))

            self.results_ready.emit(overlay_bubbles)
            self.status.emit(f"Translated {len(overlay_bubbles)} bubble(s)")

            # Wait for overlay to render, then snapshot WITH overlay so that
            # the next IDLE check doesn't falsely detect it as a new page.
            self._sleep_interruptible(0.5)
            frame_with_overlay, _ = self._capture(sct)
            self._prev_hash = _phash(frame_with_overlay)

        except Exception as e:
            self.status.emit(f"Pipeline error: {e}")
            print(f"[Pipeline] Error: {e}")
