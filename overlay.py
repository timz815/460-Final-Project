"""
overlay.py — Transparent always-on-top overlay window.
"""

import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QImage, QPen, QColor

import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "renderer_mit", os.path.join(_HERE, "renderer-mit.py"))
renderer_mit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer_mit)

TEXT_FG = (0,   0,   0)
TEXT_BG = (255, 255, 255)
SHRINK  = 0.08


class OverlayWindow(QWidget):

    def __init__(self, screen_rect: QRect, parent=None):
        super().__init__(parent)
        self._screen_rect = screen_rect
        self._bubbles:  list = []
        self._rendered: list = []
        self._bg_style = "white"
        self._setup_window()

    # ── public API ────────────────────────────────────────────────────────────

    def update_bubbles(self, bubbles: list):
        self._bubbles  = bubbles
        self._rendered = []
        for bubble_rect, text_rect, text in bubbles:
            if not text:
                continue
            result = self._render_bubble(bubble_rect, text_rect, text)
            if result:
                self._rendered.append(result)
        self.update()

    def reposition(self, screen_rect: QRect):
        self._screen_rect = screen_rect
        self.setGeometry(screen_rect)
        try:
            import ctypes
            ctypes.windll.user32.SetWindowPos(
                int(self.winId()), -1, 0, 0, 0, 0, 0x0002 | 0x0001)
        except Exception:
            pass
        if self._bubbles:
            self.update_bubbles(self._bubbles)

    def set_bg_style(self, style: str):
        self._bg_style = style
        if self._bubbles:
            self.update_bubbles(self._bubbles)

    # ── setup ─────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(self._screen_rect)
        try:
            import ctypes
            ctypes.windll.user32.SetWindowPos(
                int(self.winId()), -1, 0, 0, 0, 0, 0x0002 | 0x0001)
        except Exception:
            pass

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_bubble(self, bubble_rect: QRect, text_rect: QRect, text: str):
        try:
            if not text_rect:
                return None

            cx = text_rect.x() + text_rect.width()  // 2
            cy = text_rect.y() + text_rect.height() // 2

            fit_rect = bubble_rect if bubble_rect else text_rect
            # Use bubble rect width (wider than tight text rect) to give English
            # more horizontal room, naturally reducing line count and height overflow.
            # Also add a small extra margin on each side.
            WIDEN = 0.10
            w = max(int(fit_rect.width()  * (1.0 + WIDEN * 2)) - int(fit_rect.width()  * SHRINK * 2), 10)
            h = max(fit_rect.height() - int(fit_rect.height() * SHRINK * 2), 10)

            ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
            direction = "v" if (text_rect.height() > text_rect.width() * 1.5 and ascii_ratio < 0.5) else "h"

            # Binary search for largest font size that fits within the bubble.
            font_size = 12
            lo, hi = 12, 32
            while lo <= hi:
                mid = (lo + hi) // 2
                lines, _ = renderer_mit.calc_horizontal(mid, text, w, h)
                if mid * len(lines) <= h:
                    font_size = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if self._bg_style == "black":
                fg = (255, 255, 255)
                bg = (0, 0, 0)
            else:
                fg = TEXT_FG
                bg = TEXT_BG

            rgba = renderer_mit.auto_render(
                text=text, width=w, height=h, font_size=font_size,
                fg=fg, bg=bg, direction=direction, alignment="center",
            )

            if rgba is None or rgba.size == 0:
                return None

            # Composite against background for white/black modes
            if self._bg_style != "transparent":
                bg_col = np.array([255, 255, 255], dtype=np.float32) \
                         if self._bg_style == "white" \
                         else np.array([0, 0, 0], dtype=np.float32)
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                rgb   = rgba[:, :, :3].astype(np.float32)
                comp  = (rgb * alpha + bg_col * (1 - alpha)).clip(0, 255).astype(np.uint8)
                rgba  = np.dstack([comp, np.full(rgba.shape[:2], 255, dtype=np.uint8)])

            rh, rw = rgba.shape[:2]
            lx = cx - rw // 2
            ly = cy - rh // 2

            if self._bg_style != "transparent":
                rgb_c = np.ascontiguousarray(rgba[:, :, :3])
                qimg = QImage(rgb_c.data, rgb_c.shape[1], rgb_c.shape[0],
                              rgb_c.strides[0], QImage.Format.Format_RGB888).copy()
            else:
                arr_c = np.ascontiguousarray(rgba)
                qimg = QImage(arr_c.data, arr_c.shape[1], arr_c.shape[0],
                              arr_c.strides[0], QImage.Format.Format_RGBA8888).copy()

            return QRect(lx, ly, rw, rh), qimg

        except Exception as e:
            print(f"[OverlayWindow] Render error: {e}")
            return None

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._rendered:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._bg_style != "transparent":
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Source)

        for local_rect, qimg in self._rendered:
            painter.drawImage(local_rect.x(), local_rect.y(), qimg)

        # Debug outlines — remove when done
        # painter.setCompositionMode(
        #     QPainter.CompositionMode.CompositionMode_SourceOver)
        # painter.setPen(QPen(QColor(0, 255, 0, 180), 2))
        # for bubble_rect, text_rect, text in self._bubbles:
        #     if bubble_rect:
        #         painter.drawRect(bubble_rect)
        # painter.setPen(QPen(QColor(0, 200, 255, 180), 1))
        # for bubble_rect, text_rect, text in self._bubbles:
        #     if text_rect:
        #         painter.drawRect(text_rect)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._bubbles:
            self.update_bubbles(self._bubbles)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    sg = screen.geometry()
    cx = sg.x() + sg.width()  // 2
    cy = sg.y() + sg.height() // 2
    screen_rect = QRect(cx - 400, cy - 300, 800, 600)
    overlay = OverlayWindow(screen_rect)

    def make_bubble(x, y, w, h, text):
        bubble = QRect(x, y, w, h)
        text_r = QRect(x + 10, y + 10, w - 20, h - 20)
        return bubble, text_r, text

    fake_bubbles = [
        make_bubble(cx - 370, cy - 270, 220, 80,  "This is a translated speech bubble."),
        make_bubble(cx + 80,  cy - 200, 180, 60,  "Another bubble!"),
        make_bubble(cx - 150, cy + 100, 260, 90,  "A longer translated line that wraps."),
        make_bubble(cx - 300, cy + 50,  200, 70,  "这是中文测试"),
        make_bubble(cx - 50,  cy - 50,  60,  160, "竖排文字"),
    ]

    overlay.update_bubbles(fake_bubbles)
    overlay.show()

    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()