"""
toolbar.py — Compact always-on-top translator toolbar.
"""

import sys
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QMenu, QRubberBand, QProgressBar
from PySide6.QtCore import Qt, QRect, QPoint, QSize, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QKeySequence, QShortcut
from settings import SettingsWindow

BG       = "#1e1e24"
BG_HOVER = "#2a2a33"
BG_MENU  = "#26262e"
FG       = "#c8c8d8"
FG_DIM   = "#4a4a60"
BORDER   = "#5a5a70"
GREEN    = "#22c55e"


def _plain_style(color: str, hover: bool = True) -> str:
    hover_rule = f"QPushButton:hover {{ background: {BG_HOVER}; }}" if hover else ""
    return f"""
        QPushButton {{
            background: transparent; color: {color}; border: none;
            border-radius: 4px; padding: 0 8px;
            font-size: 13px; font-family: 'Segoe UI', sans-serif; font-weight: 500;
        }}
        {hover_rule}
        QPushButton:pressed {{ background: {BG_HOVER if hover else 'transparent'}; }}
    """

def _bordered_style(border_color: str) -> str:
    return f"""
        QPushButton {{
            background: transparent; color: {FG};
            border: 2px solid {border_color}; border-radius: 5px;
            padding: 0 8px; font-size: 13px;
            font-family: 'Segoe UI', sans-serif; font-weight: 500;
        }}
        QPushButton:hover {{ background: {BG_HOVER}; }}
        QPushButton:pressed {{ background: {BG_HOVER}; }}
    """


class SelectButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("⬚  Select", parent)
        self.setFixedHeight(32); self.setMinimumWidth(90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = False; self._refresh()

    def mark_selected(self):
        self._selected = True; self._refresh()

    def reset(self):
        self._selected = False; self._refresh()

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._refresh()

    def _refresh(self):
        if not self.isEnabled():
            self.setStyleSheet(_plain_style(FG_DIM, hover=False))
        elif self._selected:
            self.setStyleSheet(_bordered_style(BORDER))
        else:
            self.setStyleSheet(_plain_style(FG))

class TextButton(QPushButton):
    def __init__(self, icon: str, label: str, parent=None):
        super().__init__(f"  {icon}  {label}", parent)
        self.setFixedHeight(32); self.setMinimumWidth(80)
        self._sync(False)

    def _sync(self, enabled: bool):
        self.setStyleSheet(_plain_style(FG if enabled else FG_DIM, hover=enabled))
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled); self._sync(enabled)


class DotsButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("···", parent)
        self.setFixedHeight(32); self.setMinimumWidth(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(_plain_style(FG) + "QPushButton { font-size: 16px; letter-spacing: 1px; padding-bottom: 4px; }")

class CloseButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("✕", parent)
        self.setFixedSize(30, 32); self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {FG}; border: none;
                border-radius: 4px; font-size: 12px; font-weight: 600;
                font-family: 'Segoe UI', sans-serif; }}
            QPushButton:hover {{ background: #ef4444; color: white; }}
            QPushButton:pressed {{ background: #dc2626; color: white; }}
        """)


class Divider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(1, 18)
        self.setStyleSheet(f"background: {BORDER};")


# ─── Screen selector — uses grabKeyboard like the inspiration code ────────────

class ScreenSelector(QWidget):
    selected  = Signal(QRect)
    cancelled = Signal()

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._start:     QPoint | None = None
        self._end:       QPoint | None = None
        self._selecting: bool = False

        # ESC shortcut
        esc = QShortcut(QKeySequence("Escape"), self)
        esc.activated.connect(self._cancel)

    def start(self, geo):
        self.setGeometry(geo)
        self.show()
        self.raise_()
        self.activateWindow()
        self.grabKeyboard()
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── painting ────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark overlay over everything
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        if self._selecting and self._start and self._end:
            rect = QRect(self._start, self._end).normalized()

            # Punch a clear hole for the selected region
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)

            # Draw border around the hole
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(FG), 2))
            painter.drawRect(rect)

            # Dimensions label
            text = f"{rect.width()} × {rect.height()}"
            painter.setPen(Qt.GlobalColor.white)
            text_rect = painter.boundingRect(
                self.rect(), Qt.AlignmentFlag.AlignLeft, text)
            ty = rect.top() - text_rect.height() - 8
            if ty < 0:
                ty = rect.top() + 8
            bg = text_rect.adjusted(-6, -3, 6, 3)
            bg.moveTo(rect.left(), ty)
            painter.fillRect(bg, QColor(0, 0, 0, 200))
            painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, text)

            # Crosshair at cursor
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1,
                                Qt.PenStyle.DashLine))
            painter.drawLine(self._end.x(), 0,
                             self._end.x(), self.height())
            painter.drawLine(0, self._end.y(),
                             self.width(), self._end.y())

    # ── mouse ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._start     = e.position().toPoint()
            self._end       = self._start
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, e):
        if self._selecting:
            self._end = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._end       = e.position().toPoint()
            self._selecting = False
            rect = QRect(self._start, self._end).normalized()
            self.hide()
            if rect.width() > 10 and rect.height() > 10:
                self.selected.emit(
                    QRect(self.mapToGlobal(rect.topLeft()), rect.size()))
            else:
                self.cancelled.emit()
            self.releaseKeyboard()
            self.close()

    # ── keyboard ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(e)

    def _cancel(self):
        self.releaseKeyboard()
        self.cancelled.emit()
        self.close()

    def closeEvent(self, e):
        self.releaseKeyboard()
        super().closeEvent(e)


# ─── Toolbar ──────────────────────────────────────────────────────────────────

class Toolbar(QWidget):
    region_selected      = Signal(QRect)
    translate_start      = Signal()
    translate_stop       = Signal()
    settings_clicked     = Signal()
    overlay_style_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_pos = QPoint()
        self._settings_win = None
        self._region   = None
        self._models_ready = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedHeight(62)
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        pill = QWidget()
        pill.setObjectName("pill")
        pill.setStyleSheet(f"#pill {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; }}")

        pill_layout = QVBoxLayout(pill)
        pill_layout.setContentsMargins(8, 4, 8, 4)
        pill_layout.setSpacing(3)

        row = QHBoxLayout()
        row.setSpacing(6)

        self._btn_select = SelectButton()
        self._btn_select.setToolTip("Draw a region to translate")
        self._btn_select.clicked.connect(self._on_select)
        row.addWidget(self._btn_select)

        row.addWidget(Divider())

        self._btn_start = TextButton("▶", "Start")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start)
        row.addWidget(self._btn_start)

        self._btn_stop = TextButton("■", "Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        row.addWidget(self._btn_stop)

        row.addSpacing(8)

        self._btn_dots = DotsButton()
        self._btn_dots.clicked.connect(self._on_dots)
        row.addWidget(self._btn_dots)

        btn_close = CloseButton()
        btn_close.setToolTip("Close")
        #btn_close.clicked.connect(QApplication.quit)
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)

        pill_layout.addLayout(row)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 3)
        self._progress.setValue(0)
        self._progress.setStyleSheet("""
            QProgressBar {
                background: #2a2a33;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 2px;
            }
        """)
        pill_layout.addWidget(self._progress)

        outer.addWidget(pill)

        self._menu = QMenu(self)
        self._menu.setWindowFlags(self._menu.windowFlags() | Qt.WindowType.NoDropShadowWindowHint)
        self._menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._menu.setStyleSheet(f"""
            QMenu {{ background: {BG_MENU}; border: 1px solid {BORDER}; border-radius: 8px;
                padding: 4px; color: {FG}; font-size: 13px; font-family: 'Segoe UI', sans-serif; }}
            QMenu::item {{ padding: 6px 16px; border-radius: 5px; }}
            QMenu::item:selected {{ background: {BG_HOVER}; }}
        """)
        self._menu.addAction("Settings").triggered.connect(self._open_settings)
        
    def _on_dots(self):
        btn = self._btn_dots
        self._menu.exec(btn.mapToGlobal(QPoint(0, btn.height() + 4)))

    def _on_select(self):
        self.hide()
        QTimer.singleShot(150, self._launch_selector)

    def _launch_selector(self):
        self._selectors = []
        for screen in QApplication.screens():
            sel = ScreenSelector()
            sel.selected.connect(self._on_region)
            sel.selected.connect(lambda _: self._close_selectors())
            sel.cancelled.connect(self._close_selectors)
            self._selectors.append(sel)
            sel.start(screen.geometry())

    def _close_selectors(self):
        for sel in self._selectors:
            sel.close()
        self._selectors.clear()
        self._show_again()

    def _show_again(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def set_progress(self, value: int, maximum: int):
        self._progress.setRange(0, maximum if maximum > 0 else 1)
        self._progress.setValue(value)

    def mark_models_ready(self):
        self._models_ready = True
        if self._region:
            self._btn_start.setEnabled(True)

    def _on_region(self, rect: QRect):
        self._region = rect
        self._btn_select.mark_selected()
        # Only enable start if models are already loaded
        if self._models_ready:
            self._btn_start.setEnabled(True)
        self.region_selected.emit(rect)

    def _on_start(self):
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_select.setEnabled(False)
        self.translate_start.emit()

    def _on_stop(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_select.setEnabled(True)
        self.translate_stop.emit()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton and not self._drag_pos.isNull():
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = QPoint()

    def _open_settings(self):
            if self._settings_win is None or not self._settings_win.isVisible():
                self._settings_win = SettingsWindow(initial_selection=self._region)
                self._settings_win.selection_changed.connect(self._on_region)
                self._settings_win._overlay_tab.style_changed.connect(self.overlay_style_changed)
                self._settings_win.show()
            else:
                if self._region:
                    self._settings_win.set_selection(self._region)
                self._settings_win.raise_()
                self._settings_win.activateWindow()
                
    def closeEvent(self, e):
        QApplication.quit()

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    tb = Toolbar()
    tb.region_selected.connect(lambda r: print(f"Region {r.width()}×{r.height()}"))
    tb.translate_start.connect(lambda: print("▶ Start"))
    tb.translate_stop.connect(lambda: print("■ Stop"))
    tb.settings_clicked.connect(lambda: print("Settings"))
    screen = app.primaryScreen().geometry()
    tb.adjustSize()
    tb.move(screen.center().x() - tb.width() // 2, screen.y() + 40)
    tb.show()
    import signal; signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
