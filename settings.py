"""
settings.py — Settings window for the manga translator.
"""

import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QSettings, QRect, QPoint, Signal, QThread
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum, auto

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

# ─── Palette ──────────────────────────────────────────────────────────────────

BG       = "#1e1e24"
BG_CARD  = "#26262e"
BG_INPUT = "#16161c"
BG_HOVER = "#2a2a33"
FG       = "#c8c8d8"
FG_DIM   = "#64648a"
BORDER   = "#5a5a70"
ACCENT   = "#0ea5e9"
GREEN    = "#22c55e"


def _label(text: str, dim: bool = False, size: int = 13) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {FG_DIM if dim else FG}; font-size: {size}px; font-family: 'Segoe UI', sans-serif; background: transparent; border: none;")
    return lbl


# ─── Capture thread (from ocr_tool.py) ───────────────────────────────────────

class CaptureThread(QThread):
    frame_ready = Signal(QImage, QRect)
    TARGET_FPS  = 15  # Lower FPS for settings preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self._screen_index = 0
        self._running      = False
        self._lock         = __import__('threading').Lock()

    def set_target(self, screen_index: int):
        with self._lock:
            self._screen_index = screen_index

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        self._running = True
        interval = 1.0 / self.TARGET_FPS
        if HAS_MSS:
            self._run_mss(interval)
        else:
            self._run_fallback(interval)

    def _run_mss(self, interval):
        with mss.mss() as sct:
            while self._running:
                t0 = time.monotonic()
                with self._lock:
                    idx = self._screen_index
                try:
                    screens = QApplication.screens()
                    if not screens:
                        time.sleep(interval); continue
                    sidx = min(idx, len(screens) - 1)
                    g    = screens[sidx].geometry()
                    mon  = sct.monitors[sidx + 1] if sidx + 1 < len(sct.monitors) else sct.monitors[1]
                    shot = sct.grab(mon)
                    img  = QImage(shot.rgb, shot.width, shot.height, shot.width * 3, QImage.Format.Format_RGB888)
                    img  = img.scaled(g.width(), g.height(),
                                      Qt.AspectRatioMode.IgnoreAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
                    self.frame_ready.emit(img, g)
                except Exception:
                    time.sleep(interval); continue
                elapsed = time.monotonic() - t0
                sleep_t = max(0.0, interval - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    def _run_fallback(self, interval):
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                idx = self._screen_index
            try:
                screens = QApplication.screens()
                if not screens:
                    time.sleep(interval); continue
                sidx  = min(idx, len(screens) - 1)
                s     = screens[sidx]
                g     = s.geometry()
                img   = s.grabWindow(0).scaled(g.width(), g.height(),
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation).toImage()
                self.frame_ready.emit(img, g)
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)


# ─── Drag mode (from ocr_tool.py) ────────────────────────────────────────────

class DragMode(Enum):
    NONE=auto(); NEW=auto(); MOVE=auto()
    TOP_LEFT=auto(); TOP=auto(); TOP_RIGHT=auto()
    RIGHT=auto(); BOT_RIGHT=auto(); BOT=auto()
    BOT_LEFT=auto(); LEFT=auto()

_CURSOR = {
    DragMode.NONE: Qt.CursorShape.CrossCursor, DragMode.NEW: Qt.CursorShape.CrossCursor,
    DragMode.MOVE: Qt.CursorShape.SizeAllCursor,
    DragMode.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor, DragMode.TOP: Qt.CursorShape.SizeVerCursor,
    DragMode.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor, DragMode.RIGHT: Qt.CursorShape.SizeHorCursor,
    DragMode.BOT_RIGHT: Qt.CursorShape.SizeFDiagCursor, DragMode.BOT: Qt.CursorShape.SizeVerCursor,
    DragMode.BOT_LEFT: Qt.CursorShape.SizeBDiagCursor, DragMode.LEFT: Qt.CursorShape.SizeHorCursor,
}
HANDLE_R = 5
EDGE_HIT  = 6


@dataclass
class ViewTransform:
    screen_rect: QRect; widget_rect: QRect
    scale_x: float = 1.0; scale_y: float = 1.0
    offset_x: float = 0.0; offset_y: float = 0.0

    def __post_init__(self):
        if self.widget_rect.width() > 0 and self.widget_rect.height() > 0:
            self.scale_x  = self.screen_rect.width()  / self.widget_rect.width()
            self.scale_y  = self.screen_rect.height() / self.widget_rect.height()
            self.offset_x = self.widget_rect.x()
            self.offset_y = self.widget_rect.y()

    def widget_to_screen(self, pt):
        return QPoint(int(self.screen_rect.x()+(pt.x()-self.offset_x)*self.scale_x),
                      int(self.screen_rect.y()+(pt.y()-self.offset_y)*self.scale_y))
    def screen_to_widget(self, pt):
        return QPoint(int(self.offset_x+(pt.x()-self.screen_rect.x())/self.scale_x),
                      int(self.offset_y+(pt.y()-self.screen_rect.y())/self.scale_y))
    def widget_rect_to_screen(self, r):
        return QRect(self.widget_to_screen(r.topLeft()), self.widget_to_screen(r.bottomRight())).normalized()
    def screen_rect_to_widget(self, r):
        return QRect(self.screen_to_widget(r.topLeft()), self.screen_to_widget(r.bottomRight())).normalized()
    def clamp_to_widget(self, pt):
        return QPoint(max(self.widget_rect.left(), min(self.widget_rect.right(), pt.x())),
                      max(self.widget_rect.top(),  min(self.widget_rect.bottom(), pt.y())))
    def clamp_to_screen(self, pt):
        return QPoint(max(self.screen_rect.left(), min(self.screen_rect.right(), pt.x())),
                      max(self.screen_rect.top(),  min(self.screen_rect.bottom(), pt.y())))


# ─── Preview widget (adapted from ocr_tool.py) ───────────────────────────────

class ScreenPreviewWidget(QWidget):
    selection_changed = Signal(QRect)

    def __init__(self, screen_index: int = 0, initial_selection: QRect = None, parent=None):
        super().__init__(parent)
        self.screen_index  = screen_index
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 225)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0e0e12; border: none;")

        self._pixmap: Optional[QPixmap] = None
        self._screen_rect  = QRect()
        self._sel: Optional[QRect]         = None
        self._sel_screen: Optional[QRect]  = initial_selection
        self._drag_mode    = DragMode.NONE
        self._drag_origin: Optional[QPoint] = None
        self._drag_sel_screen: Optional[QRect] = None
        self._xform: Optional[ViewTransform]   = None
        self._xform_dirty  = True

        self._thread = CaptureThread(self)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.set_target(screen_index)
        self._thread.start()

    def set_screen_index(self, index: int):
        self.screen_index = index
        self._thread.set_target(index)
        self.clear_selection()

    def clear_selection(self):
        self._sel = self._sel_screen = None
        self._xform_dirty = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        self.selection_changed.emit(QRect())

    def get_screen_selection(self) -> Optional[QRect]:
        return self._sel_screen if (self._sel_screen and self._sel_screen.isValid()) else None

    def set_selection(self, screen_rect: QRect):
        self._sel_screen = screen_rect
        self._xform_dirty = True
        xform = self._ensure_xform()
        if xform:
            self._sel = xform.screen_rect_to_widget(screen_rect)
        self.update()

    def closeEvent(self, e):
        self._thread.stop(); super().closeEvent(e)

    def _ensure_xform(self):
        if not self._xform_dirty:
            return self._xform
        ir = self._calc_image_rect()
        if ir.isValid() and self._screen_rect.isValid():
            self._xform = ViewTransform(self._screen_rect, ir)
        else:
            self._xform = None
        self._xform_dirty = False
        return self._xform

    def _calc_image_rect(self):
        if not self._pixmap or self._pixmap.isNull():
            return QRect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return QRect()
        scale = min(ww / pw, wh / ph)
        iw, ih = int(pw * scale), int(ph * scale)
        return QRect((ww-iw)//2, (wh-ih)//2, iw, ih)

    def _on_frame(self, img: QImage, screen_rect: QRect):
        self._pixmap = QPixmap.fromImage(img)
        self._screen_rect = screen_rect
        self._xform_dirty = True
        if self._sel_screen and self._sel_screen.isValid():
            xform = self._ensure_xform()
            if xform:
                self._sel = xform.screen_rect_to_widget(self._sel_screen)
        self.update()

    def _handle_rects(self, sel: QRect):
        cx = sel.left() + sel.width()//2
        cy = sel.top()  + sel.height()//2
        r  = HANDLE_R
        def hr(x,y): return QRect(x-r,y-r,r*2,r*2)
        return {
            DragMode.TOP_LEFT:  hr(sel.left(),  sel.top()),
            DragMode.TOP:       hr(cx,           sel.top()),
            DragMode.TOP_RIGHT: hr(sel.right(),  sel.top()),
            DragMode.RIGHT:     hr(sel.right(),  cy),
            DragMode.BOT_RIGHT: hr(sel.right(),  sel.bottom()),
            DragMode.BOT:       hr(cx,           sel.bottom()),
            DragMode.BOT_LEFT:  hr(sel.left(),   sel.bottom()),
            DragMode.LEFT:      hr(sel.left(),   cy),
        }

    def _hit_test(self, pt, xform):
        if self._sel is None: return DragMode.NONE
        sel = self._sel
        for mode, rect in self._handle_rects(sel).items():
            if rect.adjusted(-2,-2,2,2).contains(pt): return mode
        if sel.contains(pt):
            if pt.x()-sel.left() < EDGE_HIT:   return DragMode.LEFT
            if sel.right()-pt.x() < EDGE_HIT:  return DragMode.RIGHT
            if pt.y()-sel.top() < EDGE_HIT:    return DragMode.TOP
            if sel.bottom()-pt.y() < EDGE_HIT: return DragMode.BOT
            return DragMode.MOVE
        return DragMode.NONE

    def _apply_drag_screen(self, screen_pt, xform):
        if self._drag_mode == DragMode.NEW:
            origin_screen = xform.widget_to_screen(self._drag_origin)
            return QRect(origin_screen, xform.clamp_to_screen(screen_pt)).normalized()
        if self._drag_sel_screen is None:
            return self._sel_screen or QRect()
        sel = QRect(self._drag_sel_screen)
        drag_origin_screen = xform.widget_to_screen(self._drag_origin)
        dx = screen_pt.x() - drag_origin_screen.x()
        dy = screen_pt.y() - drag_origin_screen.y()
        if self._drag_mode == DragMode.MOVE:
            new = sel.translated(dx, dy)
            sr  = self._screen_rect
            if new.left()   < sr.left():   new.moveLeft(sr.left())
            if new.top()    < sr.top():    new.moveTop(sr.top())
            if new.right()  > sr.right():  new.moveRight(sr.right())
            if new.bottom() > sr.bottom(): new.moveBottom(sr.bottom())
            return new
        l,t,r,b = sel.left(),sel.top(),sel.right(),sel.bottom()
        sr = self._screen_rect
        if self._drag_mode in (DragMode.LEFT,  DragMode.TOP_LEFT,  DragMode.BOT_LEFT):  l = max(sr.left(),  min(l+dx, r-4))
        if self._drag_mode in (DragMode.RIGHT, DragMode.TOP_RIGHT, DragMode.BOT_RIGHT): r = min(sr.right(), max(r+dx, l+4))
        if self._drag_mode in (DragMode.TOP,   DragMode.TOP_LEFT,  DragMode.TOP_RIGHT): t = max(sr.top(),   min(t+dy, b-4))
        if self._drag_mode in (DragMode.BOT,   DragMode.BOT_LEFT,  DragMode.BOT_RIGHT): b = min(sr.bottom(),max(b+dy, t+4))
        return QRect(QPoint(l,t), QPoint(r,b)).normalized()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(14,14,18))
        xform = self._ensure_xform()
        if not (self._pixmap and not self._pixmap.isNull() and xform):
            painter.setPen(QColor(80,85,105)); painter.setFont(QFont("Arial",11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview"); return
        ir = xform.widget_rect
        painter.drawPixmap(ir, self._pixmap)
        painter.setPen(QPen(QColor(35,38,55),1)); painter.setBrush(Qt.BrushStyle.NoBrush); painter.drawRect(ir)
        sel = self._sel
        if not (sel and sel.isValid() and sel.width()>2 and sel.height()>2): return
        painter.setPen(QPen(QColor(0,185,255),1)); painter.setBrush(Qt.BrushStyle.NoBrush); painter.drawRect(sel)
        painter.setClipRect(ir)
        dim = QColor(0,0,0,160)
        ir_l,ir_t,ir_r,ir_b = ir.left(),ir.top(),ir.right(),ir.bottom()
        sl,st,sr2,sb = sel.left(),sel.top(),sel.right(),sel.bottom()
        for x,y,w,h in [(ir_l,ir_t,ir.width(),st-ir_t),(ir_l,sb+1,ir.width(),ir_b-sb),
                        (ir_l,st,sl-ir_l,sel.height()),(sr2+1,st,ir_r-sr2,sel.height())]:
            if w>0 and h>0: painter.fillRect(x,y,w,h,dim)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(0,185,255),1)); painter.setBrush(Qt.BrushStyle.NoBrush); painter.drawRect(sel)
        painter.setPen(QPen(QColor(255,255,255,35),1))
        for i in (1,2):
            painter.drawLine(sl+sel.width()//3*i,st,sl+sel.width()//3*i,sb)
            painter.drawLine(sl,st+sel.height()//3*i,sr2,st+sel.height()//3*i)
        painter.setPen(QPen(QColor(0,185,255),1)); painter.setBrush(QColor(20,20,30))
        handles = self._handle_rects(sel)
        for rect in handles.values(): painter.drawRect(rect)
        painter.setBrush(QColor(0,185,255)); r = HANDLE_R-2
        for rect in handles.values():
            cx2=rect.left()+HANDLE_R; cy2=rect.top()+HANDLE_R
            painter.drawRect(cx2-r,cy2-r,r*2,r*2)
        if self._sel_screen:
            label = f"{self._sel_screen.width()} × {self._sel_screen.height()}"
            font  = QFont("Courier New",9,QFont.Weight.Bold); painter.setFont(font)
            fm=painter.fontMetrics(); lw=fm.horizontalAdvance(label)+14; lh=fm.height()+6
            lx=sel.left(); ly=sel.top()-lh-5
            if ly<0: ly=sel.top()+5
            bg=QRect(lx,ly,lw,lh)
            painter.setBrush(QColor(0,0,0,220)); painter.setPen(Qt.PenStyle.NoPen); painter.drawRoundedRect(bg,3,3)
            painter.setPen(QColor(0,215,255)); painter.drawText(bg,Qt.AlignmentFlag.AlignCenter,label)

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton: return
        xform = self._ensure_xform()
        if xform is None: return
        pt = e.position().toPoint()
        if not xform.widget_rect.contains(pt): return
        hit = self._hit_test(pt, xform)
        if hit == DragMode.NONE:
            self._drag_mode=DragMode.NEW; self._drag_origin=xform.clamp_to_widget(pt)
            self._sel=QRect(self._drag_origin,self._drag_origin)
            self._sel_screen=xform.widget_rect_to_screen(self._sel)
        else:
            self._drag_mode=hit; self._drag_origin=pt
            self._drag_sel_screen=QRect(self._sel_screen) if self._sel_screen else None
        self.update()

    def mouseMoveEvent(self, e):
        pt=e.position().toPoint(); xform=self._ensure_xform()
        if xform is None: return
        if self._drag_mode==DragMode.NONE:
            self.setCursor(_CURSOR.get(self._hit_test(pt,xform),Qt.CursorShape.CrossCursor)); return
        screen_pt=xform.widget_to_screen(pt)
        new=self._apply_drag_screen(screen_pt,xform)
        self._sel_screen=new; self._sel=xform.screen_rect_to_widget(new)
        self.setCursor(_CURSOR.get(self._drag_mode,Qt.CursorShape.CrossCursor)); self.update()

    def mouseReleaseEvent(self, e):
        if e.button()!=Qt.MouseButton.LeftButton: return
        if self._drag_mode==DragMode.NONE: return
        xform=self._ensure_xform()
        if xform:
            pt=e.position().toPoint(); screen_pt=xform.widget_to_screen(pt)
            new=self._apply_drag_screen(screen_pt,xform)
            if new.width()<8 or new.height()<8:
                if self._drag_mode==DragMode.NEW: self._sel=self._sel_screen=None
            else:
                self._sel_screen=new; self._sel=xform.screen_rect_to_widget(new)
        self._drag_mode=DragMode.NONE; self._drag_origin=None; self._drag_sel_screen=None
        if self._sel_screen and self._sel_screen.isValid(): self.selection_changed.emit(self._sel_screen)
        if xform: self.setCursor(_CURSOR.get(self._hit_test(e.position().toPoint(),xform),Qt.CursorShape.CrossCursor))
        self.update()

    def resizeEvent(self, e):
        self._xform_dirty=True
        if self._sel_screen and self._sel_screen.isValid():
            xform=self._ensure_xform()
            if xform: self._sel=xform.screen_rect_to_widget(self._sel_screen)
        super().resizeEvent(e)


# ─── Monitor selector bar ─────────────────────────────────────────────────────

class MonitorBar(QWidget):
    screen_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._layout  = QHBoxLayout(self)
        self._layout.setContentsMargins(0,0,0,0)
        self._layout.setSpacing(6)
        self._buttons: List[QPushButton] = []
        self._current = 0
        self.rebuild()

    def rebuild(self):
        for b in self._buttons: b.deleteLater()
        self._buttons.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i, screen in enumerate(QApplication.screens()):
            g   = screen.geometry()
            btn = QPushButton(f"Display {i+1}  {g.width()}×{g.height()}")
            btn.setCheckable(True); btn.setChecked(i==self._current)
            btn.setProperty("monitor_idx", i)
            btn.clicked.connect(self._clicked)
            btn.setStyleSheet(f"""
                QPushButton {{ background: {'#0a2040' if i==self._current else BG_HOVER};
                    color: {'#55bbff' if i==self._current else FG_DIM};
                    border: 1px solid {'#0077cc' if i==self._current else BORDER};
                    border-radius: 5px; padding: 5px 12px; font-size: 12px;
                    font-family: 'Segoe UI', sans-serif; }}
                QPushButton:hover {{ border-color: {ACCENT}; color: {FG}; }}
            """)
            self._layout.addWidget(btn)
            self._buttons.append(btn)
        self._layout.addStretch()

    def _clicked(self):
        btn = self.sender(); idx = btn.property("monitor_idx")
        self._current = idx
        self.rebuild()
        self.screen_selected.emit(idx)

    def current_index(self): return self._current


# ─── Model path widget ────────────────────────────────────────────────────────

class ModelPathWidget(QWidget):
    path_changed = Signal(str)

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        row = QHBoxLayout(self)
        row.setContentsMargins(0,0,0,0); row.setSpacing(8)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.setReadOnly(True)
        self._edit.setStyleSheet(f"""
            QLineEdit {{ background: {BG_INPUT}; color: {FG}; border: 1px solid {BORDER};
                border-radius: 5px; padding: 6px 10px; font-size: 12px;
                font-family: 'Segoe UI', sans-serif; }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        row.addWidget(self._edit)
        btn = QPushButton("Browse…")
        btn.setFixedHeight(34); btn.setMinimumWidth(80)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{ background: {BG_HOVER}; color: {FG}; border: 1px solid {BORDER};
                border-radius: 5px; padding: 0 14px; font-size: 12px; font-family: 'Segoe UI', sans-serif; }}
            QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
            QPushButton:pressed {{ background: {BG_INPUT}; }}
        """)
        btn.clicked.connect(self._browse)
        row.addWidget(btn)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder",
                                                 self._edit.text() or os.path.expanduser("~"))
        if path:
            self._edit.setText(path); self.path_changed.emit(path)

    def get_path(self): return self._edit.text()
    def set_path(self, p): self._edit.setText(p)


class StatusBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent); self.set_none()

    def set_none(self):  self._set("No model selected", "#64648a", "#2a2a3a")
    def set_ok(self, n): self._set(f"✓  {n}", GREEN, "#1a2e1a")
    def _set(self, text, color, bg):
        self.setText(text)
        self.setStyleSheet(f"color: {color}; background: {bg}; border: 1px solid {color}40; border-radius: 4px; padding: 4px 10px; font-size: 12px; font-family: 'Segoe UI', sans-serif;")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

class ModelTab(QWidget):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setStyleSheet(f"background: {BG}; border: none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20,20,20,20); layout.setSpacing(12)

        from PySide6.QtWidgets import QComboBox

        combo_style = f"""
            QComboBox {{
                background: {BG_INPUT}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 5px;
                padding: 5px 10px; font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QComboBox:focus {{ border-color: {ACCENT}; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {BG_INPUT}; color: {FG};
                border: 1px solid {BORDER};
                selection-background-color: {BG_HOVER};
                outline: none;
                padding: 0px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 5px 10px;
                background: {BG_INPUT};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background: {BG_HOVER};
            }}
        """

        layout.addWidget(_label("Language", size=14))

        from_row = QHBoxLayout()
        from_row.addWidget(_label("From"))
        self._from_lang = QComboBox()
        self._from_lang.addItems(["Chinese", "Japanese", "Korean"])
        self._from_lang.setCurrentText(self._settings.value("from_lang", "Chinese"))
        self._from_lang.setStyleSheet(combo_style)
        self._from_lang.currentTextChanged.connect(
            lambda v: (self._settings.setValue("from_lang", v), self._settings.sync()))
        from_row.addWidget(self._from_lang)
        from_row.addStretch()
        layout.addLayout(from_row)

        to_row = QHBoxLayout()
        to_row.addWidget(_label("To"))
        self._to_lang = QComboBox()
        self._to_lang.addItems(["English"])
        self._to_lang.setStyleSheet(combo_style)
        to_row.addWidget(self._to_lang)
        to_row.addStretch()
        layout.addLayout(to_row)

        layout.addSpacing(8)

        # Title row
        title_row = QHBoxLayout()
        title_row.addWidget(_label("Translation Model", size=14))
        title_row.addStretch()
        self._status = StatusBadge()
        title_row.addWidget(self._status)
        layout.addLayout(title_row)

        desc = _label("Select a HuggingFace model folder.\nLeave blank to use Helsinki-NLP/opus-mt-zh-en (auto-downloaded).",
                      dim=True, size=12)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self._model_path = ModelPathWidget("Default: Helsinki-NLP/opus-mt-zh-en")
        self._model_path.path_changed.connect(self._on_path_changed)
        layout.addWidget(self._model_path)

        btn_clear = QPushButton("Use default (Helsinki)")
        btn_clear.setFixedHeight(28)
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.setStyleSheet(f"QPushButton {{ background: transparent; color: {FG_DIM}; border: none; font-size: 12px; text-decoration: underline; }} QPushButton:hover {{ color: {FG}; }}")
        btn_clear.clicked.connect(self._clear_model)
        layout.addWidget(btn_clear, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addStretch()

        saved = self._settings.value("model_path", "")
        if saved:
            self._model_path.set_path(saved)
            self._status.set_ok(os.path.basename(saved) or saved)

    def _on_path_changed(self, path):
        self._settings.setValue("model_path", path); self._settings.sync()
        self._status.set_ok(os.path.basename(path) or path)

    def _clear_model(self):
        self._model_path.set_path(""); self._settings.setValue("model_path", "")
        self._settings.sync(); self._status.set_none()

    def get_model_path(self): return self._model_path.get_path()


class DisplayTab(QWidget):
    selection_changed = Signal(QRect)

    def __init__(self, settings: QSettings, initial_selection: QRect = None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setStyleSheet(f"background: {BG}; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16,16,16,16); layout.setSpacing(10)
        layout.addWidget(_label("Change Selection", size=14))

        # Monitor selector
        self._monitor_bar = MonitorBar()
        layout.addWidget(self._monitor_bar)

        # Preview
        self._preview = ScreenPreviewWidget(
            screen_index=0,
            initial_selection=initial_selection
        )
        self._preview.selection_changed.connect(self._on_selection)
        layout.addWidget(self._preview, stretch=1)

        # Status
        self._status = _label("Drag to adjust the translation region.", dim=True, size=12)
        layout.addWidget(self._status)

        self._monitor_bar.screen_selected.connect(self._on_screen)

    def _on_screen(self, idx):
        saved = self._preview.get_screen_selection()
        self._preview.set_screen_index(idx)
        if saved and saved.isValid():
            self._preview.set_selection(saved)

    def _on_selection(self, rect: QRect):
        if rect.isValid():
            self._status.setStyleSheet(f"color: {FG}; font-size: 12px; font-family: 'Segoe UI', sans-serif; background: transparent; border: none;")
            self._status.setText(f"Region: {rect.width()} × {rect.height()} at ({rect.x()}, {rect.y()})")
        self.selection_changed.emit(rect)

    def get_selection(self): return self._preview.get_screen_selection()
    #def set_selection(self, rect: QRect): self._preview.set_selection(rect)
    def set_selection(self, rect: QRect):
        self._display_tab.set_selection(rect)

    def closeEvent(self, e):
        self._preview._thread.stop(); super().closeEvent(e)

class OverlayTab(QWidget):
    style_changed = Signal(str)  # 'transparent', 'white', 'black'

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setStyleSheet(f"background: {BG}; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(_label("Text Background", size=14))
        layout.addWidget(_label("Controls what appears behind translated text in the text region.", dim=True, size=12))

        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        self._group = QButtonGroup(self)

        radio_style = f"""
            QRadioButton {{
                color: {FG}; font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
                background: transparent; border: none;
                padding: 4px 0;
            }}
            QRadioButton::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {BORDER}; border-radius: 7px;
                background: {BG_INPUT};
            }}
            QRadioButton::indicator:checked {{
                background: {ACCENT}; border-color: {ACCENT};
            }}
        """

        saved = self._settings.value("overlay_style", "white")

        for label, value in [("Transparent", "transparent"), ("White", "white"), ("Black", "black")]:
            btn = QRadioButton(label)
            btn.setStyleSheet(radio_style)
            btn.setProperty("value", value)
            btn.setChecked(value == saved)
            btn.toggled.connect(self._on_changed)
            self._group.addButton(btn)
            layout.addWidget(btn)

        layout.addStretch()

    def _on_changed(self, checked):
        if checked:
            btn = self.sender()
            value = btn.property("value")
            self._settings.setValue("overlay_style", value)
            self._settings.sync()
            self.style_changed.emit(value)

    def get_style(self) -> str:
        return self._settings.value("overlay_style", "transparent")

class DetectorTab(QWidget):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.setStyleSheet(f"background: {BG}; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(_label("Detection Settings", size=14))

        from PySide6.QtWidgets import QDoubleSpinBox

        spin_style = f"""
            QDoubleSpinBox {{
                background: {BG_INPUT}; color: {FG};
                border: 1px solid {BORDER}; border-radius: 5px;
                padding: 4px 8px; font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
            }}
            QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
        """

        form = QHBoxLayout()
        form_col = QVBoxLayout()
        form_col.setSpacing(12)

        # Idle interval
        row1 = QHBoxLayout()
        row1.addWidget(_label("Check every (seconds)"))
        self._idle = QDoubleSpinBox()
        self._idle.setRange(1.0, 60.0)
        self._idle.setSingleStep(0.5)
        self._idle.setValue(float(self._settings.value("idle_interval", 5.0)))
        self._idle.setStyleSheet(spin_style)
        self._idle.setFixedWidth(80)
        self._idle.valueChanged.connect(lambda v: self._save("idle_interval", v))
        row1.addWidget(self._idle)
        row1.addStretch()
        form_col.addLayout(row1)

        # Burst interval
        row2 = QHBoxLayout()
        row2.addWidget(_label("Burst check every (seconds)"))
        self._burst = QDoubleSpinBox()
        self._burst.setRange(0.1, 5.0)
        self._burst.setSingleStep(0.1)
        self._burst.setValue(float(self._settings.value("burst_interval", 0.5)))
        self._burst.setStyleSheet(spin_style)
        self._burst.setFixedWidth(80)
        self._burst.valueChanged.connect(lambda v: self._save("burst_interval", v))
        row2.addWidget(self._burst)
        row2.addStretch()
        form_col.addLayout(row2)

        # Stable duration
        row3 = QHBoxLayout()
        row3.addWidget(_label("Stable before translate again (seconds)"))
        self._stable = QDoubleSpinBox()
        self._stable.setRange(0.5, 10.0)
        self._stable.setSingleStep(0.5)
        self._stable.setValue(float(self._settings.value("stable_duration", 1.0)))
        self._stable.setStyleSheet(spin_style)
        self._stable.setFixedWidth(80)
        self._stable.valueChanged.connect(lambda v: self._save("stable_duration", v))
        row3.addWidget(self._stable)
        row3.addStretch()
        form_col.addLayout(row3)

        layout.addLayout(form_col)
        layout.addStretch()

    def _save(self, key, value):
        self._settings.setValue(key, value)
        self

# ─── Settings Window ─────────────────────────────────────────────────────────

class SettingsWindow(QMainWindow):
    model_changed     = Signal(str)
    selection_changed = Signal(QRect)

    def __init__(self, initial_selection: QRect = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 480)
        self.resize(680, 520)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        self._settings = QSettings("MangaTranslator", "Translator")
        self._settings.clear()

        central = QWidget()
        central.setStyleSheet(f"background: {BG}; border: none;")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {BG}; }}
            QTabBar::tab {{ background: transparent; color: {FG_DIM}; padding: 10px 20px;
                font-size: 13px; font-family: 'Segoe UI', sans-serif;
                border: none; border-bottom: 2px solid transparent; }}
            QTabBar::tab:selected {{ color: {FG}; border-bottom: 2px solid {ACCENT}; }}
            QTabBar::tab:hover {{ color: {FG}; }}
        """)

        self._model_tab   = ModelTab(self._settings)
        self._display_tab = DisplayTab(self._settings, initial_selection)
        self._display_tab.selection_changed.connect(self.selection_changed)
        self._overlay_tab = OverlayTab(self._settings)
        self._detector_tab = DetectorTab(self._settings)


        self._tabs.addTab(self._model_tab,   "Model")
        self._tabs.addTab(self._display_tab, "Display")
        self._tabs.addTab(self._overlay_tab, "Overlay")
        self._tabs.addTab(self._detector_tab, "Detector")

        layout.addWidget(self._tabs)

    def get_model_path(self): return self._model_tab.get_model_path()
    def get_selection(self):  return self._display_tab.get_selection()
    def get_overlay_style(self) -> str:
        return self._overlay_tab.get_style()
    def get_detector_settings(self) -> dict:
        return self._detector_tab.get_values()

    def closeEvent(self, e):
        self.model_changed.emit(self.get_model_path())
        self._display_tab.closeEvent(e)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SettingsWindow()
    win.model_changed.connect(lambda p: print(f"Model: {p!r}"))
    win.selection_changed.connect(lambda r: print(f"Selection: {r.width()}×{r.height()}"))
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
