"""
renderer.py — Self-contained manga-style text renderer.

Extracted and cleaned up from manga-image-translator's text_render.py.
No dependency on the MIT package structure — only needs:
    freetype, numpy, cv2, langcodes, pyhyphen

Provides:
    render_text_horizontal(text, width, height, font_size, fg, bg) -> np.ndarray (RGBA)
    render_text_vertical(text, height, font_size, fg, bg)          -> np.ndarray (RGBA)
    auto_render(text, width, height, font_size, fg, bg, direction) -> np.ndarray (RGBA)

The returned RGBA image can be warped into the detected bubble quadrilateral
and composited onto the overlay window.
"""

import os
import re
import cv2
import numpy as np
import freetype
import functools
import logging
from pathlib import Path
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ─────────────────────────────────────────────────────────────────────────────
# Font setup
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(_HERE, "fonts")

FALLBACK_FONTS = [
    os.path.join(FONTS_DIR, "msyh.ttc"),
    os.path.join(FONTS_DIR, "Arial-Unicode-Regular.ttf"),
    os.path.join(FONTS_DIR, "msgothic.ttc"),
]

FONT_SELECTION: List[freetype.Face] = []
_font_cache = {}


def _get_cached_font(path: str) -> freetype.Face:
    path = path.replace("\\", "/")
    if path not in _font_cache:
        _font_cache[path] = freetype.Face(Path(path).open("rb"))
    return _font_cache[path]


def set_font(font_path: str = ""):
    global FONT_SELECTION
    if font_path and os.path.isfile(font_path):
        selection = [font_path] + FALLBACK_FONTS
    else:
        selection = FALLBACK_FONTS
    FONT_SELECTION = [_get_cached_font(p) for p in selection if os.path.isfile(p)]


set_font()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_punctuation(char: str) -> bool:
    punc = set('。，、；：？！…—～·「」『』【】〔〕《》〈〉""\'\'()（）[]{}.,;:?!-')
    return char in punc


def compact_special_symbols(text: str) -> str:
    text = text.replace("...", "\u2026")
    text = text.replace("..", "\u2026")
    text = re.sub(r"([^\w\s])[ \u3000]+", r"\1", text)
    return text


class _NS:
    pass


class Glyph:
    def __init__(self, glyph):
        self.bitmap = _NS()
        self.bitmap.buffer = glyph.bitmap.buffer
        self.bitmap.rows = glyph.bitmap.rows
        self.bitmap.width = glyph.bitmap.width
        self.advance = _NS()
        self.advance.x = glyph.advance.x
        self.advance.y = glyph.advance.y
        self.bitmap_left = glyph.bitmap_left
        self.bitmap_top = glyph.bitmap_top
        self.metrics = _NS()
        self.metrics.vertBearingX = glyph.metrics.vertBearingX
        self.metrics.vertBearingY = glyph.metrics.vertBearingY
        self.metrics.horiBearingX = glyph.metrics.horiBearingX
        self.metrics.horiBearingY = glyph.metrics.horiBearingY
        self.metrics.horiAdvance = glyph.metrics.horiAdvance
        self.metrics.vertAdvance = glyph.metrics.vertAdvance


@functools.lru_cache(maxsize=1024, typed=True)
def get_char_glyph(cdpt: str, font_size: int, direction: int) -> Glyph:
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        else:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt)
        return Glyph(face.glyph)


def get_char_border(cdpt: str, font_size: int, direction: int):
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        else:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
        return face.glyph.get_glyph()


# ─────────────────────────────────────────────────────────────────────────────
# Color compositing
# ─────────────────────────────────────────────────────────────────────────────

def add_color(
    bw_char_map: np.ndarray,
    color: Tuple[int, int, int],
    stroke_char_map: np.ndarray,
    stroke_color: Optional[Tuple[int, int, int]],
) -> np.ndarray:
    if bw_char_map.size == 0:
        return np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype=np.uint8)

    if stroke_color is None:
        x, y, w, h = cv2.boundingRect(bw_char_map)
    else:
        x, y, w, h = cv2.boundingRect(stroke_char_map)

    fg = np.zeros((h, w, 4), dtype=np.uint8)
    fg[:, :, 0] = color[0]
    fg[:, :, 1] = color[1]
    fg[:, :, 2] = color[2]
    fg[:, :, 3] = bw_char_map[y:y + h, x:x + w]

    if stroke_color is None:
        stroke_color = color
    bg = np.zeros((stroke_char_map.shape[0], stroke_char_map.shape[1], 4), dtype=np.uint8)
    bg[:, :, 0] = stroke_color[0]
    bg[:, :, 1] = stroke_color[1]
    bg[:, :, 2] = stroke_color[2]
    bg[:, :, 3] = stroke_char_map

    fg_alpha = fg[:, :, 3] / 255.0
    bg_alpha = 1.0 - fg_alpha
    bg[y:y + h, x:x + w, :] = (
        fg_alpha[:, :, np.newaxis] * fg[:, :, :]
        + bg_alpha[:, :, np.newaxis] * bg[y:y + h, x:x + w, :]
    )
    return bg


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal character width helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_char_offset_x(font_size: int, cdpt: str) -> int:
    glyph = get_char_glyph(cdpt, font_size, 0)
    bitmap = glyph.bitmap
    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        return glyph.advance.x >> 6
    return glyph.metrics.horiAdvance >> 6


def get_string_width(font_size: int, text: str) -> int:
    return sum(get_char_offset_x(font_size, c) for c in text)


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal layout
# ─────────────────────────────────────────────────────────────────────────────

def calc_horizontal(
    font_size: int,
    text: str,
    max_width: int,
    max_height: int,
) -> Tuple[List[str], List[int]]:
    """Split text into lines that fit within max_width."""
    max_width = max(max_width, 2 * font_size)
    whitespace_w = get_char_offset_x(font_size, " ")

    words = re.split(r"\s+", text) if " " in text else list(text)
    word_widths = [get_string_width(font_size, w) for w in words]

    line_text_list: List[str] = []
    line_width_list: List[int] = []
    line_words: List[str] = []
    line_width = 0

    for word, word_w in zip(words, word_widths):
        gap = whitespace_w if line_width > 0 else 0
        if line_width + gap + word_w <= max_width:
            line_words.append(word)
            line_width += gap + word_w
        else:
            if line_words:
                sep = " " if " " in text else ""
                line_text_list.append(sep.join(line_words))
                line_width_list.append(line_width)
            line_words = [word]
            line_width = word_w

    if line_words:
        sep = " " if " " in text else ""
        line_text_list.append(sep.join(line_words))
        line_width_list.append(line_width)

    return line_text_list, line_width_list


def put_char_horizontal(
    font_size: int,
    cdpt: str,
    pen: List[int],
    canvas_text: np.ndarray,
    canvas_border: np.ndarray,
    border_size: int,
) -> int:
    slot = get_char_glyph(cdpt, font_size, 0)
    bitmap = slot.bitmap

    char_offset_x = slot.metrics.horiAdvance >> 6 if slot.metrics.horiAdvance else slot.advance.x >> 6

    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        return char_offset_x

    bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
    char_place_x = pen[0] + slot.bitmap_left
    char_place_y = pen[1] - slot.bitmap_top

    py0 = max(0, char_place_y)
    px0 = max(0, char_place_x)
    py1 = min(canvas_text.shape[0], char_place_y + bitmap.rows)
    px1 = min(canvas_text.shape[1], char_place_x + bitmap.width)

    if py0 < py1 and px0 < px1:
        src = bitmap_char[py0 - char_place_y:py1 - char_place_y, px0 - char_place_x:px1 - char_place_x]
        if src.shape == canvas_text[py0:py1, px0:px1].shape:
            canvas_text[py0:py1, px0:px1] = src

    if border_size > 0:
        try:
            glyph_border = get_char_border(cdpt, font_size, 0)
            stroker = freetype.Stroker()
            stroker.set(
                64 * max(int(0.07 * font_size), 1),
                freetype.FT_STROKER_LINEJOIN_ROUND,
                freetype.FT_STROKER_LINECAP_ROUND,
                0,
            )
            glyph_border.stroke(stroker, destroy=True)
            blyph = glyph_border.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
            bb = blyph.bitmap
            if bb.rows * bb.width > 0 and len(bb.buffer) == bb.rows * bb.width:
                bmap = np.array(bb.buffer, dtype=np.uint8).reshape((bb.rows, bb.width))
                bx = int(round(char_place_x + bitmap.width / 2 - bb.width / 2))
                by = int(round(char_place_y + bitmap.rows / 2 - bb.rows / 2))
                by0 = max(0, by); bx0 = max(0, bx)
                by1 = min(canvas_border.shape[0], by + bb.rows)
                bx1 = min(canvas_border.shape[1], bx + bb.width)
                if by0 < by1 and bx0 < bx1:
                    src_b = bmap[by0 - by:by1 - by, bx0 - bx:bx1 - bx]
                    tgt = canvas_border[by0:by1, bx0:bx1]
                    if src_b.shape == tgt.shape:
                        canvas_border[by0:by1, bx0:bx1] = cv2.add(tgt, src_b)
        except Exception:
            pass

    return char_offset_x


def put_text_horizontal(
    font_size: int,
    text: str,
    width: int,
    height: int,
    fg: Tuple[int, int, int],
    bg: Optional[Tuple[int, int, int]],
    alignment: str = "center",
    line_spacing: int = 0,
) -> np.ndarray:
    text = compact_special_symbols(text)
    if not text:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_y = int(font_size * (line_spacing or 0.01))

    line_text_list, line_width_list = calc_horizontal(font_size, text, width, height)

    canvas_w = max(line_width_list) + (font_size + bg_size) * 2
    canvas_h = font_size * len(line_width_list) + spacing_y * (len(line_width_list) - 1) + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    pen_orig = [font_size + bg_size, font_size + bg_size]

    for line_text, line_width in zip(line_text_list, line_width_list):
        pen_line = pen_orig.copy()
        if alignment == "center":
            pen_line[0] += (max(line_width_list) - line_width) // 2
        elif alignment == "right":
            pen_line[0] += max(line_width_list) - line_width

        for c in line_text:
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)
    x, y, w, h = cv2.boundingRect(canvas_border)
    if w == 0 or h == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    return line_box[y:y + h, x:x + w]


# ─────────────────────────────────────────────────────────────────────────────
# Vertical layout
# ─────────────────────────────────────────────────────────────────────────────

def calc_vertical(font_size: int, text: str, max_height: int) -> Tuple[List[str], List[int]]:
    line_text_list: List[str] = []
    line_height_list: List[int] = []
    line_str = ""
    line_height = 0

    for cdpt in text:
        if line_height == 0 and cdpt == " ":
            continue
        slot = get_char_glyph(cdpt, font_size, 1)
        bitmap = slot.bitmap
        if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
            char_offset_y = slot.metrics.vertAdvance >> 6
        else:
            char_offset_y = slot.metrics.vertAdvance >> 6

        if line_height + char_offset_y > max_height:
            line_text_list.append(line_str)
            line_height_list.append(line_height)
            line_str = ""
            line_height = 0

        line_height += char_offset_y
        line_str += cdpt

    line_text_list.append(line_str)
    line_height_list.append(line_height)
    return line_text_list, line_height_list


def put_char_vertical(
    font_size: int,
    cdpt: str,
    pen: List[int],
    canvas_text: np.ndarray,
    canvas_border: np.ndarray,
    border_size: int,
) -> int:
    slot = get_char_glyph(cdpt, font_size, 1)
    bitmap = slot.bitmap

    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        return slot.metrics.vertAdvance >> 6

    char_offset_y = slot.metrics.vertAdvance >> 6
    bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
    char_place_x = pen[0] + (slot.metrics.vertBearingX >> 6)
    char_place_y = pen[1] + (slot.metrics.vertBearingY >> 6)

    py0 = max(0, char_place_y); px0 = max(0, char_place_x)
    py1 = min(canvas_text.shape[0], char_place_y + bitmap.rows)
    px1 = min(canvas_text.shape[1], char_place_x + bitmap.width)

    if py0 < py1 and px0 < px1:
        src = bitmap_char[py0 - char_place_y:py1 - char_place_y, px0 - char_place_x:px1 - char_place_x]
        if src.shape == canvas_text[py0:py1, px0:px1].shape:
            canvas_text[py0:py1, px0:px1] = src

    if border_size > 0:
        try:
            glyph_border = get_char_border(cdpt, font_size, 1)
            stroker = freetype.Stroker()
            stroker.set(
                64 * max(int(0.07 * font_size), 1),
                freetype.FT_STROKER_LINEJOIN_ROUND,
                freetype.FT_STROKER_LINECAP_ROUND,
                0,
            )
            glyph_border.stroke(stroker, destroy=True)
            blyph = glyph_border.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
            bb = blyph.bitmap
            if bb.rows * bb.width > 0 and len(bb.buffer) == bb.rows * bb.width:
                bmap = np.array(bb.buffer, dtype=np.uint8).reshape((bb.rows, bb.width))
                bx = int(round(char_place_x + bitmap.width / 2 - bb.width / 2))
                by = int(round(char_place_y + bitmap.rows / 2 - bb.rows / 2))
                by0 = max(0, by); bx0 = max(0, bx)
                by1 = min(canvas_border.shape[0], by + bb.rows)
                bx1 = min(canvas_border.shape[1], bx + bb.width)
                if by0 < by1 and bx0 < bx1:
                    src_b = bmap[by0 - by:by1 - by, bx0 - bx:bx1 - bx]
                    tgt = canvas_border[by0:by1, bx0:bx1]
                    if src_b.shape == tgt.shape:
                        canvas_border[by0:by1, bx0:bx1] = cv2.add(tgt, src_b)
        except Exception:
            pass

    return char_offset_y


def put_text_vertical(
    font_size: int,
    text: str,
    height: int,
    fg: Tuple[int, int, int],
    bg: Optional[Tuple[int, int, int]],
    alignment: str = "center",
    line_spacing: int = 0,
) -> np.ndarray:
    text = compact_special_symbols(text)
    if not text:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_x = int(font_size * (line_spacing or 0.2))

    line_text_list, line_height_list = calc_vertical(font_size, text, height)

    num_char_y = max(height // font_size, 1)
    num_char_x = len(text) // num_char_y + 1
    canvas_x = font_size * num_char_x + spacing_x * (num_char_x - 1) + (font_size + bg_size) * 2
    canvas_y = font_size * num_char_y + (font_size + bg_size) * 2

    canvas_text = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    pen_orig = [canvas_x - (font_size + bg_size), font_size + bg_size]

    for line_text, line_height in zip(line_text_list, line_height_list):
        pen_line = pen_orig.copy()
        if alignment == "center":
            pen_line[1] += (max(line_height_list) - line_height) // 2
        elif alignment == "right":
            pen_line[1] += max(line_height_list) - line_height

        for c in line_text:
            offset_y = put_char_vertical(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[1] += offset_y
        pen_orig[0] -= spacing_x + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)
    x, y, w, h = cv2.boundingRect(canvas_border)
    if w == 0 or h == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    return line_box[y:y + h, x:x + w]


# ─────────────────────────────────────────────────────────────────────────────
# Auto renderer — picks horizontal or vertical based on bubble aspect ratio
# ─────────────────────────────────────────────────────────────────────────────

def auto_render(
    text: str,
    width: int,
    height: int,
    font_size: int,
    fg: Tuple[int, int, int] = (0, 0, 0),
    bg: Optional[Tuple[int, int, int]] = (255, 255, 255),
    direction: str = "auto",
    alignment: str = "center",
) -> np.ndarray:
    """
    Render text and return an RGBA numpy array.

    Args:
        text:       Text to render
        width:      Target bubble width (pixels)
        height:     Target bubble height (pixels)
        font_size:  Font size in pixels
        fg:         Foreground (text) color RGB
        bg:         Background (stroke/outline) color RGB, None for no outline
        direction:  'h' force horizontal, 'v' force vertical, 'auto' decide by aspect ratio
        alignment:  'left', 'center', 'right'

    Returns:
        RGBA numpy array ready for compositing
    """
    if direction == "auto":
        direction = "v" if height > width * 1.5 else "h"

    if direction == "v":
        return put_text_vertical(font_size, text, height, fg, bg, alignment)
    else:
        return put_text_horizontal(font_size, text, width, height, fg, bg, alignment)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Test horizontal
    img_h = auto_render(
        "Hello this is a test of the horizontal renderer",
        width=300, height=80, font_size=20,
        fg=(0, 0, 0), bg=(255, 255, 255),
        direction="h",
    )
    cv2.imwrite("renderer_test_horizontal.png", cv2.cvtColor(img_h, cv2.COLOR_RGBA2BGRA))
    print(f"Horizontal: {img_h.shape}")

    # Test vertical
    img_v = auto_render(
        "竖排文字测试",
        width=40, height=200, font_size=24,
        fg=(0, 0, 0), bg=(255, 255, 255),
        direction="v",
    )
    cv2.imwrite("renderer_test_vertical.png", cv2.cvtColor(img_v, cv2.COLOR_RGBA2BGRA))
    print(f"Vertical: {img_v.shape}")

    # Test Chinese horizontal
    img_zh = auto_render(
        "这是中文水平排版测试文字",
        width=300, height=60, font_size=22,
        fg=(0, 0, 0), bg=(255, 255, 255),
        direction="h",
    )
    cv2.imwrite("renderer_test_chinese.png", cv2.cvtColor(img_zh, cv2.COLOR_RGBA2BGRA))
    print(f"Chinese: {img_zh.shape}")
    print("All tests saved.")


if __name__ == "__main__":
    main()
