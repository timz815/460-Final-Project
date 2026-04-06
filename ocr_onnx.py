"""
ocr_onnx.py — CT-style PPOCRv5 ONNX pipeline following comic-translate exactly.

Pipeline (mirrors CT's PPOCRv5Engine.process_image):
    1. Det on full frame  → finds all text line quads on the page
    2. Crop each quad     → individual line strips
    3. Rec on all strips  → batched by aspect ratio (8 per batch)
    4. Match lines to bubbles → via bounding box containment

This is how CT achieves near-100% accuracy — det runs once on the full page,
not per-bubble, so it finds all text lines regardless of bubble size.
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Tuple, Optional
from PySide6.QtCore import QRect

_HERE = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(_HERE, "models", "ocr", "ppocr-v5-onnx")
DET_FILE   = os.path.join(MODELS_DIR, "ch_PP-OCRv5_mobile_det.onnx")
# REC_FILE   = os.path.join(MODELS_DIR, "ch_PP-OCRv5_rec_mobile_infer.onnx")
# DICT_FILE  = os.path.join(MODELS_DIR, "ppocrv5_dict.txt")
REC_FILES = {
    "Chinese":  (os.path.join(MODELS_DIR, "ch_PP-OCRv5_rec_mobile_infer.onnx"),
                 os.path.join(MODELS_DIR, "ppocrv5_dict.txt")),
    "Japanese": (os.path.join(MODELS_DIR, "japan_PP-OCRv5_rec_mobile_infer.onnx"),
                 os.path.join(MODELS_DIR, "japan_ppocrv5_dict.txt")),
    "Korean":   (os.path.join(MODELS_DIR, "korean_PP-OCRv5_rec_mobile_infer.onnx"),
                 os.path.join(MODELS_DIR, "korean_ppocrv5_dict.txt")),
}

# Recognition model input shape (C, H, W) — H=48 is fixed, W is dynamic
REC_SHAPE = (3, 48, 320)


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

DET_URL  = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"
REC_URL  = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx"
DICT_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt"


def _download(url: str, path: str):
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[OCRonnx] Downloading {os.path.basename(path)}...")
    urllib.request.urlretrieve(url, path)
    print(f"[OCRonnx] Saved to {path}")


def ensure_models():
    if not os.path.exists(DET_FILE):
        _download(DET_URL, DET_FILE)
    rec_file, dict_file = REC_FILES["Chinese"]
    if not os.path.exists(rec_file):
        _download(REC_URL, rec_file)
    if not os.path.exists(dict_file):
        _download(DICT_URL, dict_file)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — matches CT's preprocessing.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def _det_preprocess(img: np.ndarray, limit_side_len: int = 960,
                    limit_type: str = 'min') -> np.ndarray:
    """Resize so min side >= limit, snap to multiple of 32, normalize."""
    h, w = img.shape[:2]
    if limit_type == 'max':
        ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
    else:
        ratio = float(limit_side_len) / min(h, w) if min(h, w) < limit_side_len else 1.0
    nh = max(int(round(h * ratio / 32) * 32), 32)
    nw = max(int(round(w * ratio / 32) * 32), 32)
    resized = cv2.resize(img, (nw, nh))
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.expand_dims(x.transpose(2, 0, 1), 0).astype(np.float32)


def _rec_resize_norm(img: np.ndarray, img_shape: tuple,
                     max_wh_ratio: float) -> np.ndarray:
    """Resize crop to fixed height, pad width to max_ratio. Matches CT exactly."""
    c, H, W = img_shape
    h, w = img.shape[:2]
    ratio = w / float(max(h, 1))
    target_w = int(H * max_wh_ratio)
    resized_w = min(target_w, max(1, int(np.ceil(H * ratio))))
    resized = cv2.resize(img, (resized_w, H))
    x = resized.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = (x - 0.5) / 0.5
    out = np.zeros((c, H, target_w), dtype=np.float32)
    out[:, :, :resized_w] = x
    return out


def _crop_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-crop a quadrilateral. Auto-rotate tall crops."""
    pts = quad.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w = max(w, 1); h = max(h, 1)
    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h))
    if h / float(w) >= 1.5:
        crop = np.rot90(crop)
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing — matches CT's DBPostProcessor exactly
# ─────────────────────────────────────────────────────────────────────────────

def _db_postprocess(pred: np.ndarray, orig_h: int, orig_w: int,
                    thresh: float = 0.3, box_thresh: float = 0.5,
                    unclip_ratio: float = 2.0) -> np.ndarray:
    """Convert DB probability map to sorted quadrilateral boxes."""
    import pyclipper
    from shapely.geometry import Polygon

    prob = pred[0, 0]
    mask = (prob > thresh).astype(np.uint8) * 255
    ph, pw = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contours:
        if len(c) < 4:
            continue
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect)
        pts_s = sorted(pts.tolist(), key=lambda p: p[0])
        left  = sorted(pts_s[:2], key=lambda p: p[1])
        right = sorted(pts_s[2:], key=lambda p: p[1])
        box = np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

        # Score check
        xs = box[:, 0].clip(0, pw - 1)
        ys = box[:, 1].clip(0, ph - 1)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        region = prob[ymin:ymax + 1, xmin:xmax + 1]
        if region.size == 0 or float(region.mean()) < box_thresh:
            continue

        # Unclip
        try:
            poly = Polygon(box)
            dist = poly.area * unclip_ratio / (poly.length + 1e-6)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(box.astype(int).tolist(),
                        pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = pco.Execute(dist)
            if not expanded:
                continue
            exp_pts = np.array(expanded[0], dtype=np.float32)
            if len(exp_pts) < 4:
                continue
            rect2 = cv2.minAreaRect(exp_pts.reshape(-1,1,2).astype(np.int32))
            pts2 = cv2.boxPoints(rect2)
            pts2_s = sorted(pts2.tolist(), key=lambda p: p[0])
            left2  = sorted(pts2_s[:2], key=lambda p: p[1])
            right2 = sorted(pts2_s[2:], key=lambda p: p[1])
            box = np.array([left2[0], right2[0], right2[1], left2[1]], dtype=np.float32)
        except Exception:
            pass

        # Scale to original
        box[:, 0] = np.clip(box[:, 0] / pw * orig_w, 0, orig_w)
        box[:, 1] = np.clip(box[:, 1] / ph * orig_h, 0, orig_h)

        rw = int(np.linalg.norm(box[0] - box[1]))
        rh = int(np.linalg.norm(box[0] - box[3]))
        if rw < 4 or rh < 4:
            continue

        # Order clockwise: TL, TR, BR, BL
        xs2 = box[:, 0]; ys2 = box[:, 1]
        left_pts  = sorted(box[np.argsort(xs2)[:2]].tolist(), key=lambda p: p[1])
        right_pts = sorted(box[np.argsort(xs2)[2:]].tolist(), key=lambda p: p[1])
        box = np.array([left_pts[0], right_pts[0], right_pts[1], left_pts[1]],
                       dtype=np.float32)
        box[:, 0] = np.clip(box[:, 0], 0, orig_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, orig_h - 1)
        boxes.append(box.astype(np.int32))

    return np.array(boxes, dtype=np.int32) if boxes else np.zeros((0, 4, 2), dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# CTC Decoder — matches CT's CTCLabelDecoder exactly
# ─────────────────────────────────────────────────────────────────────────────

class _CTCDecoder:
    def __init__(self, dict_path: str):
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip('\n') for line in f]
        # vocab: blank=0, then chars, then space
        num_classes_hint = len(chars) + 2
        self.vocab = [''] + chars + [' ']

    def __call__(self, logits: np.ndarray,
                 prob_threshold: float = 0.0) -> Tuple[List[str], List[float]]:
        """Greedy CTC decode. logits: (N, T, C). Matches CT prob_threshold=0.0."""
        if logits.ndim == 2:
            logits = logits[None]
        num_classes = logits.shape[-1]

        # Align vocab to model classes
        vocab = self.vocab
        if num_classes != len(vocab):
            if num_classes == len(vocab) - 1:
                vocab = vocab[:-1]
            elif num_classes > len(vocab):
                vocab = vocab + [''] * (num_classes - len(vocab))

        # Already probabilities (no softmax needed — model outputs probs)
        probs = logits

        texts, confs = [], []
        blank = 0
        for n in range(probs.shape[0]):
            seq = probs[n]
            idxs = seq.argmax(axis=-1)
            last = -1
            decoded_chars, scores = [], []
            for t, i in enumerate(idxs):
                i = int(i)
                if i != blank and i != last:
                    p = float(seq[t, i])
                    if p >= prob_threshold and i < len(vocab):
                        ch = vocab[i]
                        if ch:
                            decoded_chars.append(ch)
                            scores.append(p)
                last = i
            texts.append(''.join(decoded_chars))
            confs.append(float(np.mean(scores)) if scores else 0.0)
        return texts, confs


# ─────────────────────────────────────────────────────────────────────────────
# Line-to-bubble matching — matches CT's lists_to_blk_list logic
# ─────────────────────────────────────────────────────────────────────────────

def _rect_contains(outer: QRect, inner_xyxy: Tuple[int,int,int,int],
                   min_overlap: float = 0.5) -> bool:
    """Check if inner box is mostly contained within outer QRect."""
    ox1, oy1, ox2, oy2 = outer.x(), outer.y(), outer.x()+outer.width(), outer.y()+outer.height()
    ix1, iy1, ix2, iy2 = inner_xyxy

    inter_x1 = max(ox1, ix1); inter_y1 = max(oy1, iy1)
    inter_x2 = min(ox2, ix2); inter_y2 = min(oy2, iy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    inner_area = max((ix2 - ix1) * (iy2 - iy1), 1)
    return inter_area / inner_area >= min_overlap


def _match_lines_to_bubbles(
    line_bboxes: List[Tuple[int,int,int,int]],
    line_texts:  List[str],
    bubble_rects: List[QRect],
) -> List[str]:
    """
    For each bubble, find all detected lines that fall inside it and join their texts.
    Matches CT's lists_to_blk_list logic.
    """
    bubble_texts = [[] for _ in bubble_rects]

    for (x1, y1, x2, y2), text in zip(line_bboxes, line_texts):
        if not text:
            continue
        for b_idx, rect in enumerate(bubble_rects):
            if _rect_contains(rect, (x1, y1, x2, y2)):
                bubble_texts[b_idx].append((y1, text))  # store y for sorting
                break

    # Sort lines top-to-bottom within each bubble and join
    results = []
    for lines in bubble_texts:
        lines.sort(key=lambda t: t[0])
        results.append(''.join(t for _, t in lines))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OCRonnx — CT-style full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class OCRonnx:
    """
    CT-style PPOCRv5 ONNX OCR pipeline.

    Mirrors comic-translate's PPOCRv5Engine.process_image exactly:
      1. Det on full frame → all text line quads
      2. Crop each quad → line strips
      3. Rec on all strips batched by aspect ratio
      4. Match lines to bubble rects

    Args:
        device:          'cuda' or 'cpu'
        min_confidence:  Minimum line confidence to include (0 = keep all)
    """

    def __init__(self, device: str = 'cuda', min_confidence: float = 0.0):
        self.device         = device
        self.min_confidence = min_confidence
        self._det_sess      = None
        self._rec_sess      = None
        self._decoder       = None

    # ── public API ────────────────────────────────────────────────────────────

    def load(self):
        import onnxruntime as ort
        ensure_models()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if self.device == 'cuda' else ['CPUExecutionProvider']
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._det_sess = ort.InferenceSession(DET_FILE, sess_options=so, providers=providers)
        # self._rec_sess = ort.InferenceSession(REC_FILE, sess_options=so, providers=providers)
        # self._decoder  = _CTCDecoder(DICT_FILE)
        from PySide6.QtCore import QSettings
        lang = QSettings("MangaTranslator", "Translator").value("from_lang", "Chinese")
        rec_file, dict_file = REC_FILES.get(lang, REC_FILES["Chinese"])
        if not os.path.exists(rec_file):
            print(f"[OCRonnx] Model for {lang} not found, falling back to Chinese")
            rec_file, dict_file = REC_FILES["Chinese"]
        self._rec_sess = ort.InferenceSession(rec_file, sess_options=so, providers=providers)
        self._decoder  = _CTCDecoder(dict_file)

        print(f"[OCRonnx] Loaded on {self.device}")

    def read_bubbles(self, frame: np.ndarray, bubbles: list) -> list:
        """
        Args:
            frame:   Full RGB frame (H, W, 3)
            bubbles: List of (QRect text_rect, pts) from BubbleDetectorCT

        Returns:
            List of (QRect, text) — one per bubble with detected text
        """
        if self._det_sess is None:
            raise RuntimeError("OCRonnx not loaded. Call load() first.")
        if not bubbles or frame is None or frame.size == 0:
            return []

        # Convert RGB → BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img_h, img_w = frame_bgr.shape[:2]

        # ── Step 1: Det on combined text region only ──────────────────────────
        # Find bounding box covering all text_rects to skip empty page areas
        PAD = 10
        all_rects = [rect for rect, pts in bubbles]
        rx1 = max(0,     min(r.x() for r in all_rects) - PAD)
        ry1 = max(0,     min(r.y() for r in all_rects) - PAD)
        rx2 = min(img_w, max(r.x() + r.width()  for r in all_rects) + PAD)
        ry2 = min(img_h, max(r.y() + r.height() for r in all_rects) + PAD)
        region = frame_bgr[ry1:ry2, rx1:rx2]

        inp     = _det_preprocess(region, limit_side_len=960, limit_type='min')
        det_in  = self._det_sess.get_inputs()[0].name
        det_out = self._det_sess.get_outputs()[0].name
        pred    = self._det_sess.run([det_out], {det_in: inp})[0]
        rh, rw  = region.shape[:2]
        raw_boxes = _db_postprocess(pred, rh, rw,
                                    thresh=0.3, box_thresh=0.5, unclip_ratio=2.0)

        # Shift coordinates back to full frame space
        boxes = []
        for box in raw_boxes:
            b = box.copy()
            b[:, 0] += rx1
            b[:, 1] += ry1
            boxes.append(b)
        boxes = np.array(boxes, dtype=np.int32) if boxes else np.zeros((0,4,2), dtype=np.int32)
        if len(boxes) == 0:
            return []

        # ── Step 2: Crop each detected line quad ──────────────────────────────
        crops    = [_crop_quad(frame_bgr, box.astype(np.float32)) for box in boxes]
        line_bboxes = []
        for box in boxes:
            xs = box[:, 0]; ys = box[:, 1]
            line_bboxes.append((int(xs.min()), int(ys.min()),
                                int(xs.max()), int(ys.max())))

        # ── Step 3: Rec on all crops batched by aspect ratio ──────────────────
        line_texts = self._rec_batch(crops)

        # ── Step 4: Match lines to bubble rects ───────────────────────────────
        bubble_rects = [rect for rect, pts in bubbles]
        bubble_text_list = _match_lines_to_bubbles(line_bboxes, line_texts, bubble_rects)

        results = []
        for rect, text in zip(bubble_rects, bubble_text_list):
            if text.strip():
                results.append((rect, text))
        return results

    def unload(self):
        del self._det_sess
        del self._rec_sess
        self._det_sess = None
        self._rec_sess = None

    # ── recognition ───────────────────────────────────────────────────────────

    def _rec_batch(self, crops: List[np.ndarray], batch_size: int = 8) -> List[str]:
        """Batch crops by aspect ratio — matches CT's _rec_infer exactly."""
        if not crops:
            return []

        c, H, W = REC_SHAPE
        ratios = [cr.shape[1] / float(max(cr.shape[0], 1)) for cr in crops]
        order  = np.argsort(ratios)
        texts  = [''] * len(crops)

        inp_name = self._rec_sess.get_inputs()[0].name
        out_name = self._rec_sess.get_outputs()[0].name

        for b_start in range(0, len(crops), batch_size):
            idxs     = order[b_start:b_start + batch_size]
            max_ratio = max(ratios[i] for i in idxs) if len(idxs) > 0 else W / float(H)

            batch = np.concatenate([
                _rec_resize_norm(crops[i], REC_SHAPE, max_ratio)[None]
                for i in idxs
            ], axis=0).astype(np.float32)

            logits = self._rec_sess.run([out_name], {inp_name: batch})[0]

            # Transpose if needed (N, C, T) → (N, T, C)
            if logits.ndim == 3 and logits.shape[1] > logits.shape[2]:
                logits = np.transpose(logits, (0, 2, 1))

            # CT uses prob_threshold=0.0 — no filtering
            dec_texts, dec_confs = self._decoder(logits, prob_threshold=0.0)

            for local_i, orig_i in enumerate(idxs):
                if dec_confs[local_i] >= self.min_confidence:
                    texts[orig_i] = dec_texts[local_i]

        return texts


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import time

    if len(sys.argv) < 2:
        print("Usage: python ocr_onnx.py <comic_image>")
        return

    img_path = sys.argv[1]
    frame_bgr = cv2.imread(img_path)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    from bubble_detector_ct import BubbleDetectorCT
    detector = BubbleDetectorCT(device='cuda')
    detector.load()
    results = detector.detect(frame)
    bubbles = [(text_rect, np.array([
        [text_rect.x(),                   text_rect.y()                   ],
        [text_rect.x()+text_rect.width(),  text_rect.y()                   ],
        [text_rect.x()+text_rect.width(),  text_rect.y()+text_rect.height()],
        [text_rect.x(),                   text_rect.y()+text_rect.height()],
    ], dtype=np.float32)) for _, text_rect, _ in results]
    print(f"Detected {len(bubbles)} text regions")
    detector.unload()

    ocr = OCRonnx(device='cuda', min_confidence=0.0)
    ocr.load()

    # warmup
    ocr.read_bubbles(frame, bubbles)

    t0 = time.perf_counter()
    ocr_results = ocr.read_bubbles(frame, bubbles)
    t1 = time.perf_counter()

    print(f"\nFound {len(ocr_results)} bubbles with text in {(t1-t0)*1000:.1f}ms:")
    for rect, text in ocr_results:
        print(f"  ({rect.x()},{rect.y()}) → {text!r}")

    ocr.unload()


if __name__ == "__main__":
    main()
