"""
bubble_detector_ct.py — Speech bubble detector using comic-translate's RT-DETR-v2.

Detects both bubble shapes and text regions in comic/manga images.
Model: ogkalu/comic-text-and-bubble-detector (172MB, auto-downloads from HuggingFace)

Classes:
    0 = bubble       (speech bubble shape, no text box)
    1 = text_bubble  (text region inside a bubble)
    2 = text_free    (text region outside a bubble, SFX etc)

Usage:
    detector = BubbleDetectorCT(device='cuda')
    detector.load()
    results = detector.detect(frame)  # frame is RGB numpy array
    # results -> list of (bubble_rect, text_rect, is_in_bubble)
    #   bubble_rect: QRect of the bubble shape (may be None for text_free)
    #   text_rect:   QRect of the text region
    #   is_in_bubble: bool
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QRect
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor


# ─────────────────────────────────────────────────────────────────────────────
# BubbleDetectorCT
# ─────────────────────────────────────────────────────────────────────────────

class BubbleDetectorCT:
    """
    Wraps comic-translate's RT-DETR-v2 bubble detector.

    Args:
        device:               'cuda' or 'cpu'
        confidence_threshold: Minimum detection confidence (0-1)
    """

    REPO = 'ogkalu/comic-text-and-bubble-detector'
    LABEL_BUBBLE       = 0
    LABEL_TEXT_BUBBLE  = 1
    LABEL_TEXT_FREE    = 2

    def __init__(self, device: str = 'cuda', confidence_threshold: float = 0.3):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None

    # ── public API ────────────────────────────────────────────────────────────

    def load(self):
        """Load model from HuggingFace (downloads on first run, cached after)."""
        print(f"[BubbleDetectorCT] Loading model from {self.REPO}...")
        self._processor = RTDetrImageProcessor.from_pretrained(
            self.REPO,
            size={'width': 640, 'height': 640},
        )
        self._model = RTDetrV2ForObjectDetection.from_pretrained(self.REPO)
        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"[BubbleDetectorCT] Loaded on {self.device}")

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect bubbles and text regions in an RGB frame.

        Args:
            frame: RGB numpy array (H, W, 3)

        Returns:
            List of (bubble_rect, text_rect, text) where:
                bubble_rect: QRect of bubble shape, or None if text_free
                text_rect:   QRect of text region
                text:        empty string (filled later by OCR)

            bubble_rect is the BUBBLE shape — use this for overlay rendering.
            text_rect is the TEXT region — use this for OCR cropping.
        """
        if self._model is None:
            raise RuntimeError("BubbleDetectorCT not loaded. Call load() first.")

        pil_image = Image.fromarray(frame)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]

        # Separate bubble shapes from text regions
        bubble_boxes = []   # class 0: pure bubble shapes
        text_boxes   = []   # class 1+2: text regions

        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
            x1, y1, x2, y2 = map(int, box.tolist())
            lbl = label.item()
            if lbl == self.LABEL_BUBBLE:
                bubble_boxes.append(QRect(x1, y1, x2 - x1, y2 - y1))
            elif lbl in (self.LABEL_TEXT_BUBBLE, self.LABEL_TEXT_FREE):
                text_boxes.append((QRect(x1, y1, x2 - x1, y2 - y1), lbl))

        # Match each text region to its containing bubble
        results_out = []
        for text_rect, lbl in text_boxes:
            matched_bubble = None
            if lbl == self.LABEL_TEXT_BUBBLE:
                for bubble_rect in bubble_boxes:
                    if self._rect_contains(bubble_rect, text_rect) or \
                       self._rects_overlap(bubble_rect, text_rect):
                        matched_bubble = bubble_rect
                        break
            results_out.append((matched_bubble, text_rect, ""))

        return results_out

    def unload(self):
        """Release model from memory."""
        del self._model
        del self._processor
        self._model = None
        self._processor = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rect_contains(outer: QRect, inner: QRect) -> bool:
        return outer.contains(inner)

    @staticmethod
    def _rects_overlap(a: QRect, b: QRect) -> bool:
        return a.intersects(b)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python bubble_detector_ct.py <comic_image>")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    frame_bgr = cv2.imread(img_path)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detector = BubbleDetectorCT(device='cuda', confidence_threshold=0.3)
    print("Loading model...")
    detector.load()

    print("Detecting...")
    results = detector.detect(frame)

    print(f"\nFound {len(results)} text regions:")
    for i, (bubble_rect, text_rect, _) in enumerate(results):
        in_bubble = bubble_rect is not None
        print(f"  [{i}] text=({text_rect.x()},{text_rect.y()},{text_rect.width()}x{text_rect.height()}) "
              f"in_bubble={in_bubble}")
        if bubble_rect:
            print(f"       bubble=({bubble_rect.x()},{bubble_rect.y()},{bubble_rect.width()}x{bubble_rect.height()})")

    # Draw results
    vis = frame_bgr.copy()
    for bubble_rect, text_rect, _ in results:
        # Draw text rect in cyan
        cv2.rectangle(vis,
                      (text_rect.x(), text_rect.y()),
                      (text_rect.x() + text_rect.width(), text_rect.y() + text_rect.height()),
                      (255, 200, 0), 2)
        # Draw bubble rect in green
        if bubble_rect:
            cv2.rectangle(vis,
                          (bubble_rect.x(), bubble_rect.y()),
                          (bubble_rect.x() + bubble_rect.width(), bubble_rect.y() + bubble_rect.height()),
                          (0, 255, 0), 1)

    out_path = "bubble_detector_ct_output.png"
    cv2.imwrite(out_path, vis)
    print(f"\nSaved to {out_path}")

    detector.unload()


if __name__ == "__main__":
    main()
