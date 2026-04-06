# Comic Overlay Translator

A real-time comic translation overlay for Windows. Captures a screen region, detects speech bubbles, reads text via OCR, translates it, and renders the translation directly over the original. Will automatically detect when scrolled or page changed.

## How it works

1. Draw a region over the comic panel using the Select button
2. Press Start — the pipeline runs automatically
3. When a page change is detected, the overlay clears and re-translates the new page
4. Press Stop to end the session

## Requirements

Python 3.10+ with the following packages:

```
PySide6
numpy
opencv-python
torch
Pillow
transformers
onnxruntime-gpu
imagehash
mss
freetype-py
```

Install PyTorch separately from https://pytorch.org matching your CUDA version.

If you do not have a GPU, use `onnxruntime` instead of `onnxruntime-gpu`.

If using a local LoRA translation model, also install:

```
unsloth
```

## Files required

```
main.py
toolbar.py
overlay.py
pipeline_worker.py
settings.py
renderer-mit.py
bubble_detector_ct.py
ocr_onnx.py
fonts/
  msyh.ttc
  Arial-Unicode-Regular.ttf
  msgothic.ttc
  NotoSansMonoCJK-VF.ttf.ttc
```

## Models

Models are downloaded automatically on first run:

- **Bubble detector** — `ogkalu/comic-text-and-bubble-detector` (172MB, cached via HuggingFace)
- **OCR** — PP-OCRv5 ONNX models (downloaded to `models/ocr/ppocr-v5-onnx/`)
- **Translator** — `Helsinki-NLP/opus-mt-*` via HuggingFace Transformers (default)

## Translation model

By default, uses Helsinki-NLP opus-mt models for Chinese, Japanese, or Korean to English.

To use a local LoRA fine-tuned model, open Settings and point the model path to a folder containing `adapter_config.json`. The model is loaded via Unsloth in 4-bit quantization.

## Settings

Open the settings panel via the `...` menu on the toolbar:

- **Language** — source language (Chinese, Japanese, Korean)
- **Translation Model** — path to a local LoRA model folder, or leave blank for default
- **Capture Region** — visually select the screen region to translate
- **Overlay Style** — Transparent, White, or Black background behind translated text
- **Detection Timing** — tune idle interval (motion change), burst interval, and stable duration (motion stop)

## Running

```
python main.py
```
