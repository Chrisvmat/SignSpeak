<div align="center">

# SignSpeak

**Real-time American Sign Language (ASL) Recognition & Speech — Windows Desktop App**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-00BCD4?style=flat-square)](https://ai.google.dev/edge/mediapipe)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](https://github.com/Chrisvmat/SignSpeak/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

SignSpeak captures ASL hand gestures via webcam, classifies them in real-time using a trained 1D CNN, and converts the output into spoken audio — no internet connection required.

[**⬇ Download SignSpeak.exe**](https://github.com/Chrisvmat/SignSpeak/releases) · [View Paper](#) · [Report Bug](https://github.com/Chrisvmat/SignSpeak/issues)

</div>

---

## Overview

SignSpeak is a real-time desktop application that bridges American Sign Language and spoken communication. It uses Google's MediaPipe Tasks API to extract 21 hand landmarks per frame, feeds a 63-dimensional normalized feature vector into a custom 1D CNN, and outputs classified signs through a sentence-building UI that speaks the result via text-to-speech.

The model achieves **~99.5% test accuracy** across 43 sign classes and runs entirely on-device.

---

## Supported Signs — 36 Classes

| Category | Signs |
|---|---|
| **Letters** | A – Z (26 signs) |
| **Digits** | 1 – 10 (10 signs) ||

---

## Pipeline

```
Webcam
  │
  ▼
MediaPipe Hand Landmarker (Tasks API)
  │  21 landmarks × (x, y, z) → 63 raw features
  ▼
Feature Extraction & Wrist-Relative Normalization
  │  63-dimensional float vector
  ▼
1D CNN Classifier  (signspeak_model.keras)
  │  Conv1D → BatchNorm → Dropout → Dense → Softmax (43 classes)
  ▼
Sentence Builder UI  (OpenCV overlay)
  │  Space / 4 s hold → add sign · Backspace · Clear · S → speak
  ▼
gTTS + pygame  (Text-to-Speech, Australian English)
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | 1D Convolutional Neural Network |
| Input shape | (63,) — wrist-normalised landmark vector |
| Output classes | 43 |
| Test accuracy | ~99.5% |
| Training environment | Google Colab (T4 GPU) |
| Data augmentation | Z-axis rotation · Scale jitter · Gaussian noise · X-axis flip · Combined |
| Checkpoint strategy | `ModelCheckpoint` — best validation accuracy saved |
| Model file | `app/signspeak_model.keras` |

---

## Project Structure

```
SignSpeak/
├── app/
│   ├── main.py                   # Entry point & CLI argument parser
│   ├── hand_tracking.py          # MediaPipe Tasks API — landmark extraction
│   ├── sign_recognition.py       # CNN inference + SentenceBuilder UI
│   ├── label_map.json            # Class index → sign label mapping (43 classes)
│   ├── signspeak_model.keras     # Trained model weights (v2)
│   ├── hand_landmarker.task      # MediaPipe hand landmarker bundle
│   ├── SignSpeak.spec            # PyInstaller spec — rebuild the .exe from this
│   └── ss_v3.ico                 # Application icon
├── colab/                        # Colab training utilities & scripts
├── legacy/                       # v1 reference (not used in production)
│   ├── sign_recognition_v1.py
│   └── signspeak_model_old.keras
├── thumb/                        # Screenshots & demo thumbnails
├── signspeak_train.ipynb         # Full model training notebook (Colab)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### Option A — Executable (Windows, no Python needed)

1. Go to [**Releases**](https://github.com/Chrisvmat/SignSpeak/releases)
2. Download `SignSpeak.exe` from the latest release
3. Run it — no installation required

### Option B — Run from Source

**Requirements:** Python 3.10+, a connected webcam

```bash
git clone https://github.com/Chrisvmat/SignSpeak.git
cd SignSpeak
pip install -r requirements.txt
cd app
python main.py
```

> **Camera not detected?** The app defaults to camera index `1`. If your webcam isn't found, open `main.py` and change `default=1` → `default=0` in the argument parser.

---

## Controls

| Key | Action |
|---|---|
| `Space` | Add current sign to sentence (manual mode) |
| `Backspace` | Remove last character |
| `C` | Clear the full sentence |
| `S` | Speak the sentence aloud via TTS |
| `Q` / `Esc` | Quit |

> **Auto-add:** Hold any sign steady for ~4 seconds and it will be added automatically without pressing Space.

---

## Rebuild the Executable

If you modify the source and want to regenerate `SignSpeak.exe`:

```bash
cd app
pyinstaller SignSpeak.spec
# Output: app/dist/SignSpeak.exe
```

---

## Training

The model was trained on a custom landmark dataset collected across 43 ASL classes. See [`signspeak_train.ipynb`](signspeak_train.ipynb) for the complete pipeline — data loading, augmentation, model definition, `ModelCheckpoint` saving, and evaluation.

To retrain:
1. Open `signspeak_train.ipynb` in Google Colab
2. Mount your Drive and point to your landmark CSV files
3. Run all cells — the best checkpoint saves automatically as `signspeak_model.keras`

---

## Requirements

Core runtime dependencies (see [`requirements.txt`](requirements.txt) for pinned versions):

```
opencv-python
mediapipe
tensorflow
gtts
pygame
```

---

## License

MIT License — see [LICENSE](LICENSE) for details. Free to use, modify, and distribute with attribution.

---

## Author

**Chris V. Mat** &nbsp;·&nbsp; [GitHub @Chrisvmat](https://github.com/Chrisvmat)

*Built as an academic and portfolio project — B.E. Computer Science (AI & ML)*

---

<div align="center">
<sub>Made with ❤️ and a lot of hand signs</sub>
</div>
