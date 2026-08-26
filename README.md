# Quest1: Video Dialogue Finder

A comprehensive deep learning and NLP solution to locate exact spoken dialogue lines within a video, returning the precise timestamp (in milliseconds), frame number, extracted dialogue text, confidence scores, and the corresponding video frame image.

---

## 📌 Project & Task Overview

Given a video file (or video URL) and an input text query representing a spoken dialogue line (e.g. *"My mind rebels at stagnation"*), the system:
1. Processes the video's audio track and transcribes it with **word-level millisecond timestamps**.
2. Handles query variations, typos, and paraphrasing using a **Hybrid Semantic + Fuzzy Search**.
3. Locates the exact frame where the target dialogue is first spoken.
4. Extracts and saves the video frame image, generating a structured JSON output and interactive visual presentation.

---

## 🏗️ Repository Architecture

```
Quest1/
├── pytest.ini                  # Pytest runner configuration
├── README.md                   # Project overview & local setup guide
│
├── src/                        # Core Python package codebase
│   ├── config.py               # Path definitions, device setup, hyperparameter defaults
│   ├── download_data.py        # Video/audio downloader utility (yt-dlp)
│   ├── transcription.py       # Qwen3-ASR model loader & chunk transcription pipeline
│   ├── matching.py            # SentenceTransformer semantic vector & RapidFuzz hybrid search
│   ├── utils.py                # FFmpeg audio processing & OpenCV frame extraction utilities
│   ├── pipeline.py             # End-to-end CLI pipeline orchestrator
│   └── requirements.txt        # Core project dependencies
│
├── documentation/              # Design notes & architectural documentation
│   └── approach.md             # Initial design thoughts, edge cases & workflow breakdown
│
├── notebooks/                  # Development & experimentation notebooks
│   ├── model-1.ipynb           # Experiment 1: ASR & transcript generation
│   └── model-2.ipynb           # Experiment 2: Semantic + fuzzy dialogue matching
│
├── tests/                      # Automated unit test suite
│   ├── test_config.py          # Configuration tests
│   ├── test_download_data.py   # Data downloader tests
│   ├── test_transcription.py  # Transcription logic tests
│   ├── test_matching.py       # Matching & search algorithm tests
│   ├── test_utils.py          # Audio/video processing utility tests
│   └── test_pipeline.py       # Pipeline integration tests
│
├── data/                       # Local video and audio assets (git-ignored)
│   ├── video.mp4               # Target input video file
│   └── audio.wav               # Extracted 16kHz mono audio file
│
└── outputs/                    # Processed output artifacts (git-ignored)
    ├── final_result.json       # JSON output containing timestamps, scores, and metadata
    ├── dialogue_frame.jpg      # Extracted video frame image
    └── qwen_results/           # Saved word-level transcript JSON
```

---

## ⚡ How the Pipeline Works

1. **Audio Extraction & Chunking**:
   - **FFmpeg** extracts a 16kHz mono WAV file from the source video.
   - Audio is divided into 30-second overlapping chunks (5-second overlap) to avoid GPU memory limits and prevent missing boundary words.
2. **ASR & Forced Alignment**:
   - **Qwen3-ASR-1.7B** transcribes each chunk and aligns every word to precise start and end millisecond timestamps.
   - Results are flattened into a global, video-relative transcript (`outputs/qwen_results/qwen_transcript.json`).
3. **Hybrid Dialogue Matching**:
   - **Semantic Search**: Encodes transcript sliding windows and user query into 384-dimensional vectors via `sentence-transformers/all-MiniLM-L6-v2` (handles paraphrasing).
   - **Fuzzy Search**: Computes string Levenshtein distance via `rapidfuzz` (handles ASR transcription errors and typos).
   - **Combined Score**: Merges scores: `0.7 * Semantic + 0.3 * Fuzzy`.
4. **Fine-Grained Word-Span Localization**:
   - Optimizes start/end word boundaries to pinpoint exact millisecond range (`start_ms` & `end_ms`).
5. **Frame Extraction**:
   - **OpenCV** navigates to `start_ms`, captures the exact video frame, calculates the frame index, and saves it to `outputs/dialogue_frame.jpg`.

---

## 🚀 Local Setup Guide

Follow these steps to set up and run the project locally on Windows, macOS, or Linux.

### 1. Prerequisites

- **Python 3.10+**
- **FFmpeg** (Required for audio extraction):
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu/Debian**: `sudo apt install ffmpeg`
  - **Windows**: Download build from [ffmpeg.org](https://ffmpeg.org/download.html) and add its `bin/` folder to your system PATH.
- **CUDA GPU** (Recommended for fast ASR inference; CPU is supported as fallback).

---

### 2. Virtual Environment Setup

From the project root folder:

```bash
# Create a virtual environment
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate
```

---

### 3. Install Dependencies

```bash
pip install -r src/requirements.txt
```

---

## 💻 Running the Application


### Using Command Line Interface (CLI)

Run the full pipeline directly from your terminal:

```bash
# Run full pipeline with a dialogue query
python -m src.pipeline --dialogue "My mind rebels at stagnation"
```

**Reusing Saved Transcripts (Fast Mode)**:
To query different dialogue lines without re-running the heavy ASR step:

```bash
python -m src.pipeline --dialogue "a different dialogue line" --skip-transcription
```

**Available CLI Flags**:

| Flag | Default | Description |
|---|---|---|
| `--dialogue` | *(Required)* | Target line of dialogue to locate. |
| `--video` | `data/video.mp4` | Path to source video file. |
| `--audio` | `data/audio.wav` | Path to source audio file (extracted if missing). |
| `--skip-transcription` | `False` | Reuses existing transcript JSON to save time. |
| `--output` | `outputs/final_result.json` | Path to save output JSON. |

---

## 🧪 Running Unit Tests

Run the automated test suite with Pytest:

```bash
pytest
```

The test suite mocks heavy models (torch, Qwen-ASR, OpenCV) so it runs in less than a second without requiring GPU hardware or model downloads.

---

## 📄 Output Formats

Upon completion, results are written to `outputs/final_result.json`:

```json
{
  "input_dialogue": "My mind rebels at stagnation",
  "matched_dialogue": "my mind rebels at stagnation",
  "timestamp": "00:01:24.500",
  "start_ms": 84500,
  "end_ms": 87200,
  "semantic_score": 0.965,
  "fuzzy_score": 0.981,
  "combined_score": 0.970,
  "frame_number": 2028,
  "frame_timestamp_ms": 84500,
  "frame_path": "outputs/dialogue_frame.jpg"
}
```

---

## 📚 Further Documentation

For design notes, initial observations, edge-case analysis, and future extension ideas, check out [`documentation/approach.md`](file:///d:/Pavithra/GitHub/Quest1_Assignment/Quest1/documentation/approach.md).
