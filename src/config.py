"""
Central configuration for the Quest1 dialogue-localization pipeline.

Combines the setup from model-1.ipynb (Qwen ASR transcription) and
model-2.ipynb (semantic + fuzzy dialogue matching) into one place.

Kaggle-specific paths (/kaggle/input, /kaggle/working) and the
yt-dlp download / kaggle-dataset-upload steps from the original
notebooks are dropped here since the raw video/audio already live
locally under data/.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
INPUT_VIDEO_PATH = DATA_DIR / "video.mp4"
INPUT_AUDIO_PATH = DATA_DIR / "audio.wav"

OUTPUT_DIR = BASE_DIR / "outputs"
CHUNKS_DIR = OUTPUT_DIR / "audio_chunks"
RESULTS_DIR = OUTPUT_DIR / "qwen_results"

TRANSCRIPT_PATH = RESULTS_DIR / "qwen_transcript.json"
CHUNK_RESULTS_PATH = RESULTS_DIR / "chunk_results.json"
FRAME_PATH = OUTPUT_DIR / "dialogue_frame.jpg"
FINAL_RESULT_PATH = OUTPUT_DIR / "final_result.json"

for _dir in (OUTPUT_DIR, CHUNKS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------------------------
# Source video acquisition (model-1.ipynb, section 3.1)
# Only needed to (re)create data/video.mp4 + data/audio.wav from scratch -
# see download_data.py. The pipeline itself just reads from DATA_DIR.
# ---------------------------------------------------------------------------
SOURCE_VIDEO_URL = "https://ok.ru/video/248244667877"
MAX_VIDEO_HEIGHT = 720

# ---------------------------------------------------------------------------
# Audio chunking (model-1.ipynb, section 4.4)
# ---------------------------------------------------------------------------
CHUNK_SECONDS = 60
OVERLAP_SECONDS = 5

# ---------------------------------------------------------------------------
# Qwen ASR + forced aligner (model-1.ipynb, section 5.1)
# ---------------------------------------------------------------------------
ASR_MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_MODEL_NAME = "Qwen/Qwen3-ForcedAligner-0.6B"
MAX_NEW_TOKENS = 4096

# ---------------------------------------------------------------------------
# Semantic + fuzzy dialogue matching (model-2.ipynb, sections 6-13)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

WINDOW_SIZE = 30          # words per coarse search window
WINDOW_STRIDE = 10        # word stride between windows
TOP_K = 10                # candidates kept at the coarse-search stage

SEMANTIC_WEIGHT = 0.7
FUZZY_WEIGHT = 0.3

FINE_MATCH_MAX_EXTRA_WORDS = 10   # search slack around the coarse window