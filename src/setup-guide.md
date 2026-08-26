# Setup Guide

Step-by-step instructions to get from a fresh clone to a finished
dialogue-localization result.

## 1. Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) on your PATH (used for audio extraction)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download a build from ffmpeg.org and add its `bin/` folder to PATH
- A CUDA-capable GPU is strongly recommended for the ASR step (Qwen3-ASR-1.7B).
  It will run on CPU, just much slower.

## 2. Create and activate a virtual environment

From the project root (the folder containing `src/`, `data/`, etc.):

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash / MINGW64 - what your prompt shows)
source .venv/Scripts/activate
```

You should see `(.venv)` appear at the start of your shell prompt.

## 3. Install dependencies

For running the full pipeline:

```bash
pip install -r src/requirements.txt
```

This pulls in torch, torchaudio, opencv, the Qwen ASR package, transformers,
sentence-transformers, and rapidfuzz. It's a large install (multiple GB with
CUDA-enabled torch) and will take a while.

If you only want to run the unit tests (no GPU, no model downloads, runs in
under a second), you don't need the full install — see step 6.

## 4. Confirm your data is in place

The pipeline expects:

```
data/video.mp4
data/audio.wav
```

These should already be in the repo. If they're ever missing (or you want
to refresh them from the original source), regenerate them with:

```bash
python -m src.download_data
```

This requires [yt-dlp](https://github.com/yt-dlp/yt-dlp) to be installed
(`pip install yt-dlp`) and ffmpeg on your PATH. Pass `--force` to overwrite
existing files, or `--url` / `--max-height` to pull from a different source.

## 5. Run the pipeline

From the project root:

```bash
python -m src.pipeline --dialogue "My mind rebels at stagnation"
```

What this does:
1. Extracts audio from `data/video.mp4` if `data/audio.wav` isn't already there.
2. Splits the audio into overlapping chunks.
3. Transcribes every chunk with Qwen3-ASR (this is the slow, GPU-heavy step).
4. Builds one global, timestamped transcript.
5. Searches the transcript for your `--dialogue` line (semantic + fuzzy
   matching, so paraphrasing is fine).
6. Extracts the matching video frame.
7. Saves everything to `outputs/final_result.json` and prints a summary.

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--dialogue` | *(required)* | The line of dialogue to locate. |
| `--video` | `data/video.mp4` | Source video path. |
| `--audio` | `data/audio.wav` | Source audio path (extracted from `--video` if missing). |
| `--skip-transcription` | off | Reuse a transcript already saved at `outputs/qwen_results/qwen_transcript.json` instead of re-running ASR. Use this to try multiple dialogue queries quickly after the first full run. |
| `--output` | `outputs/final_result.json` | Where to save the result JSON. |

Example of a fast follow-up query against an existing transcript:

```bash
python -m src.pipeline --dialogue "a different line" --skip-transcription
```

### Where to find your output

- `outputs/final_result.json` — matched dialogue, timestamp, scores, frame path.
- `outputs/dialogue_frame.jpg` — the extracted video frame.
- `outputs/qwen_results/qwen_transcript.json` — the full transcript, reusable via `--skip-transcription`.

## 6. Run the tests

The test suite mocks out torch/torchaudio/opencv/qwen_asr/sentence-transformers,
so it runs in well under a second with no GPU and no model downloads.

Install just the test dependencies (much lighter than step 3):

```bash
pip install -r tests/requirements-dev.txt
```

Add `pytest.ini` and the `tests/` folder to the project root if they aren't
there already, then run:

```bash
pytest
```

or, from anywhere:

```bash
python -m pytest
```

You should see all tests pass in well under a second.

## Troubleshooting

- **`ffmpeg: command not found`** — install ffmpeg and make sure it's on PATH (step 1).
- **CUDA out of memory** — the pipeline unloads the ASR model before matching,
  but if you're on a small GPU, try closing other GPU processes or running on CPU.
- **`yt-dlp: command not found`** when running `download_data.py`  — `pip install yt-dlp`.
- **Tests fail on import** — make sure you're running `pytest` from the
  project root (or that `pytest.ini` is at the project root) so `tests/conftest.py`
  can find `src/` on the path.