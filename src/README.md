# src/

Production code for the Quest1 video dialogue-localization pipeline —
combines the Qwen3-ASR transcription approach from `model-1.ipynb` with the
semantic + fuzzy dialogue-matching approach from `model-2.ipynb` into one
runnable pipeline.

**What it does:** given a video and a line of dialogue (exact or
paraphrased), it transcribes the video's audio, finds where that line is
spoken, and extracts the matching video frame.

## Files

| File | Purpose |
|---|---|
| `config.py` | All paths, model names, and tunable constants (chunk size, search weights, etc.) in one place. |
| `utils.py` | Model-agnostic helpers: downloading/loading video, extracting/splitting audio, grabbing frames, text normalization, timestamp formatting. |
| `transcription.py` | Loads Qwen3-ASR + forced aligner, transcribes audio chunks with timestamps, and flattens them into one global transcript (model-1.ipynb). |
| `matching.py` | `DialogueMatcher`: locates a line of dialogue in the transcript using combined semantic (sentence-transformers) + fuzzy (rapidfuzz) search, then a fine-grained word-span search (model-2.ipynb). This is what supersedes model-1's exact-substring matching — it also handles paraphrased queries. |
| `pipeline.py` | CLI entry point. Wires transcription → matching → frame extraction → JSON result end to end. |
| `download_data.py` | Standalone script to (re)create `data/video.mp4` / `data/audio.wav` from the original source URL, if you ever need to refresh them. Not part of the normal run — the pipeline reads directly from `data/`. |
| `requirements.txt` | Runtime dependencies for the full pipeline (ASR model, embedding model, video/audio libs). |

## See also

- `SETUP.md` — step-by-step install and run instructions.
- `../tests/` — the unit test suite (mocks out heavy ML deps, runs fast/offline).