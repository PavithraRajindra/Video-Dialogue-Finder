"""
ASR stage of the pipeline (model-1.ipynb, section 5): load Qwen3-ASR
with its forced aligner, transcribe audio chunks with word/segment-level
timestamps, and flatten everything into one global, video-relative
transcript that the matching stage (model-2.ipynb) consumes.
"""

import gc
import json

import torch
from qwen_asr import Qwen3ASRModel

import config


def load_qwen_model():
    """Load the Qwen3-ASR model + forced aligner onto the configured device."""
    return Qwen3ASRModel.from_pretrained(
        config.ASR_MODEL_NAME,
        dtype=config.TORCH_DTYPE,
        device_map=config.DEVICE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        forced_aligner=config.FORCED_ALIGNER_MODEL_NAME,
        forced_aligner_kwargs=dict(
            dtype=config.TORCH_DTYPE,
            device_map=config.DEVICE,
        ),
    )


def unload_qwen_model(qwen_model):
    """Free GPU memory held by the ASR model once transcription is done."""
    del qwen_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def transcribe_chunks(qwen_model, chunks, results_path=None):
    """
    Transcribe every audio chunk with word-level timestamps.

    Saves progress to `results_path` after each chunk (so a crash mid-run
    doesn't lose earlier work), mirroring model-1.ipynb section 5.2.

    Returns a list of dicts: {chunk_index, chunk_start_seconds,
    chunk_end_seconds, text, time_stamps: [{text, start_time, end_time}]}
    """
    results_path = results_path or config.CHUNK_RESULTS_PATH
    all_results = []

    for i, chunk in enumerate(chunks):
        print(
            f"[{i + 1}/{len(chunks)}] Processing chunk {chunk['index']} "
            f"({chunk['start_seconds']:.2f}s - {chunk['end_seconds']:.2f}s)"
        )

        result = qwen_model.transcribe(
            audio=str(chunk["path"]),
            language=None,
            return_time_stamps=True,
        )[0]

        if not result.text or not result.text.strip():
            print("  -> No speech detected.")
            chunk_result = {
                "chunk_index": chunk["index"],
                "chunk_start_seconds": chunk["start_seconds"],
                "chunk_end_seconds": chunk["end_seconds"],
                "text": "",
                "time_stamps": [],
            }
        else:
            print(f"  -> Transcription: {result.text[:100]}...")
            chunk_result = {
                "chunk_index": chunk["index"],
                "chunk_start_seconds": chunk["start_seconds"],
                "chunk_end_seconds": chunk["end_seconds"],
                "text": result.text,
                # Flatten to plain dicts immediately so every downstream
                # consumer (including a reloaded JSON file) sees the same
                # shape - avoids the object-vs-dict mismatch the original
                # notebook hit between sections 5.2 and 5.3.
                "time_stamps": [
                    {
                        "text": ts.text,
                        "start_time": ts.start_time,
                        "end_time": ts.end_time,
                    }
                    for ts in result.time_stamps
                ],
            }

        all_results.append(chunk_result)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  -> Saved.\n")

    return all_results


def build_global_transcript(all_qwen_results):
    """
    Flatten per-chunk, chunk-relative timestamps into one global,
    video-relative transcript (model-1.ipynb, section 5.3).

    Returns a list of {text, start_ms, end_ms, chunk_index} dicts,
    sorted by start time.
    """
    qwen_transcript = []

    for chunk_result in all_qwen_results:
        if not chunk_result["text"]:
            continue

        chunk_start = chunk_result["chunk_start_seconds"]
        chunk_index = chunk_result["chunk_index"]

        for ts in chunk_result["time_stamps"]:
            global_start = chunk_start + ts["start_time"]
            global_end = chunk_start + ts["end_time"]

            qwen_transcript.append({
                "text": ts["text"],
                "start_ms": round(global_start * 1000),
                "end_ms": round(global_end * 1000),
                "chunk_index": chunk_index,
            })

    # Overlapping chunks can produce out-of-order / duplicate entries;
    # sort by start time so downstream windowing sees a clean sequence.
    qwen_transcript.sort(key=lambda item: item["start_ms"])
    return qwen_transcript


def save_transcript(qwen_transcript, path=None):
    path = path or config.TRANSCRIPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qwen_transcript, f, ensure_ascii=False, indent=2)
    return path


def load_transcript(path=None):
    path = path or config.TRANSCRIPT_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)