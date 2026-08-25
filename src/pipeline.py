"""
End-to-end pipeline combining model-1.ipynb (Qwen3-ASR transcription) and
model-2.ipynb (semantic + fuzzy dialogue matching):

  1. Make sure we have audio (extract from video if needed).
  2. Split audio into overlapping chunks.
  3. Transcribe every chunk with Qwen3-ASR + forced aligner.
  4. Flatten into one global, video-relative transcript.
  5. Locate the target line of dialogue via semantic + fuzzy search.
  6. Extract the matching video frame.
  7. Save the final result as JSON.

Usage:
    python -m src.pipeline --dialogue "My mind rebels at stagnation"

    # Re-run matching only, reusing a transcript from a previous run:
    python -m src.pipeline --dialogue "..." --skip-transcription
"""

import argparse
import json
import pprint
from pathlib import Path

import config
import transcription
import utils
from matching import DialogueMatcher


def _resolve_path(raw_path):
    """Resolve a CLI path arg against BASE_DIR if it isn't already absolute."""
    path = Path(raw_path)
    return path if path.is_absolute() else config.BASE_DIR / path


def run_transcription_stage(video_path, audio_path):
    """Steps 1-4: produce (and save) the global word-level transcript."""
    if not audio_path.exists():
        print(f"No audio found at {audio_path}, extracting from video...")
        utils.extract_audio(video_path, audio_path)

    cap, metadata = utils.load_video(video_path)
    utils.print_video_info(metadata)
    cap.release()

    print("Splitting audio into chunks...")
    chunks = utils.split_audio(
        audio_path,
        config.CHUNKS_DIR,
        chunk_seconds=config.CHUNK_SECONDS,
        overlap_seconds=config.OVERLAP_SECONDS,
    )
    print(f"Created {len(chunks)} chunks")

    print("Loading Qwen3-ASR model...")
    qwen_model = transcription.load_qwen_model()

    try:
        all_results = transcription.transcribe_chunks(qwen_model, chunks)
    finally:
        transcription.unload_qwen_model(qwen_model)

    qwen_transcript = transcription.build_global_transcript(all_results)
    transcription.save_transcript(qwen_transcript)
    print(f"Saved transcript ({len(qwen_transcript)} entries) -> {config.TRANSCRIPT_PATH}")

    return qwen_transcript


def run_matching_stage(qwen_transcript, dialogue, video_path):
    """Steps 5-6: locate the dialogue and grab the matching frame."""
    print("Loading embedding model & building transcript windows...")
    matcher = DialogueMatcher(qwen_transcript)

    print(f"Searching for: {dialogue!r}")
    match = matcher.find_best_match(dialogue)

    if match is None:
        raise RuntimeError("No match found for the given dialogue.")

    frame_info = utils.extract_frame_at_timestamp(
        video_path, match["start_ms"], config.FRAME_PATH
    )

    final_result = {
        "input_dialogue": match["input_dialogue"],
        "matched_dialogue": match["matched_dialogue"],
        "timestamp": utils.format_timestamp(match["start_ms"]),
        "start_ms": match["start_ms"],
        "end_ms": match["end_ms"],
        "semantic_score": match["semantic_score"],
        "fuzzy_score": match["fuzzy_score"],
        "combined_score": match["combined_score"],
        "frame_number": frame_info["frame_number"],
        "frame_timestamp_ms": frame_info["actual_timestamp_ms"],
        "frame_path": frame_info["frame_path"],
    }

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Video dialogue localization pipeline")
    parser.add_argument(
        "--dialogue", required=True,
        help="Line of dialogue to locate in the video (paraphrasing is fine).",
    )
    parser.add_argument(
        "--video", default=str(config.INPUT_VIDEO_PATH),
        help="Path to the source video (default: data/video.mp4).",
    )
    parser.add_argument(
        "--audio", default=str(config.INPUT_AUDIO_PATH),
        help="Path to the source audio (default: data/audio.wav; "
             "extracted from --video if missing).",
    )
    parser.add_argument(
        "--skip-transcription", action="store_true",
        help="Reuse an existing transcript JSON instead of re-running ASR.",
    )
    parser.add_argument(
        "--output", default=str(config.FINAL_RESULT_PATH),
        help="Where to save the final result JSON.",
    )
    args = parser.parse_args()

    video_path = _resolve_path(args.video)
    audio_path = _resolve_path(args.audio)

    if args.skip_transcription:
        print(f"Loading existing transcript from {config.TRANSCRIPT_PATH} ...")
        qwen_transcript = transcription.load_transcript()
    else:
        qwen_transcript = run_transcription_stage(video_path, audio_path)

    final_result = run_matching_stage(qwen_transcript, args.dialogue, video_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\n=== Final Result ===")
    pprint.pprint(final_result)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()