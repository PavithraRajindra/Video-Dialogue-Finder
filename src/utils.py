"""
Low-level, model-agnostic helpers used throughout the pipeline:
video/audio I/O (model-1.ipynb section 2.1) and text normalization /
timestamp formatting (model-1.ipynb section 2.2, model-2.ipynb section 4/15).
"""

import re
import subprocess
from pathlib import Path

import cv2
import torchaudio


# ---------------------------------------------------------------------------
# Source acquisition
# ---------------------------------------------------------------------------
def download_video(url, output_path, max_height=None):
    """
    Download a source video with yt-dlp (model-1.ipynb, section 3.1).
    This is how data/video.mp4 was originally created - re-run it via
    download_data.py if you ever need to refresh/recreate the source clip.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_height is None:
        format_option = "bestvideo+bestaudio/best"
    else:
        format_option = (
            f"bestvideo[height<={max_height}]"
            f"+bestaudio/"
            f"best[height<={max_height}]"
        )

    command = [
        "yt-dlp",
        "-f", format_option,
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url,
    ]

    print("Downloading video...")
    print(f"Output: {output_path}")

    subprocess.run(command, check=True)

    if not output_path.exists():
        raise RuntimeError(
            "Video download completed, but the output file was not found."
        )

    print("Download complete.")
    return output_path


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def load_video(path):
    """Open a video file and return (cv2.VideoCapture, metadata dict)."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at path: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }
    return cap, metadata


def print_video_info(metadata):
    print(f"FPS: {metadata['fps']}")
    print(f"Total Frames: {metadata['frame_count']}")
    print(f"Duration: {metadata['duration']:.2f} seconds")


def get_frame(cap, frame_number):
    """Grab a single frame (RGB) by index from an already-open capture."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def save_frame(frame, output_path):
    """Save an RGB frame (as returned by get_frame) to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_frame)


def extract_frame_at_timestamp(video_path, timestamp_ms, output_path):
    """
    Seek to `timestamp_ms` in `video_path` and save that frame to
    `output_path`. Returns frame number / actual timestamp actually hit,
    since cv2's millisecond seeking can land on a neighbouring frame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    success, frame = cap.read()

    if not success:
        cap.release()
        raise RuntimeError(f"Could not extract frame at {timestamp_ms} ms")

    # OpenCV reports the *next* frame index after read(), hence subtract 1.
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    actual_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    cv2.imwrite(str(output_path), frame)
    cap.release()

    return {
        "frame_number": frame_number,
        "actual_timestamp_ms": actual_timestamp_ms,
        "frame_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def extract_audio(video_path, audio_path):
    """Extract mono 16kHz PCM audio from a video using ffmpeg."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]

    subprocess.run(command, check=True)

    if not audio_path.exists():
        raise RuntimeError("Audio extraction failed; file was not created.")

    return audio_path


def split_audio(audio_path, output_dir, chunk_seconds=60, overlap_seconds=5):
    """
    Split an audio file into overlapping chunks (for ASR on long audio).
    Returns a list of {index, path, start_seconds, end_seconds} dicts.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = torchaudio.load(str(audio_path))

    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step_samples = chunk_samples - overlap_samples
    total_samples = waveform.shape[1]

    chunks = []
    start = 0
    chunk_index = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]

        chunk_path = output_dir / f"chunk_{chunk_index:04d}.wav"
        torchaudio.save(str(chunk_path), chunk, sample_rate)

        chunks.append({
            "index": chunk_index,
            "path": chunk_path,
            "start_seconds": start / sample_rate,
            "end_seconds": end / sample_rate,
        })

        chunk_index += 1
        start += step_samples

    return chunks


# ---------------------------------------------------------------------------
# Text / timestamp helpers
# ---------------------------------------------------------------------------
def normalize_text(text):
    """
    Lowercase, strip punctuation (keeping apostrophes so contractions
    like "it's" survive), and collapse whitespace.
    """
    text = text.lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_timestamp(milliseconds):
    """Format a millisecond offset as HH:MM:SS.mmm."""
    total_seconds = milliseconds / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"