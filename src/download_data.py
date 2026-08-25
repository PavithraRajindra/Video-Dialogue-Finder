"""
(Re)creates data/video.mp4 and data/audio.wav from the original source
(model-1.ipynb, section 3.1: Source Video Acquisition).

You normally don't need this - data/video.mp4 and data/audio.wav are
already checked into the repo. Run this only if you need to refresh or
recreate the source clip from scratch.

The Kaggle dataset-upload steps from the original notebook are left out
here since the rest of the pipeline reads video/audio straight out of
data/, with no Kaggle dataset in between.

Usage:
    python -m src.download_data
    python -m src.download_data --url "https://..." --max-height 1080
"""

import argparse

import config
import utils


def main():
    parser = argparse.ArgumentParser(description="Download/refresh the source video + audio")
    parser.add_argument("--url", default=config.SOURCE_VIDEO_URL,
                         help="Source video URL (passed to yt-dlp).")
    parser.add_argument("--max-height", type=int, default=config.MAX_VIDEO_HEIGHT,
                         help="Cap the downloaded video's height, e.g. 720.")
    parser.add_argument("--video-out", default=str(config.INPUT_VIDEO_PATH),
                         help="Where to save the downloaded video.")
    parser.add_argument("--audio-out", default=str(config.INPUT_AUDIO_PATH),
                         help="Where to save the extracted audio.")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite video/audio even if they already exist.")
    args = parser.parse_args()

    if not args.force and utils.Path(args.video_out).exists():
        print(f"{args.video_out} already exists, skipping download (use --force to overwrite).")
    else:
        utils.download_video(args.url, args.video_out, max_height=args.max_height)

    if not args.force and utils.Path(args.audio_out).exists():
        print(f"{args.audio_out} already exists, skipping audio extraction (use --force to overwrite).")
    else:
        utils.extract_audio(args.video_out, args.audio_out)

    print("Done.")


if __name__ == "__main__":
    main()