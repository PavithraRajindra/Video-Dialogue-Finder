import json
from pathlib import Path

import pytest

import config
import pipeline


class FakeCap:
    """Minimal stand-in for the cv2.VideoCapture object load_video returns."""
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------
class TestResolvePath:
    def test_relative_path_is_joined_with_base_dir(self):
        result = pipeline._resolve_path("data/video.mp4")
        assert result == config.BASE_DIR / "data" / "video.mp4"

    def test_absolute_path_is_left_untouched(self, tmp_path):
        abs_path = tmp_path / "video.mp4"
        result = pipeline._resolve_path(str(abs_path))
        assert result == abs_path


# ---------------------------------------------------------------------------
# run_transcription_stage
# ---------------------------------------------------------------------------
class TestRunTranscriptionStage:
    def _patch_common(self, monkeypatch, calls):
        fake_cap = FakeCap()
        calls["cap"] = fake_cap
        monkeypatch.setattr(pipeline.utils, "load_video",
                             lambda path: (fake_cap, {"fps": 30}))
        monkeypatch.setattr(pipeline.utils, "print_video_info", lambda metadata: None)
        monkeypatch.setattr(pipeline.utils, "split_audio",
                             lambda audio_path, chunks_dir, chunk_seconds, overlap_seconds:
                             calls.setdefault("chunks", ["chunk0", "chunk1"]))
        monkeypatch.setattr(pipeline.transcription, "load_qwen_model",
                             lambda: calls.setdefault("model", "fake_model"))
        monkeypatch.setattr(pipeline.transcription, "transcribe_chunks",
                             lambda model, chunks: calls.setdefault("raw_results", ["r0", "r1"]))
        monkeypatch.setattr(pipeline.transcription, "unload_qwen_model",
                             lambda model: calls.setdefault("unloaded", model))
        monkeypatch.setattr(pipeline.transcription, "build_global_transcript",
                             lambda raw: calls.setdefault("transcript", [{"text": "hi"}]))
        monkeypatch.setattr(pipeline.transcription, "save_transcript",
                             lambda transcript: calls.setdefault("saved", transcript))

    def test_extracts_audio_when_missing(self, tmp_path, monkeypatch):
        calls = {}
        self._patch_common(monkeypatch, calls)
        monkeypatch.setattr(pipeline.utils, "extract_audio",
                             lambda video, audio: calls.setdefault("extracted", (video, audio)))

        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"  # does not exist

        result = pipeline.run_transcription_stage(video_path, audio_path)

        assert calls["extracted"] == (video_path, audio_path)
        assert calls["model"] == "fake_model"
        assert calls["unloaded"] == "fake_model"
        assert calls["cap"].released is True
        assert result == [{"text": "hi"}]

    def test_skips_extraction_when_audio_exists(self, tmp_path, monkeypatch):
        calls = {}
        self._patch_common(monkeypatch, calls)

        def fail_extract(video, audio):
            raise AssertionError("extract_audio should not be called when audio already exists")

        monkeypatch.setattr(pipeline.utils, "extract_audio", fail_extract)

        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"already here")

        pipeline.run_transcription_stage(video_path, audio_path)
        # No AssertionError raised means extract_audio was correctly skipped

    def test_model_is_unloaded_even_if_transcription_fails(self, tmp_path, monkeypatch):
        calls = {}
        self._patch_common(monkeypatch, calls)

        def boom(model, chunks):
            raise RuntimeError("ASR blew up")

        monkeypatch.setattr(pipeline.transcription, "transcribe_chunks", boom)

        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"data")

        with pytest.raises(RuntimeError, match="ASR blew up"):
            pipeline.run_transcription_stage(video_path, audio_path)

        assert calls["unloaded"] == "fake_model"


# ---------------------------------------------------------------------------
# run_matching_stage
# ---------------------------------------------------------------------------
class TestRunMatchingStage:
    def test_assembles_final_result(self, monkeypatch):
        fake_match = {
            "input_dialogue": "hello",
            "matched_dialogue": "hello there",
            "start_ms": 1000,
            "end_ms": 2000,
            "semantic_score": 0.9,
            "fuzzy_score": 0.8,
            "combined_score": 0.87,
        }

        class FakeMatcher:
            def __init__(self, transcript):
                self.transcript = transcript

            def find_best_match(self, dialogue):
                return fake_match

        monkeypatch.setattr(pipeline, "DialogueMatcher", FakeMatcher)
        monkeypatch.setattr(pipeline.utils, "extract_frame_at_timestamp",
                             lambda video, ms, out: {
                                 "frame_number": 42, "actual_timestamp_ms": 1001.0,
                                 "frame_path": str(out),
                             })
        monkeypatch.setattr(pipeline.utils, "format_timestamp", lambda ms: "00:00:01.000")

        result = pipeline.run_matching_stage([{"text": "hi"}], "hello", "video.mp4")

        assert result["matched_dialogue"] == "hello there"
        assert result["frame_number"] == 42
        assert result["timestamp"] == "00:00:01.000"
        assert result["combined_score"] == 0.87

    def test_raises_when_no_match_found(self, monkeypatch):
        class FakeMatcher:
            def __init__(self, transcript):
                pass

            def find_best_match(self, dialogue):
                return None

        monkeypatch.setattr(pipeline, "DialogueMatcher", FakeMatcher)

        with pytest.raises(RuntimeError, match="No match found"):
            pipeline.run_matching_stage([], "hello", "video.mp4")


# ---------------------------------------------------------------------------
# main() - CLI wiring
# ---------------------------------------------------------------------------
class TestMain:
    def test_skip_transcription_uses_saved_transcript(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline.transcription, "load_transcript",
                             lambda: [{"text": "hi"}])

        def fail_run_transcription(*args, **kwargs):
            raise AssertionError("should not re-run transcription when --skip-transcription is set")

        monkeypatch.setattr(pipeline, "run_transcription_stage", fail_run_transcription)
        monkeypatch.setattr(pipeline, "run_matching_stage",
                             lambda transcript, dialogue, video: {"matched_dialogue": "hi"})

        output_path = tmp_path / "result.json"
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--dialogue", "hi", "--skip-transcription",
             "--output", str(output_path)],
        )

        pipeline.main()

        assert json.loads(output_path.read_text()) == {"matched_dialogue": "hi"}

    def test_default_flow_runs_transcription_then_matching(self, tmp_path, monkeypatch):
        calls = {}

        def fake_run_transcription(video, audio):
            calls["transcribed"] = (video, audio)
            return ["t"]

        def fake_run_matching(transcript, dialogue, video):
            calls["matched"] = (transcript, dialogue, video)
            return {"ok": True}

        monkeypatch.setattr(pipeline, "run_transcription_stage", fake_run_transcription)
        monkeypatch.setattr(pipeline, "run_matching_stage", fake_run_matching)

        output_path = tmp_path / "result.json"
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"

        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--dialogue", "test line",
             "--video", str(video_path), "--audio", str(audio_path),
             "--output", str(output_path)],
        )

        pipeline.main()

        assert calls["transcribed"] == (video_path, audio_path)
        assert calls["matched"] == (["t"], "test line", video_path)
        assert json.loads(output_path.read_text()) == {"ok": True}