import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest

import utils


# ---------------------------------------------------------------------------
# Text / timestamp helpers (pure functions - no mocking needed)
# ---------------------------------------------------------------------------
class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self):
        assert utils.normalize_text("Hello, World!!") == "hello world"

    def test_collapses_whitespace(self):
        assert utils.normalize_text("too    many   spaces") == "too many spaces"

    def test_keeps_apostrophes_for_contractions(self):
        assert utils.normalize_text("It's a test") == "it's a test"

    def test_normalizes_curly_apostrophe(self):
        assert utils.normalize_text("It\u2019s fine") == "it's fine"

    def test_empty_string(self):
        assert utils.normalize_text("") == ""


class TestFormatTimestamp:
    def test_zero(self):
        assert utils.format_timestamp(0) == "00:00:00.000"

    def test_seconds_and_millis(self):
        assert utils.format_timestamp(1500) == "00:00:01.500"

    def test_minutes_boundary(self):
        assert utils.format_timestamp(60_000) == "00:01:00.000"

    def test_hours_boundary(self):
        assert utils.format_timestamp(3_661_250) == "01:01:01.250"


# ---------------------------------------------------------------------------
# download_video / extract_audio - subprocess is mocked, no real yt-dlp/ffmpeg
# ---------------------------------------------------------------------------
class TestDownloadVideo:
    def test_builds_expected_command_with_max_height(self, tmp_path, monkeypatch):
        output_path = tmp_path / "video.mp4"
        recorded = {}

        def fake_run(command, check):
            recorded["command"] = command
            recorded["check"] = check
            output_path.write_bytes(b"fake video")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = utils.download_video("https://example.com/v", output_path, max_height=720)

        assert result == output_path
        assert recorded["check"] is True
        command = recorded["command"]
        assert command[0] == "yt-dlp"
        assert "-f" in command
        fmt = command[command.index("-f") + 1]
        assert "height<=720" in fmt
        assert str(output_path) in command
        assert "https://example.com/v" in command

    def test_no_max_height_uses_best_format(self, tmp_path, monkeypatch):
        output_path = tmp_path / "video.mp4"

        def fake_run(command, check):
            output_path.write_bytes(b"fake video")

        monkeypatch.setattr(subprocess, "run", fake_run)

        utils.download_video("https://example.com/v", output_path)
        # No assertion error means the "best" branch ran without needing height

    def test_raises_if_output_missing_after_download(self, tmp_path, monkeypatch):
        output_path = tmp_path / "video.mp4"
        monkeypatch.setattr(subprocess, "run", lambda command, check: None)

        with pytest.raises(RuntimeError, match="output file was not found"):
            utils.download_video("https://example.com/v", output_path)


class TestExtractAudio:
    def test_builds_expected_ffmpeg_command(self, tmp_path, monkeypatch):
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "out" / "audio.wav"
        recorded = {}

        def fake_run(command, check):
            recorded["command"] = command
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"fake audio")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = utils.extract_audio(video_path, audio_path)

        assert result == audio_path
        command = recorded["command"]
        assert command[0] == "ffmpeg"
        assert str(video_path) in command
        assert str(audio_path) in command
        assert "16000" in command  # sample rate
        assert "1" in command      # mono channel count

    def test_raises_if_audio_missing_after_extraction(self, tmp_path, monkeypatch):
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        monkeypatch.setattr(subprocess, "run", lambda command, check: None)

        with pytest.raises(RuntimeError, match="Audio extraction failed"):
            utils.extract_audio(video_path, audio_path)


# ---------------------------------------------------------------------------
# split_audio - torchaudio is stubbed; we swap in numpy arrays that support
# the same .shape / slicing operations the real tensors would.
# ---------------------------------------------------------------------------
class TestSplitAudio:
    def _fake_waveform(self, total_samples, channels=1):
        return np.zeros((channels, total_samples), dtype=np.float32)

    def test_chunk_boundaries_no_remainder(self, tmp_path, monkeypatch):
        sample_rate = 100
        total_samples = 1000  # 10 seconds
        waveform = self._fake_waveform(total_samples)

        monkeypatch.setattr(utils.torchaudio, "load",
                             lambda path: (waveform, sample_rate))
        saved = []
        monkeypatch.setattr(utils.torchaudio, "save",
                             lambda path, chunk, sr: saved.append((path, chunk.shape, sr)))

        chunks = utils.split_audio(
            tmp_path / "audio.wav", tmp_path / "chunks",
            chunk_seconds=5, overlap_seconds=1,
        )

        # step = (5-1)*100 = 400 samples/step, chunk = 500 samples
        # starts: 0, 400, 800 -> ends: 500, 900, 1000
        assert [c["index"] for c in chunks] == [0, 1, 2]
        assert chunks[0]["start_seconds"] == 0
        assert chunks[0]["end_seconds"] == 5.0
        assert chunks[1]["start_seconds"] == 4.0
        assert chunks[2]["end_seconds"] == 10.0
        assert len(saved) == 3

    def test_output_dir_is_created(self, tmp_path, monkeypatch):
        waveform = self._fake_waveform(200)
        monkeypatch.setattr(utils.torchaudio, "load", lambda path: (waveform, 100))
        monkeypatch.setattr(utils.torchaudio, "save", lambda path, chunk, sr: None)

        out_dir = tmp_path / "nested" / "chunks"
        assert not out_dir.exists()

        utils.split_audio(tmp_path / "audio.wav", out_dir, chunk_seconds=5, overlap_seconds=1)

        assert out_dir.is_dir()

    def test_chunk_paths_are_zero_padded_and_sequential(self, tmp_path, monkeypatch):
        waveform = self._fake_waveform(150)
        monkeypatch.setattr(utils.torchaudio, "load", lambda path: (waveform, 100))
        monkeypatch.setattr(utils.torchaudio, "save", lambda path, chunk, sr: None)

        chunks = utils.split_audio(tmp_path / "audio.wav", tmp_path / "chunks",
                                    chunk_seconds=1, overlap_seconds=0)

        names = [c["path"].name for c in chunks]
        assert names == sorted(names)
        assert names[0] == "chunk_0000.wav"


# ---------------------------------------------------------------------------
# Video helpers - cv2 is fully stubbed; we build a lightweight fake capture.
# ---------------------------------------------------------------------------
class FakeCapture:
    def __init__(self, opened=True, fps=25.0, frame_count=250, read_result=(True, "raw_frame")):
        self._opened = opened
        self._fps = fps
        self._frame_count = frame_count
        self._read_result = read_result
        self.set_calls = []
        self.released = False
        self._pos_frames = 0
        self._pos_msec = 0.0

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == utils.cv2.CAP_PROP_FPS:
            return self._fps
        if prop == utils.cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if prop == utils.cv2.CAP_PROP_POS_FRAMES:
            return self._pos_frames
        if prop == utils.cv2.CAP_PROP_POS_MSEC:
            return self._pos_msec
        return 0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop == utils.cv2.CAP_PROP_POS_FRAMES:
            self._pos_frames = value + 1  # cv2 reports "next" frame after read
        if prop == utils.cv2.CAP_PROP_POS_MSEC:
            self._pos_msec = value
            self._pos_frames = int(value / 1000 * self._fps) + 1

    def read(self):
        return self._read_result

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def _reset_cv2_constants():
    """Give the stubbed cv2 module stable, distinguishable constant values."""
    utils.cv2.CAP_PROP_FPS = "FPS"
    utils.cv2.CAP_PROP_FRAME_COUNT = "FRAME_COUNT"
    utils.cv2.CAP_PROP_POS_FRAMES = "POS_FRAMES"
    utils.cv2.CAP_PROP_POS_MSEC = "POS_MSEC"
    utils.cv2.COLOR_BGR2RGB = "BGR2RGB"
    utils.cv2.COLOR_RGB2BGR = "RGB2BGR"
    utils.cv2.cvtColor = lambda frame, code: (frame, code)
    yield


class TestLoadVideo:
    def test_returns_capture_and_metadata(self, monkeypatch):
        fake_cap = FakeCapture(fps=30.0, frame_count=300)
        monkeypatch.setattr(utils.cv2, "VideoCapture", lambda path: fake_cap)

        cap, metadata = utils.load_video("some/path.mp4")

        assert cap is fake_cap
        assert metadata["fps"] == 30.0
        assert metadata["frame_count"] == 300
        assert metadata["duration"] == pytest.approx(10.0)

    def test_raises_when_capture_fails_to_open(self, monkeypatch):
        fake_cap = FakeCapture(opened=False)
        monkeypatch.setattr(utils.cv2, "VideoCapture", lambda path: fake_cap)

        with pytest.raises(ValueError, match="Could not open video"):
            utils.load_video("bad/path.mp4")

    def test_zero_fps_gives_zero_duration(self, monkeypatch):
        fake_cap = FakeCapture(fps=0, frame_count=100)
        monkeypatch.setattr(utils.cv2, "VideoCapture", lambda path: fake_cap)

        _, metadata = utils.load_video("path.mp4")
        assert metadata["duration"] == 0


class TestPrintVideoInfo(object):
    def test_prints_expected_lines(self, capsys):
        utils.print_video_info({"fps": 29.97, "frame_count": 900, "duration": 30.03})
        out = capsys.readouterr().out
        assert "FPS: 29.97" in out
        assert "Total Frames: 900" in out
        assert "Duration: 30.03 seconds" in out


class TestGetFrame:
    def test_returns_converted_frame_on_success(self, monkeypatch):
        cap = FakeCapture(read_result=(True, "raw_frame"))
        frame = utils.get_frame(cap, frame_number=42)

        assert cap.set_calls == [(utils.cv2.CAP_PROP_POS_FRAMES, 42)]
        assert frame == ("raw_frame", utils.cv2.COLOR_BGR2RGB)

    def test_returns_none_on_failed_read(self):
        cap = FakeCapture(read_result=(False, None))
        assert utils.get_frame(cap, frame_number=0) is None


class TestSaveFrame:
    def test_writes_converted_frame_to_disk(self, tmp_path, monkeypatch):
        written = {}
        monkeypatch.setattr(utils.cv2, "imwrite",
                             lambda path, frame: written.update(path=path, frame=frame))

        out_path = tmp_path / "nested" / "frame.jpg"
        utils.save_frame("rgb_frame", out_path)

        assert out_path.parent.is_dir()
        assert written["path"] == str(out_path)
        assert written["frame"] == ("rgb_frame", utils.cv2.COLOR_RGB2BGR)


class TestExtractFrameAtTimestamp:
    def test_extracts_frame_and_reports_position(self, tmp_path, monkeypatch):
        cap = FakeCapture(fps=25.0)
        monkeypatch.setattr(utils.cv2, "VideoCapture", lambda path: cap)
        monkeypatch.setattr(utils.cv2, "imwrite", lambda path, frame: None)

        out_path = tmp_path / "frame.jpg"
        info = utils.extract_frame_at_timestamp("video.mp4", 2000, out_path)

        assert (utils.cv2.CAP_PROP_POS_MSEC, 2000) in cap.set_calls
        assert info["frame_path"] == str(out_path)
        assert info["actual_timestamp_ms"] == 2000
        assert cap.released is True

    def test_raises_when_capture_fails_to_open(self, tmp_path, monkeypatch):
        cap = FakeCapture(opened=False)
        monkeypatch.setattr(utils.cv2, "VideoCapture", lambda path: cap)

        with pytest.raises(RuntimeError, match="Could not open video"):
            utils.extract_frame_at_timestamp("video.mp4", 0, tmp_path / "f.jpg")

    def test_raises_when_read_fails(self, tmp_path, monkeypatch):
        cap = FakeCapture(read_result=(False, None))
        monkeypatch.setattr(utils.cv2, "VideoCapture", lambda path: cap)

        with pytest.raises(RuntimeError, match="Could not extract frame"):
            utils.extract_frame_at_timestamp("video.mp4", 500, tmp_path / "f.jpg")