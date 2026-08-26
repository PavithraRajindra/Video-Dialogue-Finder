import download_data


class TestDownloadDataMain:
    def test_downloads_and_extracts_when_nothing_exists(self, tmp_path, monkeypatch):
        calls = {}
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"

        monkeypatch.setattr(download_data.utils, "download_video",
                             lambda url, out, max_height: calls.setdefault("download", (url, str(out), max_height)))
        monkeypatch.setattr(download_data.utils, "extract_audio",
                             lambda video, audio: calls.setdefault("extract", (str(video), str(audio))))

        monkeypatch.setattr(
            "sys.argv",
            ["download_data.py", "--url", "https://example.com/v", "--max-height", "480",
             "--video-out", str(video_path), "--audio-out", str(audio_path)],
        )

        download_data.main()

        assert calls["download"] == ("https://example.com/v", str(video_path), 480)
        assert calls["extract"] == (str(video_path), str(audio_path))

    def test_skips_download_and_extraction_when_files_already_exist(self, tmp_path, monkeypatch):
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"v")
        audio_path.write_bytes(b"a")

        def fail(*args, **kwargs):
            raise AssertionError("should not be called when files already exist")

        monkeypatch.setattr(download_data.utils, "download_video", fail)
        monkeypatch.setattr(download_data.utils, "extract_audio", fail)

        monkeypatch.setattr(
            "sys.argv",
            ["download_data.py", "--video-out", str(video_path), "--audio-out", str(audio_path)],
        )

        download_data.main()  # no AssertionError -> both were correctly skipped

    def test_force_overwrites_existing_files(self, tmp_path, monkeypatch):
        calls = {"download": False, "extract": False}
        video_path = tmp_path / "video.mp4"
        audio_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"v")
        audio_path.write_bytes(b"a")

        monkeypatch.setattr(download_data.utils, "download_video",
                             lambda url, out, max_height: calls.__setitem__("download", True))
        monkeypatch.setattr(download_data.utils, "extract_audio",
                             lambda video, audio: calls.__setitem__("extract", True))

        monkeypatch.setattr(
            "sys.argv",
            ["download_data.py", "--video-out", str(video_path), "--audio-out", str(audio_path), "--force"],
        )

        download_data.main()

        assert calls["download"] is True
        assert calls["extract"] is True

    def test_uses_config_defaults_when_no_args_given(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(download_data.utils, "download_video",
                             lambda url, out, max_height: captured.setdefault("args", (url, out, max_height)))
        monkeypatch.setattr(download_data.utils, "extract_audio", lambda video, audio: None)
        # Force the "doesn't exist yet" branch regardless of the real data/ dir
        monkeypatch.setattr(download_data.utils.Path, "exists", lambda self: False)

        monkeypatch.setattr("sys.argv", ["download_data.py"])

        download_data.main()

        url, out, max_height = captured["args"]
        assert url == download_data.config.SOURCE_VIDEO_URL
        assert max_height == download_data.config.MAX_VIDEO_HEIGHT