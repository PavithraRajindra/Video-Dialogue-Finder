import config


def test_paths_are_derived_from_base_dir():
    assert config.DATA_DIR == config.BASE_DIR / "data"
    assert config.INPUT_VIDEO_PATH == config.DATA_DIR / "video.mp4"
    assert config.INPUT_AUDIO_PATH == config.DATA_DIR / "audio.wav"
    assert config.TRANSCRIPT_PATH == config.RESULTS_DIR / "qwen_transcript.json"


def test_output_directories_are_created():
    assert config.OUTPUT_DIR.is_dir()
    assert config.CHUNKS_DIR.is_dir()
    assert config.RESULTS_DIR.is_dir()


def test_audio_chunking_constants_are_sane():
    assert config.CHUNK_SECONDS > 0
    assert 0 <= config.OVERLAP_SECONDS < config.CHUNK_SECONDS


def test_matching_weights_sum_to_one():
    assert config.SEMANTIC_WEIGHT + config.FUZZY_WEIGHT == 1.0


def test_source_video_url_and_height_are_set():
    assert config.SOURCE_VIDEO_URL.startswith("http")
    assert config.MAX_VIDEO_HEIGHT > 0