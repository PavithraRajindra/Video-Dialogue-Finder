import json
from types import SimpleNamespace

import config
import transcription


# ---------------------------------------------------------------------------
# load_qwen_model / unload_qwen_model
# ---------------------------------------------------------------------------
class TestLoadQwenModel:
    def test_passes_expected_arguments(self, monkeypatch):
        captured = {}

        def fake_from_pretrained(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs
            return "the_model"

        monkeypatch.setattr(transcription.Qwen3ASRModel, "from_pretrained", fake_from_pretrained)

        result = transcription.load_qwen_model()

        assert result == "the_model"
        assert captured["model_name"] == config.ASR_MODEL_NAME
        assert captured["kwargs"]["max_new_tokens"] == config.MAX_NEW_TOKENS
        assert captured["kwargs"]["forced_aligner"] == config.FORCED_ALIGNER_MODEL_NAME


class TestUnloadQwenModel:
    def test_runs_without_error(self, monkeypatch):
        fake_model = SimpleNamespace()
        # torch is stubbed for tests - give the memory-stat calls numeric
        # return values so the f-string formatting in the real function works.
        monkeypatch.setattr(transcription.torch.cuda, "memory_allocated", lambda: 0)
        monkeypatch.setattr(transcription.torch.cuda, "memory_reserved", lambda: 0)

        transcription.unload_qwen_model(fake_model)


# ---------------------------------------------------------------------------
# transcribe_chunks
# ---------------------------------------------------------------------------
class FakeTimeStamp:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class FakeResult:
    def __init__(self, text, time_stamps):
        self.text = text
        self.time_stamps = time_stamps


class TestTranscribeChunks:
    def test_transcribes_and_saves_each_chunk(self, tmp_path):
        chunks = [
            {"index": 0, "path": "chunk_0.wav", "start_seconds": 0.0, "end_seconds": 5.0},
            {"index": 1, "path": "chunk_1.wav", "start_seconds": 4.0, "end_seconds": 9.0},
        ]

        results_by_path = {
            "chunk_0.wav": [FakeResult("hello world", [
                FakeTimeStamp("hello", 0.0, 0.5),
                FakeTimeStamp("world", 0.5, 1.0),
            ])],
            "chunk_1.wav": [FakeResult("", [])],
        }

        class FakeModel:
            def transcribe(self, audio, language, return_time_stamps):
                return results_by_path[audio]

        results_path = tmp_path / "chunk_results.json"
        all_results = transcription.transcribe_chunks(FakeModel(), chunks, results_path=results_path)

        assert len(all_results) == 2
        assert all_results[0]["text"] == "hello world"
        assert all_results[0]["time_stamps"] == [
            {"text": "hello", "start_time": 0.0, "end_time": 0.5},
            {"text": "world", "start_time": 0.5, "end_time": 1.0},
        ]
        assert all_results[1]["text"] == ""
        assert all_results[1]["time_stamps"] == []

        # Progress is saved after every chunk
        with open(results_path) as f:
            saved = json.load(f)
        assert saved == all_results

    def test_handles_whitespace_only_text_as_silence(self, tmp_path):
        chunks = [{"index": 0, "path": "c.wav", "start_seconds": 0.0, "end_seconds": 1.0}]

        class FakeModel:
            def transcribe(self, audio, language, return_time_stamps):
                return [FakeResult("   ", [])]

        results = transcription.transcribe_chunks(FakeModel(), chunks, results_path=tmp_path / "r.json")
        assert results[0]["text"] == ""


# ---------------------------------------------------------------------------
# build_global_transcript
# ---------------------------------------------------------------------------
class TestBuildGlobalTranscript:
    def test_aligns_chunk_relative_to_global_ms(self):
        all_results = [
            {
                "chunk_index": 0,
                "chunk_start_seconds": 10.0,
                "text": "hi",
                "time_stamps": [{"text": "hi", "start_time": 0.0, "end_time": 0.5}],
            },
        ]

        transcript = transcription.build_global_transcript(all_results)

        assert transcript == [
            {"text": "hi", "start_ms": 10000, "end_ms": 10500, "chunk_index": 0},
        ]

    def test_skips_silent_chunks(self):
        all_results = [
            {"chunk_index": 0, "chunk_start_seconds": 0.0, "text": "", "time_stamps": []},
            {
                "chunk_index": 1,
                "chunk_start_seconds": 5.0,
                "text": "hey",
                "time_stamps": [{"text": "hey", "start_time": 0.0, "end_time": 0.4}],
            },
        ]

        transcript = transcription.build_global_transcript(all_results)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "hey"

    def test_sorts_by_start_time_across_overlapping_chunks(self):
        # Chunk 1 starts later in the file but its start_seconds overlaps
        # chunk 0's tail - the flattened transcript should still come out
        # in chronological order.
        all_results = [
            {
                "chunk_index": 1,
                "chunk_start_seconds": 55.0,
                "text": "later",
                "time_stamps": [{"text": "later", "start_time": 0.0, "end_time": 0.5}],
            },
            {
                "chunk_index": 0,
                "chunk_start_seconds": 0.0,
                "text": "earlier",
                "time_stamps": [{"text": "earlier", "start_time": 0.0, "end_time": 0.5}],
            },
        ]

        transcript = transcription.build_global_transcript(all_results)
        assert [item["text"] for item in transcript] == ["earlier", "later"]


# ---------------------------------------------------------------------------
# save_transcript / load_transcript
# ---------------------------------------------------------------------------
class TestTranscriptPersistence:
    def test_round_trip(self, tmp_path):
        transcript = [{"text": "hi", "start_ms": 0, "end_ms": 500, "chunk_index": 0}]
        path = tmp_path / "nested" / "transcript.json"

        saved_path = transcription.save_transcript(transcript, path=path)
        assert saved_path == path
        assert path.exists()

        loaded = transcription.load_transcript(path=path)
        assert loaded == transcript