import numpy as np
import pytest

import config
from matching import DialogueMatcher, build_transcript_windows, fuzzy_score
from utils import normalize_text


def make_transcript(words, ms_per_word=500):
    """Build a simple word-level transcript with evenly spaced timestamps."""
    transcript = []
    for i, word in enumerate(words):
        transcript.append({
            "text": word,
            "start_ms": i * ms_per_word,
            "end_ms": (i + 1) * ms_per_word,
        })
    return transcript


class FakeEmbedder:
    """
    Deterministic stand-in for SentenceTransformer: encodes each text as a
    normalized bag-of-words vector over a fixed vocabulary. This gives
    "semantic" similarity that behaves sensibly for tests (identical text
    -> similarity 1.0, overlapping words -> partial similarity) without
    downloading any real model.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            words = text.split()
            vec = np.array([words.count(w) for w in self.vocab], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


def build_vocab(*texts):
    vocab = set()
    for text in texts:
        vocab.update(normalize_text(text).split())
    return sorted(vocab)


# ---------------------------------------------------------------------------
# build_transcript_windows
# ---------------------------------------------------------------------------
class TestBuildTranscriptWindows:
    def test_basic_windowing(self):
        transcript = make_transcript(["a", "b", "c", "d", "e"])
        windows = build_transcript_windows(transcript, window_size=3, stride=2)

        # start=0 -> [a,b,c] (not yet at the end, keep going)
        # start=2 -> [c,d,e] (reaches the end -> stop, no start=4 window)
        assert [w["start_index"] for w in windows] == [0, 2]
        assert windows[0]["text"] == "a b c"
        assert windows[0]["start_ms"] == 0
        assert windows[0]["end_ms"] == 1500
        # Last window always reaches the end of the transcript
        assert windows[-1]["end_index"] == 5

    def test_normalized_text_matches_normalize_text(self):
        transcript = make_transcript(["Hello,", "World"])
        windows = build_transcript_windows(transcript, window_size=2, stride=2)
        assert windows[0]["normalized_text"] == normalize_text("Hello, World")

    def test_empty_transcript_gives_no_windows(self):
        assert build_transcript_windows([], window_size=5, stride=2) == []

    def test_uses_config_defaults_when_not_given(self):
        transcript = make_transcript(["w"] * 40)
        windows = build_transcript_windows(transcript)
        assert windows[0]["end_index"] == min(config.WINDOW_SIZE, len(transcript))


# ---------------------------------------------------------------------------
# fuzzy_score (uses real rapidfuzz - it's lightweight and deterministic)
# ---------------------------------------------------------------------------
class TestFuzzyScore:
    def test_identical_strings_score_one(self):
        assert fuzzy_score("hello world", "hello world") == 1.0

    def test_completely_different_strings_score_low(self):
        assert fuzzy_score("hello world", "zzz qqq xxx") < 0.3

    def test_minor_typo_scores_high_but_not_perfect(self):
        score = fuzzy_score("my mind rebels at stagnation", "my mind rebles at stagnation")
        assert 0.9 < score < 1.0

    def test_ignores_case_and_punctuation(self):
        assert fuzzy_score("Hello, World!", "hello world") == 1.0


# ---------------------------------------------------------------------------
# DialogueMatcher: semantic_search / combined_search / fine_grained_search
# ---------------------------------------------------------------------------
class TestDialogueMatcherSemanticSearch:
    def test_exact_text_match_ranks_first(self):
        transcript = make_transcript(
            "my mind rebels at stagnation the day was cold and grey".split()
        )
        vocab = build_vocab(
            "my mind rebels at stagnation",
            "the day was cold and grey",
            "my mind rebels at stagnation the day",
        )
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab),
            window_size=5, stride=5,
        )

        results = matcher.semantic_search("my mind rebels at stagnation", top_k=2)

        assert results[0]["text"] == "my mind rebels at stagnation"
        assert results[0]["semantic_score"] == pytest.approx(1.0)

    def test_top_k_limits_result_count(self):
        transcript = make_transcript(["word"] * 20)
        vocab = ["word"]
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab), window_size=4, stride=4,
        )
        results = matcher.semantic_search("word", top_k=2)
        assert len(results) == 2


class TestDialogueMatcherCombinedSearch:
    def test_combined_score_formula(self):
        transcript = make_transcript("alpha beta gamma delta".split())
        vocab = build_vocab("alpha beta gamma delta")
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab), window_size=4, stride=4,
        )

        results = matcher.combined_search(
            "alpha beta gamma delta", top_k=1,
            semantic_weight=0.7, fuzzy_weight=0.3,
        )
        top = results[0]
        expected = 0.7 * top["semantic_score"] + 0.3 * top["fuzzy_score"]
        assert top["combined_score"] == pytest.approx(expected)

    def test_results_sorted_descending_by_combined_score(self):
        transcript = make_transcript(
            "completely unrelated words here alpha beta gamma delta".split()
        )
        vocab = build_vocab("completely unrelated words here", "alpha beta gamma delta")
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab), window_size=4, stride=4,
        )

        results = matcher.combined_search("alpha beta gamma delta")
        scores = [r["combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["text"] == "alpha beta gamma delta"


class TestDialogueMatcherFineGrainedSearch:
    def test_finds_exact_word_span(self):
        words = "well before my mind rebels at stagnation and after".split()
        transcript = make_transcript(words)
        vocab = build_vocab(" ".join(words))
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab), window_size=len(words), stride=len(words),
        )

        # Coarse candidate spans the whole transcript; fine search should
        # narrow down to exactly "my mind rebels at stagnation".
        candidate = {"start_index": 0, "end_index": len(words)}
        fine_results = matcher.fine_grained_search(
            "my mind rebels at stagnation", candidate, max_extra_words=2
        )

        best = fine_results[0]
        assert best["text"] == "my mind rebels at stagnation"
        assert best["fuzzy_score"] == pytest.approx(1.0)
        assert best["start_ms"] == words.index("my") * 500
        assert best["end_ms"] == (words.index("stagnation") + 1) * 500


class TestDialogueMatcherFindBestMatch:
    def test_end_to_end_match(self):
        words = "the day was long my mind rebels at stagnation and grey".split()
        transcript = make_transcript(words)
        vocab = build_vocab(" ".join(words))
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab), window_size=5, stride=3,
        )

        match = matcher.find_best_match("my mind rebels at stagnation")

        assert match["matched_dialogue"] == "my mind rebels at stagnation"
        assert match["input_dialogue"] == "my mind rebels at stagnation"
        assert match["fuzzy_score"] == pytest.approx(1.0)
        assert match["start_ms"] < match["end_ms"]

    def test_returns_none_when_no_coarse_results(self):
        transcript = make_transcript(["only", "word"])
        vocab = build_vocab("only word")
        matcher = DialogueMatcher(
            transcript, embedding_model=FakeEmbedder(vocab), window_size=2, stride=2,
        )
        matcher.combined_search = lambda *args, **kwargs: []

        assert matcher.find_best_match("anything") is None