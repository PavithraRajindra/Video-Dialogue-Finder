"""
Dialogue-localization stage (model-2.ipynb): given the word-level transcript
produced by transcription.py, find where a target line of dialogue was
spoken - robust to paraphrasing (semantic search) and to minor ASR
transcription errors (fuzzy matching) - then pin down the exact word span.

This is the "final" matching approach: model-1.ipynb's exact-substring
find_dialogue() only works when the query matches the transcript verbatim,
which is why model-2.ipynb's semantic + fuzzy pipeline supersedes it here.
"""

import numpy as np
import torch
from rapidfuzz.fuzz import ratio
from sentence_transformers import SentenceTransformer

import config
from utils import normalize_text


def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)


def build_transcript_windows(transcript, window_size=None, stride=None):
    """
    Group the word-level transcript into overlapping multi-word windows
    so semantic search has enough context per candidate (model-2.ipynb,
    section 5).
    """
    window_size = window_size or config.WINDOW_SIZE
    stride = stride or config.WINDOW_STRIDE

    windows = []
    n = len(transcript)

    for start_idx in range(0, n, stride):
        end_idx = min(start_idx + window_size, n)
        words = transcript[start_idx:end_idx]

        if not words:
            continue

        text = " ".join(item["text"] for item in words)

        windows.append({
            "start_index": start_idx,
            "end_index": end_idx,
            "text": text,
            "normalized_text": normalize_text(text),
            "start_ms": words[0]["start_ms"],
            "end_ms": words[-1]["end_ms"],
        })

        if end_idx == n:
            break

    return windows


def fuzzy_score(query, candidate):
    """Normalized (0-1) edit-distance similarity between two strings."""
    return ratio(normalize_text(query), normalize_text(candidate)) / 100.0


class DialogueMatcher:
    """
    Wraps a transcript + embedding model so semantic search, fuzzy
    search, and fine-grained search (model-2.ipynb, sections 6-13) can
    share the precomputed window embeddings across multiple queries.
    """

    def __init__(self, transcript, embedding_model=None,
                 window_size=None, stride=None):
        self.transcript = transcript
        self.embedding_model = embedding_model or load_embedding_model()
        self.windows = build_transcript_windows(transcript, window_size, stride)

        window_texts = [w["normalized_text"] for w in self.windows]
        self.window_embeddings = self.embedding_model.encode(
            window_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def semantic_search(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        query_normalized = normalize_text(query)

        query_embedding = self.embedding_model.encode(
            [query_normalized],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        similarities = self.window_embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = self.windows[idx].copy()
            result["semantic_score"] = float(similarities[idx])
            result["window_index"] = int(idx)
            results.append(result)

        return results

    def combined_search(self, query, top_k=None,
                         semantic_weight=None, fuzzy_weight=None):
        top_k = top_k or config.TOP_K
        semantic_weight = semantic_weight if semantic_weight is not None else config.SEMANTIC_WEIGHT
        fuzzy_weight = fuzzy_weight if fuzzy_weight is not None else config.FUZZY_WEIGHT

        all_results = self.semantic_search(query, top_k=len(self.windows))

        for result in all_results:
            result["fuzzy_score"] = fuzzy_score(query, result["text"])
            result["combined_score"] = (
                semantic_weight * result["semantic_score"]
                + fuzzy_weight * result["fuzzy_score"]
            )

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results[:top_k]

    def fine_grained_search(self, query, candidate, max_extra_words=None):
        """
        Slide a variable-length word window around a coarse candidate to
        find the exact start/end word span that best matches the query.
        """
        max_extra_words = max_extra_words or config.FINE_MATCH_MAX_EXTRA_WORDS

        query_words = normalize_text(query).split()
        query_length = len(query_words)

        search_start = max(0, candidate["start_index"] - max_extra_words)
        search_end = min(len(self.transcript), candidate["end_index"] + max_extra_words)

        min_length = max(1, query_length - max_extra_words)
        max_length = query_length + max_extra_words

        candidates = []
        for start in range(search_start, search_end):
            for length in range(min_length, max_length + 1):
                end = start + length
                if end > len(self.transcript):
                    continue

                words = self.transcript[start:end]
                text = " ".join(word["text"] for word in words)
                score = fuzzy_score(query, text)

                candidates.append({
                    "text": text,
                    "start_index": start,
                    "end_index": end,
                    "start_ms": words[0]["start_ms"],
                    "end_ms": words[-1]["end_ms"],
                    "fuzzy_score": score,
                })

        candidates.sort(key=lambda x: x["fuzzy_score"], reverse=True)
        return candidates

    def find_best_match(self, query, top_k=None,
                         semantic_weight=None, fuzzy_weight=None,
                         max_extra_words=None):
        """
        Full pipeline for one query: coarse combined search -> fine-grained
        word-span search on the best coarse candidate. Returns a dict with
        the matched text, exact start/end ms, and every intermediate score.
        """
        coarse_results = self.combined_search(
            query, top_k=top_k,
            semantic_weight=semantic_weight, fuzzy_weight=fuzzy_weight,
        )

        if not coarse_results:
            return None

        best_coarse_match = coarse_results[0]
        fine_results = self.fine_grained_search(
            query, best_coarse_match, max_extra_words=max_extra_words
        )

        if not fine_results:
            return None

        best_match = fine_results[0]

        return {
            "input_dialogue": query,
            "matched_dialogue": best_match["text"],
            "start_ms": best_match["start_ms"],
            "end_ms": best_match["end_ms"],
            "semantic_score": best_coarse_match["semantic_score"],
            "fuzzy_score": best_match["fuzzy_score"],
            "combined_score": best_coarse_match["combined_score"],
        }