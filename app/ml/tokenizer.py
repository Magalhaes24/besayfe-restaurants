"""
Persistent vocabulary learning system using Byte Pair Encoding (BPE)-inspired approach.
Learns domain-specific tokens from every correction and builds a vocabulary over time.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from app.ml.pattern_store import PatternStore


class Token:
    """Represents a learned token in the vocabulary."""

    def __init__(
        self,
        text: str,
        frequency: int = 1,
        token_type: str = "ingredient",  # ingredient, allergen, compound, pattern
        confidence: float = 0.5,
        language: str = "pt",
    ):
        self.text = text.lower().strip()
        self.frequency = frequency
        self.token_type = token_type
        self.confidence = confidence
        self.language = language
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "frequency": self.frequency,
            "type": self.token_type,
            "confidence": round(self.confidence, 2),
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
        }


class TokenMerge:
    """Represents a learned merge of two tokens into a compound token."""

    def __init__(
        self,
        token_a: str,
        token_b: str,
        merged_token: str,
        frequency: int = 1,
        confidence: float = 0.5,
    ):
        self.token_a = token_a.lower().strip()
        self.token_b = token_b.lower().strip()
        self.merged_token = merged_token.lower().strip()
        self.frequency = frequency
        self.confidence = confidence
        self.created_at = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "a": self.token_a,
            "b": self.token_b,
            "merged": self.merged_token,
            "frequency": self.frequency,
            "confidence": round(self.confidence, 2),
            "created_at": self.created_at.isoformat(),
        }


class TokenVocabulary:
    """
    Learns and manages a persistent vocabulary of tokens and merges.
    Similar to BPE, but domain-specific for restaurant ingredients and allergens.
    """

    def __init__(self, pattern_store: PatternStore):
        self.pattern_store = pattern_store
        self.vocabulary: dict[str, Token] = {}  # text -> Token
        self.merges: list[TokenMerge] = []  # learned token merges
        self.token_sequences: defaultdict(list) = defaultdict(list)  # pairs seen together
        self.cooccurrence_matrix: defaultdict(int) = defaultdict(int)  # (a,b) -> count
        self._load_vocabulary()

    def _load_vocabulary(self):
        """Load persisted vocabulary from database."""
        data = self.pattern_store.load_vocabulary()

        # Reconstruct tokens
        if "vocabulary" in data:
            for text, token_data in data["vocabulary"].items():
                token = Token(
                    text=token_data["text"],
                    frequency=token_data.get("frequency", 1),
                    token_type=token_data.get("type", "ingredient"),
                    confidence=token_data.get("confidence", 0.5),
                    language=token_data.get("language", "pt"),
                )
                # Restore timestamps
                if token_data.get("created_at"):
                    token.created_at = datetime.fromisoformat(token_data["created_at"])
                if token_data.get("last_used"):
                    token.last_used = datetime.fromisoformat(token_data["last_used"])
                self.vocabulary[text.lower().strip()] = token

        # Reconstruct merges
        if "merges" in data:
            for merge_data in data["merges"]:
                merge = TokenMerge(
                    merge_data["a"],
                    merge_data["b"],
                    merge_data["merged"],
                    frequency=merge_data.get("frequency", 1),
                    confidence=merge_data.get("confidence", 0.5),
                )
                if merge_data.get("created_at"):
                    merge.created_at = datetime.fromisoformat(merge_data["created_at"])
                self.merges.append(merge)

        # Reconstruct cooccurrence matrix
        if "cooccurrence_matrix" in data:
            for pair_str, count in data["cooccurrence_matrix"].items():
                a, b = pair_str.split(":")
                self.cooccurrence_matrix[(a, b)] = count

    def observe_token(
        self, text: str, token_type: str = "ingredient", language: str = "pt"
    ) -> Token:
        """Record observation of a token in the corpus."""
        key = text.lower().strip()
        if key in self.vocabulary:
            token = self.vocabulary[key]
            token.frequency += 1
            token.last_used = datetime.utcnow()
        else:
            token = Token(text, frequency=1, token_type=token_type, language=language)
            self.vocabulary[key] = token

        # Persist to database
        self.pattern_store.save_vocabulary_token(
            text=key,
            token_type=token_type,
            frequency=token.frequency,
            confidence=token.confidence,
            language=language,
        )

        return token

    def observe_token_pair(self, token_a: str, token_b: str, context: str = "") -> None:
        """Record observation of two tokens appearing together."""
        a_lower = token_a.lower().strip()
        b_lower = token_b.lower().strip()
        pair = (a_lower, b_lower)

        self.cooccurrence_matrix[pair] += 1
        self.token_sequences[a_lower].append((b_lower, context))

        # Persist to database
        self.pattern_store.save_cooccurrence(a_lower, b_lower, count=1)

    def get_merge_candidates(self, min_frequency: int = 3) -> list[tuple]:
        """
        Get token pairs that appear together frequently enough to merit merging.
        Returns: [(token_a, token_b, frequency), ...]
        """
        candidates = [
            (pair[0], pair[1], count)
            for pair, count in self.cooccurrence_matrix.items()
            if count >= min_frequency
        ]
        # Sort by frequency descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def learn_merge(self, token_a: str, token_b: str, merged_token: str, confidence: float = 0.8) -> None:
        """
        Learn a merge: when token_a and token_b appear together, they form merged_token.
        """
        a_lower = token_a.lower().strip()
        b_lower = token_b.lower().strip()
        merged_lower = merged_token.lower().strip()

        # Record the merge
        merge = TokenMerge(a_lower, b_lower, merged_lower, frequency=1, confidence=confidence)
        self.merges.append(merge)

        # Update cooccurrence confidence based on merge success
        pair = (a_lower, b_lower)
        if pair in self.cooccurrence_matrix:
            # Increase confidence for this pairing
            current_count = self.cooccurrence_matrix[pair]
            self.cooccurrence_matrix[pair] = int(current_count * confidence)

        # Add merged token to vocabulary
        self.observe_token(merged_lower, token_type="compound")

        # Persist merge to database
        self.pattern_store.save_vocabulary_merge(
            token_a=a_lower,
            token_b=b_lower,
            merged_token=merged_lower,
            frequency=1,
            confidence=confidence,
        )

    def apply_tokenization(self, text: str) -> list[str]:
        """
        Apply learned vocabulary and merges to tokenize input text.
        Greedily applies merges in order of creation.
        """
        # Start with word-level tokens
        tokens = text.lower().split()

        # Apply each learned merge in order
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == merge.token_a and
                    tokens[i + 1] == merge.token_b):
                    # Apply merge
                    new_tokens.append(merge.merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def get_vocabulary_stats(self) -> dict:
        """Get statistics about the current vocabulary."""
        total_tokens = len(self.vocabulary)
        total_frequency = sum(t.frequency for t in self.vocabulary.values())
        total_merges = len(self.merges)
        avg_merge_confidence = (
            sum(m.confidence for m in self.merges) / len(self.merges)
            if self.merges
            else 0.0
        )

        return {
            "total_tokens": total_tokens,
            "total_frequency": total_frequency,
            "avg_token_frequency": total_frequency / max(1, total_tokens),
            "total_merges": total_merges,
            "merge_candidates": len(self.get_merge_candidates()),
            "avg_merge_confidence": round(avg_merge_confidence, 2),
            "top_tokens": [
                t.to_dict()
                for t in sorted(
                    self.vocabulary.values(),
                    key=lambda x: x.frequency,
                    reverse=True,
                )[:10]
            ],
            "recent_merges": [
                m.to_dict()
                for m in sorted(
                    self.merges,
                    key=lambda x: x.created_at,
                    reverse=True,
                )[:5]
            ],
        }

    def to_dict(self) -> dict:
        """Export vocabulary for persistence."""
        return {
            "vocabulary": {k: v.to_dict() for k, v in self.vocabulary.items()},
            "merges": [m.to_dict() for m in self.merges],
            "cooccurrence_matrix": {
                f"{a}:{b}": count
                for (a, b), count in self.cooccurrence_matrix.items()
            },
            "created_at": datetime.utcnow().isoformat(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore vocabulary from persisted data."""
        # Reconstruct tokens
        if "vocabulary" in data:
            for text, token_data in data["vocabulary"].items():
                token = Token(
                    text=token_data["text"],
                    frequency=token_data.get("frequency", 1),
                    token_type=token_data.get("type", "ingredient"),
                    confidence=token_data.get("confidence", 0.5),
                    language=token_data.get("language", "pt"),
                )
                self.vocabulary[text.lower().strip()] = token

        # Reconstruct merges
        if "merges" in data:
            for merge_data in data["merges"]:
                merge = TokenMerge(
                    merge_data["a"],
                    merge_data["b"],
                    merge_data["merged"],
                    frequency=merge_data.get("frequency", 1),
                    confidence=merge_data.get("confidence", 0.5),
                )
                self.merges.append(merge)

        # Reconstruct cooccurrence matrix
        if "cooccurrence_matrix" in data:
            for pair_str, count in data["cooccurrence_matrix"].items():
                a, b = pair_str.split(":")
                self.cooccurrence_matrix[(a, b)] = count
