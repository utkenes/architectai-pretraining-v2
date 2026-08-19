"""Scalable MinHash + LSH near-duplicate detection for corpus curation."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from architectai_pretraining.models import CorpusDocument


@dataclass
class NearDuplicateCluster:
    cluster_id: str
    canonical_document_id: str
    removed_document_ids: list[str]
    source_ids: list[str]
    estimated_similarity: float


@dataclass
class NearDedupResult:
    canonical_documents: list[CorpusDocument]
    removed_documents: list[CorpusDocument]
    clusters: list[NearDuplicateCluster]
    params: dict[str, Any] = field(default_factory=dict)


def _get_shingles(text: str, shingle_size: int = 5, max_shingles: int = 512) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < shingle_size:
        return {" ".join(words)} if words else set()
    positions = range(len(words) - shingle_size + 1)
    # Very large documentation pages can contain tens of thousands of shingles.
    # Evenly spaced deterministic sampling preserves corpus-wide scalability while
    # avoiding a data-size-dependent quadratic CPU surprise during curation.
    if len(words) > max_shingles + shingle_size:
        stride = max(1, (len(words) - shingle_size + 1) // max_shingles)
        positions = range(0, len(words) - shingle_size + 1, stride)
    return {" ".join(words[i : i + shingle_size]) for i in positions}


_MERSENNE_PRIME = 2147483647
_MAX_HASH = 0xFFFFFFFF


def _get_hash_coefficients(num_perm: int, seed: int = 42) -> tuple[list[int], list[int]]:
    import random

    rnd = random.Random(seed)
    a = [rnd.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    b = [rnd.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    return a, b


def _minhash_signature(
    shingles: set[str], coeff_a: list[int], coeff_b: list[int], num_perm: int = 128
) -> list[int]:
    if not shingles:
        return [0] * num_perm

    sig = [_MAX_HASH] * num_perm
    # Python's hash() is intentionally randomized between processes. MinHash must
    # use a portable digest or near-duplicate decisions can alter corpus outputs.
    shingle_hashes = [
        int.from_bytes(hashlib.sha256(shingle.encode()).digest()[:8], "big") & 0x7FFFFFFF
        for shingle in shingles
    ]

    for h in shingle_hashes:
        for i in range(num_perm):
            val = (coeff_a[i] * h + coeff_b[i]) % _MERSENNE_PRIME
            if val < sig[i]:
                sig[i] = val
    return sig


def _jaccard_similarity(sig1: list[int], sig2: list[int]) -> float:
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2, strict=True) if a == b)
    return matches / len(sig1)


class MinHashLSHDeduplicator:
    """Scalable MinHash + LSH near-duplicate detector."""

    def __init__(
        self,
        num_perm: int = 128,
        shingle_size: int = 5,
        similarity_threshold: float = 0.85,
        num_bands: int = 16,
        rows_per_band: int = 8,
        seed: int = 42,
        max_shingles: int = 512,
    ) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.similarity_threshold = similarity_threshold
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.seed = seed
        self.max_shingles = max_shingles
        self.coeff_a, self.coeff_b = _get_hash_coefficients(num_perm, seed)

    def deduplicate(
        self,
        docs: Sequence[CorpusDocument],
        quality_scores: dict[str, float] | None = None,
    ) -> NearDedupResult:
        if not docs:
            return NearDedupResult(
                canonical_documents=[],
                removed_documents=[],
                clusters=[],
                params=self.get_params(),
            )

        # 1. Compute MinHash signatures
        signatures: dict[str, list[int]] = {}
        for doc in docs:
            shingles = _get_shingles(doc.text, self.shingle_size, self.max_shingles)
            signatures[doc.id] = _minhash_signature(
                shingles, self.coeff_a, self.coeff_b, self.num_perm
            )

        # 2. LSH Band Bucketing
        buckets: dict[tuple[int, str], list[str]] = {}
        for doc in docs:
            sig = signatures[doc.id]
            for b in range(self.num_bands):
                start = b * self.rows_per_band
                end = start + self.rows_per_band
                band_hash = hashlib.sha256(
                    str(sig[start:end]).encode("utf-8")
                ).hexdigest()[:16]
                key = (b, band_hash)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(doc.id)

        # 3. Candidate Pairs
        candidate_pairs: set[tuple[str, str]] = set()
        for doc_ids in buckets.values():
            if len(doc_ids) > 1:
                sorted_ids = sorted(doc_ids)
                for i in range(len(sorted_ids)):
                    for j in range(i + 1, len(sorted_ids)):
                        candidate_pairs.add((sorted_ids[i], sorted_ids[j]))

        # 4. Filter Candidate Pairs by similarity threshold & build graph
        adj: dict[str, set[str]] = {doc.id: set() for doc in docs}
        pair_sims: dict[tuple[str, str], float] = {}

        for id1, id2 in candidate_pairs:
            sim = _jaccard_similarity(signatures[id1], signatures[id2])
            if sim >= self.similarity_threshold:
                adj[id1].add(id2)
                adj[id2].add(id1)
                pair_sims[(id1, id2)] = sim

        # 5. Connected components (Clusters)
        visited: set[str] = set()
        clusters: list[NearDuplicateCluster] = []
        canonical_ids: set[str] = set()
        removed_ids: set[str] = set()
        doc_map = {doc.id: doc for doc in docs}

        cluster_idx = 1
        for doc in docs:
            if doc.id in visited:
                continue

            # BFS to find cluster
            component: list[str] = []
            queue = [doc.id]
            visited.add(doc.id)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) == 1:
                canonical_ids.add(component[0])
            else:
                # Select canonical document (highest quality score or longest text)
                if quality_scores:
                    component.sort(
                        key=lambda x: (quality_scores.get(x, 0.0), len(doc_map[x].text)),
                        reverse=True,
                    )
                else:
                    component.sort(key=lambda x: len(doc_map[x].text), reverse=True)

                canonical = component[0]
                removed = component[1:]

                canonical_ids.add(canonical)
                removed_ids.update(removed)

                sources = list({doc_map[cid].source_id for cid in component})
                avg_sim = (
                    sum(
                        pair_sims.get((min(canonical, r), max(canonical, r)), self.similarity_threshold)
                        for r in removed
                    )
                    / len(removed)
                )

                clusters.append(
                    NearDuplicateCluster(
                        cluster_id=f"near_dup_{cluster_idx:04d}",
                        canonical_document_id=canonical,
                        removed_document_ids=removed,
                        source_ids=sources,
                        estimated_similarity=round(avg_sim, 4),
                    )
                )
                cluster_idx += 1

        canonical_docs = [doc for doc in docs if doc.id in canonical_ids]
        removed_docs = [doc for doc in docs if doc.id in removed_ids]

        return NearDedupResult(
            canonical_documents=canonical_docs,
            removed_documents=removed_docs,
            clusters=clusters,
            params=self.get_params(),
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "num_perm": self.num_perm,
            "shingle_size": self.shingle_size,
            "similarity_threshold": self.similarity_threshold,
            "num_bands": self.num_bands,
            "rows_per_band": self.rows_per_band,
            "max_shingles": self.max_shingles,
        }
