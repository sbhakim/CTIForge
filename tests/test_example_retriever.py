"""Tests for retrieval-backed few-shot example selection."""

import json
from pathlib import Path

from src.extraction.example_retriever import FewShotExampleRetriever


def test_retriever_returns_relevant_examples(tmp_path: Path):
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    first = {
        "text": "APT29 used Cobalt Strike to target government networks.",
        "explicit_triplets": [
            {"subject": "APT29", "relation": "uses", "object": "Cobalt Strike"},
        ],
    }
    second = {
        "text": "LockBit deployed ransomware payloads against enterprises.",
        "explicit_triplets": [
            {"subject": "LockBit", "relation": "delivers", "object": "ransomware"},
        ],
    }

    (examples_dir / "apt29.json").write_text(json.dumps(first), encoding="utf-8")
    (examples_dir / "lockbit.json").write_text(json.dumps(second), encoding="utf-8")

    retriever = FewShotExampleRetriever(examples_dir, max_examples=1)
    results = retriever.retrieve("APT29 operators used Cobalt Strike in this intrusion.")

    assert len(results) == 1
    assert "APT29" in results[0]["text"]


def test_retriever_excludes_current_document(tmp_path: Path):
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    data = {
        "text": "APT29 used Cobalt Strike.",
        "explicit_triplets": [
            {"subject": "APT29", "relation": "uses", "object": "Cobalt Strike"},
        ],
    }
    (examples_dir / "same_doc.json").write_text(json.dumps(data), encoding="utf-8")

    retriever = FewShotExampleRetriever(examples_dir, max_examples=1)
    results = retriever.retrieve("APT29 used Cobalt Strike.", exclude_doc_id="same_doc")

    assert results == []
