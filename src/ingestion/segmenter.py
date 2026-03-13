"""Document segmentation into paragraph-level chunks."""

from __future__ import annotations

import re

from src.schema.graph_schema import Chunk, CTIDocument
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def segment_document(
    doc: CTIDocument,
    max_chunk_chars: int = 2000,
    overlap_chars: int = 200,
    max_sentences_per_chunk: int = 0,
) -> CTIDocument:
    """Segment a CTI document into paragraph-level chunks.

    Strategy:
    1. Split on double newlines (paragraph boundaries)
    2. Merge short consecutive paragraphs up to max_chunk_chars
    3. Split overly long paragraphs at sentence boundaries

    Returns the document with chunks populated.
    """
    text = doc.text.strip()
    if not text:
        logger.warning(f"Document {doc.doc_id} has empty text")
        doc.chunks = []
        return doc

    # Step 1: split into raw paragraphs
    raw_paragraphs = re.split(r"\n\s*\n", text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Step 2: merge short paragraphs, split long ones
    merged_chunks: list[str] = []
    buffer = ""

    for para in raw_paragraphs:
        if len(para) > max_chunk_chars:
            # Flush buffer first
            if buffer:
                merged_chunks.append(buffer)
                buffer = ""
            # Split long paragraph at sentence boundaries
            sentences = _split_sentences(para)
            sub_buffer = ""
            sub_sentence_count = 0
            for sent in sentences:
                would_exceed_chars = len(sub_buffer) + len(sent) + 1 > max_chunk_chars
                would_exceed_sentences = (
                    max_sentences_per_chunk > 0 and sub_sentence_count >= max_sentences_per_chunk
                )
                if would_exceed_chars or would_exceed_sentences:
                    if sub_buffer:
                        merged_chunks.append(sub_buffer)
                    sub_buffer = sent
                    sub_sentence_count = 1
                else:
                    sub_buffer = f"{sub_buffer} {sent}".strip() if sub_buffer else sent
                    sub_sentence_count = sub_sentence_count + 1 if sub_buffer else 1
            if sub_buffer:
                merged_chunks.append(sub_buffer)
        elif len(buffer) + len(para) + 2 > max_chunk_chars:
            # Buffer full, flush and start new
            if buffer:
                merged_chunks.append(buffer)
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para

    if buffer:
        if max_sentences_per_chunk > 0:
            split_buffer = _split_by_sentence_limit(buffer, max_chunk_chars, max_sentences_per_chunk)
            merged_chunks.extend(split_buffer)
        else:
            merged_chunks.append(buffer)

    # Step 3: create Chunk objects with stable IDs
    chunks = []
    char_offset = 0
    for i, chunk_text in enumerate(merged_chunks):
        # Find actual position in original text
        pos = text.find(chunk_text[:50], char_offset)
        if pos == -1:
            pos = char_offset

        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_chunk{i}",
            doc_id=doc.doc_id,
            text=chunk_text,
            chunk_index=i,
            char_start=pos,
            char_end=pos + len(chunk_text),
        )
        chunks.append(chunk)
        char_offset = pos + len(chunk_text)

    doc.chunks = chunks
    logger.info(f"Document {doc.doc_id}: {len(chunks)} chunks from {len(text)} chars")
    return doc


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    # Split on period/question/exclamation followed by space and uppercase
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in parts if s.strip()]


def _split_by_sentence_limit(
    text: str,
    max_chunk_chars: int,
    max_sentences_per_chunk: int,
) -> list[str]:
    """Split a buffer by sentence count while respecting character limits."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""
    sentence_count = 0
    for sent in sentences:
        would_exceed_chars = len(current) + len(sent) + 1 > max_chunk_chars
        would_exceed_sentences = sentence_count >= max_sentences_per_chunk
        if current and (would_exceed_chars or would_exceed_sentences):
            chunks.append(current)
            current = sent
            sentence_count = 1
        else:
            current = f"{current} {sent}".strip() if current else sent
            sentence_count += 1
    if current:
        chunks.append(current)
    return chunks
