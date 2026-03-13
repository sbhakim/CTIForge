"""Data loaders for CTI reports and benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path

from src.schema.graph_schema import CTIDocument
from src.schema.entities import EntityType, CTINEXUS_TYPE_MAP
from src.schema.relations import Triple, ValidationStatus, normalize_relation
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_ctinexus_annotation(path: str | Path) -> tuple[CTIDocument, list[Triple]]:
    """Load a single CTI-Nexus annotation JSON file.

    Returns:
        (CTIDocument, list of gold Triple objects)
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    doc_id = path.stem
    doc = CTIDocument(
        doc_id=doc_id,
        title=doc_id.replace("-", " ").title(),
        source="ctinexus",
        text=data.get("text", ""),
    )

    # Build entity name -> type mapping from annotations
    entity_type_map: dict[str, EntityType] = {}
    for ent in data.get("entities", []):
        etype = CTINEXUS_TYPE_MAP.get(ent.get("entity_type", ""), EntityType.OTHER)
        entity_type_map[ent["entity_name"]] = etype

    # Convert explicit triplets to our Triple schema
    triples = []
    for i, t in enumerate(data.get("explicit_triplets", [])):
        subj = t.get("subject", "")
        obj = t.get("object", "")
        raw_rel = t.get("relation", "related_to")

        subj_type = entity_type_map.get(subj, EntityType.OTHER)
        obj_type = entity_type_map.get(obj, EntityType.OTHER)
        relation = normalize_relation(raw_rel)

        triple = Triple(
            event_id=f"{doc_id}_t{i}",
            subject=subj,
            subject_type=subj_type,
            relation=relation,
            object=obj,
            object_type=obj_type,
            evidence_text="",  # CTI-Nexus does not always provide per-triple evidence
            source_doc_id=doc_id,
            source_chunk_id=f"{doc_id}_chunk0",
            validation_status=ValidationStatus.VALIDATED,  # gold annotations
        )
        triples.append(triple)

    return doc, triples


def load_ctinexus_dataset(
    annotations_dir: str | Path,
) -> tuple[list[CTIDocument], dict[str, list[Triple]]]:
    """Load the full CTI-Nexus annotation dataset.

    Returns:
        (list of CTIDocuments, dict mapping doc_id -> list of gold Triples)
    """
    annotations_dir = Path(annotations_dir)
    docs = []
    gold_triples: dict[str, list[Triple]] = {}

    json_files = sorted(annotations_dir.glob("*.json"))
    logger.info(f"Loading {len(json_files)} CTI-Nexus annotations from {annotations_dir}")

    for fp in json_files:
        try:
            doc, triples = load_ctinexus_annotation(fp)
            docs.append(doc)
            gold_triples[doc.doc_id] = triples
        except Exception as e:
            logger.warning(f"Failed to load {fp.name}: {e}")

    logger.info(
        f"Loaded {len(docs)} documents, "
        f"{sum(len(t) for t in gold_triples.values())} total gold triples"
    )
    return docs, gold_triples


def load_plain_text(path: str | Path, doc_id: str | None = None) -> CTIDocument:
    """Load a plain text CTI report."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return CTIDocument(
        doc_id=doc_id or path.stem,
        title=path.stem.replace("-", " ").replace("_", " ").title(),
        source="plain_text",
        text=text,
    )
