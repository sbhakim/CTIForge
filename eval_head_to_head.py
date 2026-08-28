#!/usr/bin/env python3
"""Head-to-head evaluation: CTIForge vs CTI-Nexus vs AttacKG+ on the SAME data.

All systems process the same CTI documents, and outputs are scored
with the SAME evaluation function, no advantage to any system.

Usage:
    python eval_head_to_head.py --max-docs 20
    python eval_head_to_head.py --max-docs 20 --skip-nexus --skip-attackkg
    python eval_head_to_head.py --max-docs 20 --skip-ctiforge --skip-nexus  # AttacKG+ only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.pipeline import Pipeline, PipelineConfig
from src.ingestion.loaders import load_ctinexus_dataset
from src.schema.relations import Triple, ValidationStatus, RelationType, normalize_relation
from src.schema.entities import EntityType
from eval_ctinexus_decomposed import (
    compute_phase1_triplet_f1,
    compute_phase1_subject_object_f1,
    compute_phase2a_entity_type_f1,
    _normalize,
)
from src.evaluation.per_document import score_per_document, score_pooled
from src.evaluation.schema_neutral import (
    stix_compliance,
    normalization_loss,
    evidence_presence_rate,
)
from src.utils.logging import setup_logger

logger = setup_logger("head_to_head")


# =============================================================================
# Quality metrics (output-only, no gold needed)
# =============================================================================

_GENERIC_RELATIONS = frozenset({"related_to", "associated_with"})

# Semantic constraints: which object types are valid for each relation
_RELATION_OBJECT_CONSTRAINTS: dict[str, set[str]] = {
    "exploits": {"Vulnerability", "Software", "Infrastructure"},
    "attributed_to": {"ThreatActor", "Organization"},
    "communicates_with": {"IOC", "Infrastructure"},
    "variant_of": {"Malware", "Tool", "Technique", "Other"},
    "located_in": {"Location"},
    "mitigated_by": {"Organization", "Software", "Tool", "Other"},
}


def compute_quality_metrics(triples: list[Triple]) -> dict:
    """Compute output quality metrics that don't require gold annotations.

    These measure structural quality of the extracted knowledge graph:
    - Relation specificity: % of triples with specific (non-generic) relations
    - Entity type specificity: % of entities with specific (non-Other) types
    - Semantic validity: % of constrained relations with valid object types
    - Duplicate rate: % of duplicate (subject, object) pairs
    """
    if not triples:
        return {
            "relation_specificity": 0.0,
            "entity_type_specificity": 0.0,
            "semantic_validity": 0.0,
            "duplicate_rate": 0.0,
        }

    # Relation specificity
    specific = sum(1 for t in triples if t.relation.value not in _GENERIC_RELATIONS)
    relation_specificity = specific / len(triples)

    # Entity type specificity (across all subject + object types)
    all_types = [t.subject_type.value for t in triples] + [t.object_type.value for t in triples]
    non_other = sum(1 for ty in all_types if ty != "Other")
    entity_type_specificity = non_other / len(all_types) if all_types else 0.0

    # Semantic validity: among triples with constrained relations, how many
    # have a valid object type?
    constrained_total = 0
    constrained_valid = 0
    for t in triples:
        rel_key = t.relation.value
        if rel_key in _RELATION_OBJECT_CONSTRAINTS:
            constrained_total += 1
            if t.object_type.value in _RELATION_OBJECT_CONSTRAINTS[rel_key]:
                constrained_valid += 1
    semantic_validity = constrained_valid / constrained_total if constrained_total else 1.0

    # Duplicate rate: (subject, object) pairs that appear more than once
    pairs = [(t.subject.lower().strip(), t.object.lower().strip()) for t in triples]
    seen = set()
    duplicates = 0
    for p in pairs:
        if p in seen:
            duplicates += 1
        seen.add(p)
    duplicate_rate = duplicates / len(triples)

    return {
        "relation_specificity": round(relation_specificity, 4),
        "entity_type_specificity": round(entity_type_specificity, 4),
        "semantic_validity": round(semantic_validity, 4),
        "duplicate_rate": round(duplicate_rate, 4),
    }


# ---- AttacKG+ paths ----
ATTACKKG_ROOT = Path(__file__).resolve().parent.parent / "Codes" / "AttacKG-plus"
ATTACKKG_REPORT_EXTRACTOR = ATTACKKG_ROOT / "report_extractor"
ATTACKKG_MITRE_JSON = ATTACKKG_ROOT / "preprocessing" / "TTPScraping" / "mitre.json"


# =============================================================================
# CTI-Nexus runner
# =============================================================================

def run_ctinexus_on_doc(text: str, doc_id: str,
                        nexus_provider: str = "OpenAI",
                        nexus_model: str = "gpt-4o",
                        nexus_embedding: str = "text-embedding-3-large",
                        nexus_ea_model: str = None) -> list[Triple]:
    """Run CTI-Nexus pipeline on a single document and return Triple objects."""
    from ctinexus import process_cti_report
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        out_path = f.name

    try:
        kwargs = dict(
            text=text,
            provider=nexus_provider,
            model=nexus_model,
            embedding_model=nexus_embedding,
            similarity_threshold=0.6,
            output=out_path,
        )
        if nexus_ea_model:
            kwargs["ea_model"] = nexus_ea_model
        result = process_cti_report(**kwargs)
    except Exception as e:
        logger.warning(f"CTI-Nexus failed on {doc_id}: {e}")
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return []

    triples = []
    if result.get("EA") and result["EA"].get("aligned_triplets"):
        raw_triples = result["EA"]["aligned_triplets"]
    elif result.get("ET") and result["ET"].get("typed_triplets"):
        raw_triples = result["ET"]["typed_triplets"]
    elif result.get("IE") and result["IE"].get("triplets"):
        raw_triples = result["IE"]["triplets"]
    else:
        logger.warning(f"CTI-Nexus returned no triples for {doc_id}")
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return []

    for i, rt in enumerate(raw_triples):
        try:
            subj = rt.get("subject", {})
            obj = rt.get("object", {})
            if isinstance(subj, dict):
                subj_name = subj.get("entity_text") or subj.get("mention_text") or subj.get("text", "")
                subj_type_str = subj.get("mention_class") or subj.get("class", "Other")
            else:
                subj_name = str(subj)
                subj_type_str = "Other"

            if isinstance(obj, dict):
                obj_name = obj.get("entity_text") or obj.get("mention_text") or obj.get("text", "")
                obj_type_str = obj.get("mention_class") or obj.get("class", "Other")
            else:
                obj_name = str(obj)
                obj_type_str = "Other"

            raw_rel = rt.get("relation", "related_to")
            relation = normalize_relation(raw_rel)
            subj_type = _map_ctinexus_entity_type(subj_type_str)
            obj_type = _map_ctinexus_entity_type(obj_type_str)

            if not subj_name.strip() or not obj_name.strip():
                continue

            triple = Triple(
                event_id=f"nexus_{doc_id}_t{i}",
                subject=subj_name.strip(),
                subject_type=subj_type,
                relation=relation,
                object=obj_name.strip(),
                object_type=obj_type,
                evidence_text="",
                source_doc_id=doc_id,
                source_chunk_id=f"nexus_{doc_id}",
                validation_status=ValidationStatus.VALIDATED,
            )
            triples.append(triple)
        except Exception as e:
            logger.warning(f"Failed to parse CTI-Nexus triple {i} for {doc_id}: {e}")
            continue

    try:
        os.unlink(out_path)
    except OSError:
        pass

    return triples


def _map_ctinexus_entity_type(type_str: str) -> EntityType:
    key = type_str.lower().strip().replace("-", "_").replace(" ", "_")
    mapping = {
        "malware": EntityType.MALWARE,
        "threat_actor": EntityType.THREAT_ACTOR,
        "threat-actor": EntityType.THREAT_ACTOR,
        "attacker": EntityType.THREAT_ACTOR,
        "tool": EntityType.TOOL,
        "vulnerability": EntityType.VULNERABILITY,
        "organization": EntityType.ORGANIZATION,
        "location": EntityType.LOCATION,
        "campaign": EntityType.CAMPAIGN,
        "technique": EntityType.TECHNIQUE,
        "software": EntityType.SOFTWARE,
        "infrastructure": EntityType.INFRASTRUCTURE,
        "ioc": EntityType.IOC,
        "indicator": EntityType.IOC,
        "indicator:file": EntityType.FILE,
        "indicator:ip": EntityType.IOC,
        "indicator:domain": EntityType.IOC,
        "indicator:hash": EntityType.IOC,
        "indicator:url": EntityType.IOC,
        "indicator:email": EntityType.IOC,
        "file": EntityType.FILE,
    }
    for k, v in mapping.items():
        if k in key:
            return v
    return EntityType.OTHER


# =============================================================================
# AttacKG+ runner
# =============================================================================

def _setup_attackkg():
    """Add AttacKG+ report_extractor to sys.path and load mitre.json."""
    re_dir = str(ATTACKKG_REPORT_EXTRACTOR)
    if re_dir not in sys.path:
        sys.path.insert(0, re_dir)
    with open(ATTACKKG_MITRE_JSON, 'r') as f:
        mitre = json.load(f)
    return mitre


def run_attackkg_on_doc(text: str, doc_id: str, mitre: dict, client) -> list[Triple]:
    """Run AttacKG+ Stage 1 (rewrite) + Stage 2 (extract) on a single document."""
    # Import templates from AttacKG+ (requires cwd to be AttacKG+ root for relative paths)
    orig_cwd = os.getcwd()
    os.chdir(str(ATTACKKG_ROOT))
    try:
        return _run_attackkg_inner(text, doc_id, mitre, client)
    finally:
        os.chdir(orig_cwd)


def _run_attackkg_inner(text: str, doc_id: str, mitre: dict, client) -> list[Triple]:
    """Inner AttacKG+ runner (called with cwd set to AttacKG+ root)."""
    from template import rewriting_template, attack_graph_template

    # --- Stage 1: Rewrite by MITRE tactics ---
    messages, tools = rewriting_template(text, mitre)
    tool_choice = {"type": "function", "function": {"name": tools[0]['function']['name']}}

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0,
        )
        arguments = response.choices[0].message.tool_calls[0].function.arguments
        rewrite_result = json.loads(arguments)
    except Exception as e:
        logger.warning(f"AttacKG+ rewrite failed on {doc_id}: {e}")
        # Fall back to using raw text as "Others" tactic
        rewrite_result = {"Others": text}

    if not isinstance(rewrite_result, dict):
        rewrite_result = {"Others": text}

    # --- Stage 2: Extract triples per tactic ---
    all_triples = []
    for tactic, tactic_text in rewrite_result.items():
        if not tactic_text or tactic_text == "None":
            continue

        bag_text = f"article 0:\n{tactic_text}\n\n"
        messages2, tools2 = attack_graph_template(bag_text)
        tool_choice2 = {"type": "function", "function": {"name": tools2[0]['function']['name']}}

        try:
            response2 = client.chat.completions.create(
                model="gpt-4o",
                messages=messages2,
                tools=tools2,
                tool_choice=tool_choice2,
                temperature=0,
            )
            if response2.choices[0].message.tool_calls is None:
                # Parse text fallback
                text_content = response2.choices[0].message.content or ""
                triplets = _parse_attackkg_text_output(text_content, tactic)
            else:
                arguments2 = response2.choices[0].message.tool_calls[0].function.arguments
                parsed = json.loads(arguments2)
                triplets = parsed.get('triplets', [])
                for t in triplets:
                    t['tactic'] = tactic
        except Exception as e:
            logger.warning(f"AttacKG+ extract failed on {doc_id}/{tactic}: {e}")
            continue

        all_triples.extend(triplets)

    # --- Convert to our Triple format ---
    result = []
    for i, rt in enumerate(all_triples):
        try:
            subj_name = rt.get("Subject", "").strip()
            obj_name = rt.get("Object", "").strip()
            if not subj_name or not obj_name:
                continue

            subj_type = _map_attackkg_entity_type(rt.get("SubjectType", "Others"))
            obj_type = _map_attackkg_entity_type(rt.get("ObjectType", "Others"))
            relation = _map_attackkg_relation(rt.get("Relation", "related_to"))

            triple = Triple(
                event_id=f"attackkg_{doc_id}_t{i}",
                subject=subj_name,
                subject_type=subj_type,
                relation=relation,
                object=obj_name,
                object_type=obj_type,
                evidence_text="",
                source_doc_id=doc_id,
                source_chunk_id=f"attackkg_{doc_id}",
                validation_status=ValidationStatus.VALIDATED,
            )
            result.append(triple)
        except Exception as e:
            logger.warning(f"Failed to parse AttacKG+ triple {i} for {doc_id}: {e}")
            continue

    return result


def _parse_attackkg_text_output(text: str, tactic: str) -> list[dict]:
    """Parse AttacKG+ text output when function calling fails."""
    import re
    triplets = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+\.', '', line).strip()
        line = re.sub(r'^\*', '', line).strip()
        parts = line.split(';')
        if len(parts) < 3:
            continue
        try:
            sub_block = parts[0].strip()
            obj_block = parts[2].strip()
            triplets.append({
                "Subject": sub_block.split('(')[0].strip(),
                "SubjectType": sub_block.split('(')[-1].replace(')', '').strip() if '(' in sub_block else "Others",
                "Relation": parts[1].strip(),
                "Object": obj_block.split('(')[0].strip() if '(' in obj_block else obj_block,
                "ObjectType": obj_block.split('(')[-1].replace(')', '').strip() if '(' in obj_block else "Others",
                "tactic": tactic,
            })
        except Exception:
            continue
    return triplets


def _map_attackkg_entity_type(type_str: str) -> EntityType:
    """Map AttacKG+ STIX-style entity type strings to our EntityType enum."""
    key = type_str.lower().strip().replace("-", "_").replace(" ", "_")
    mapping = {
        "malware": EntityType.MALWARE,
        "threat_actor": EntityType.THREAT_ACTOR,
        "threat-actor": EntityType.THREAT_ACTOR,
        "tool": EntityType.TOOL,
        "vulnerability": EntityType.VULNERABILITY,
        "vulnerablity": EntityType.VULNERABILITY,  # AttacKG+ typo
        "vulnerable_software": EntityType.SOFTWARE,
        "vulnerable software": EntityType.SOFTWARE,
        "organization": EntityType.ORGANIZATION,
        "identity": EntityType.ORGANIZATION,
        "location": EntityType.LOCATION,
        "campaign": EntityType.CAMPAIGN,
        "tactic": EntityType.TECHNIQUE,
        "technique": EntityType.TECHNIQUE,
        "attack_pattern": EntityType.TECHNIQUE,
        "attack-pattern": EntityType.TECHNIQUE,
        "software": EntityType.SOFTWARE,
        "infrastructure": EntityType.INFRASTRUCTURE,
        "infrastrcucture": EntityType.INFRASTRUCTURE,  # AttacKG+ typo
        "domain_name": EntityType.IOC,
        "domain-name": EntityType.IOC,
        "ipv4_addr": EntityType.IOC,
        "ipv4-addr": EntityType.IOC,
        "ipv6_addr": EntityType.IOC,
        "ipv6-addr": EntityType.IOC,
        "url": EntityType.IOC,
        "hash": EntityType.IOC,
        "file": EntityType.FILE,
        "email_address": EntityType.IOC,
        "email-address": EntityType.IOC,
        "registy": EntityType.IOC,  # AttacKG+ typo for registry
        "user_account": EntityType.OTHER,
        "process": EntityType.OTHER,
        "system": EntityType.SOFTWARE,
        "platform": EntityType.SOFTWARE,
    }
    for k, v in mapping.items():
        if k in key:
            return v
    return EntityType.OTHER


def _map_attackkg_relation(rel_str: str) -> RelationType:
    """Map AttacKG+ free-text relations to our 12-type ontology."""
    rel = rel_str.lower().strip()

    # Direct mappings
    direct = {
        "use": RelationType.USES,
        "uses": RelationType.USES,
        "employ": RelationType.USES,
        "leverage": RelationType.USES,
        "utilize": RelationType.USES,
        "operate": RelationType.USES,
        "deploy": RelationType.USES,
        "execute": RelationType.USES,
        "run": RelationType.USES,
        "target": RelationType.TARGETS,
        "targets": RelationType.TARGETS,
        "attack": RelationType.TARGETS,
        "compromise": RelationType.TARGETS,
        "infect": RelationType.TARGETS,
        "impact": RelationType.TARGETS,
        "affect": RelationType.TARGETS,
        "exploit": RelationType.EXPLOITS,
        "exploits": RelationType.EXPLOITS,
        "deliver": RelationType.DELIVERS,
        "delivers": RelationType.DELIVERS,
        "distribute": RelationType.DELIVERS,
        "spread": RelationType.DELIVERS,
        "drop": RelationType.DROPS,
        "drops": RelationType.DROPS,
        "install": RelationType.DROPS,
        "download": RelationType.DROPS,
        "communicate-with": RelationType.COMMUNICATES_WITH,
        "communicate with": RelationType.COMMUNICATES_WITH,
        "connect to": RelationType.COMMUNICATES_WITH,
        "beacon-to": RelationType.COMMUNICATES_WITH,
        "beacon to": RelationType.COMMUNICATES_WITH,
        "exfiltrate-to": RelationType.COMMUNICATES_WITH,
        "resolve-to": RelationType.COMMUNICATES_WITH,
        "mitigate": RelationType.MITIGATED_BY,
        "patch": RelationType.MITIGATED_BY,
        "fix": RelationType.MITIGATED_BY,
        "attribute": RelationType.ATTRIBUTED_TO,
        "attributed_to": RelationType.ATTRIBUTED_TO,
        "belong-to": RelationType.ATTRIBUTED_TO,
        "belong to": RelationType.ATTRIBUTED_TO,
        "originate": RelationType.ATTRIBUTED_TO,
        "locate-at": RelationType.LOCATED_IN,
        "locate at": RelationType.LOCATED_IN,
        "has geolocation": RelationType.LOCATED_IN,
        "variant": RelationType.VARIANT_OF,
        "variant_of": RelationType.VARIANT_OF,
        "is variant of": RelationType.VARIANT_OF,
        "modify": RelationType.ASSOCIATED_WITH,
        "has vulnerability": RelationType.EXPLOITS,
    }

    if rel in direct:
        return direct[rel]

    # Keyword-based fallback
    for keyword, rtype in [
        ("use", RelationType.USES),
        ("employ", RelationType.USES),
        ("leverage", RelationType.USES),
        ("deploy", RelationType.USES),
        ("exploit", RelationType.EXPLOITS),
        ("target", RelationType.TARGETS),
        ("attack", RelationType.TARGETS),
        ("compromis", RelationType.TARGETS),
        ("infect", RelationType.TARGETS),
        ("deliver", RelationType.DELIVERS),
        ("distribut", RelationType.DELIVERS),
        ("drop", RelationType.DROPS),
        ("install", RelationType.DROPS),
        ("download", RelationType.DROPS),
        ("communicat", RelationType.COMMUNICATES_WITH),
        ("connect", RelationType.COMMUNICATES_WITH),
        ("beacon", RelationType.COMMUNICATES_WITH),
        ("exfiltrat", RelationType.COMMUNICATES_WITH),
        ("mitigat", RelationType.MITIGATED_BY),
        ("patch", RelationType.MITIGATED_BY),
        ("attribut", RelationType.ATTRIBUTED_TO),
        ("belong", RelationType.ATTRIBUTED_TO),
        ("locat", RelationType.LOCATED_IN),
        ("variant", RelationType.VARIANT_OF),
    ]:
        if keyword in rel:
            return rtype

    return normalize_relation(rel_str)


# =============================================================================
# Main evaluation
# =============================================================================

def _run_system(name, docs, gold_triples, run_fn, compute_f1):
    """Generic runner for any system. Returns (pred_all_dict, per_doc_list, all_pred, all_gold)."""
    print("\n" + "=" * 70)
    print(f"  Running {name} ...")
    print("=" * 70)
    pred_all = {}
    per_doc = []
    all_pred = []
    all_gold = []
    t0 = time.time()
    last_progress = t0

    for i, doc in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] {doc.doc_id[:50]} ...", end=" ", flush=True)
        triples = run_fn(doc)
        pred_all[doc.doc_id] = triples
        doc_gold = gold_triples.get(doc.doc_id, [])
        all_pred.extend(triples)
        all_gold.extend(doc_gold)
        doc_f1 = compute_f1(triples, doc_gold)
        print(f"pred={len(triples)}, gold={len(doc_gold)}, F1={doc_f1['f1']:.3f}")
        per_doc.append({"doc_id": doc.doc_id, "pred": len(triples), "gold": len(doc_gold), "f1": doc_f1["f1"]})

        now = time.time()
        if i % 5 == 0 or (now - last_progress) >= 300:
            elapsed = now - t0
            avg_per_doc = elapsed / i
            remaining = avg_per_doc * (len(docs) - i)
            running_f1 = compute_f1(all_pred, all_gold)
            print(f"\n  --- {name} Progress: {i}/{len(docs)} docs | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m | "
                  f"Running F1: {running_f1['f1']:.3f} (P={running_f1['precision']:.3f}, R={running_f1['recall']:.3f}) | "
                  f"Total pred: {len(all_pred)} ---\n", flush=True)
            last_progress = now

    elapsed = time.time() - t0
    print(f"  {name} completed in {elapsed/60:.1f}m")
    return pred_all, per_doc, all_pred, all_gold


def main():
    parser = argparse.ArgumentParser(description="Head-to-Head: CTIForge vs CTI-Nexus vs AttacKG+")
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--annotations-dir", default="data/annotations/ctinexus")
    parser.add_argument("-o", "--output-dir", default="output/head_to_head")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-ctiforge", action="store_true", help="Skip CTIForge")
    parser.add_argument("--skip-nexus", action="store_true", help="Skip CTI-Nexus (use cached)")
    parser.add_argument("--skip-attackkg", action="store_true", help="Skip AttacKG+ (use cached)")
    parser.add_argument("--nexus-provider", default="OpenAI", help="CTI-Nexus provider (OpenAI, Local_HF, etc.)")
    parser.add_argument("--nexus-model", default="gpt-4o", help="CTI-Nexus LLM model")
    parser.add_argument("--nexus-embedding", default="text-embedding-3-large", help="CTI-Nexus embedding model")
    parser.add_argument("--nexus-ea-model", default=None, help="CTI-Nexus EA stage embedding model override (e.g. Local_HF/sentence-transformers/all-MiniLM-L6-v2)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading dataset from {args.annotations_dir} ...")
    docs, gold_triples = load_ctinexus_dataset(args.annotations_dir)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"  Loaded {len(docs)} documents\n")

    systems = {}  # name -> {triplet, pair, type, total_predicted, per_doc}

    # --- CTIForge ---
    if not args.skip_ctiforge:
        sg_cache = out / "ctiforge_cached_triples.json"
        config = PipelineConfig(args.config)
        pipeline = Pipeline(config=config)

        def run_sg(doc):
            _, triples = pipeline.process_document(doc)
            return [t for t in triples if t.validation_status in (ValidationStatus.VALIDATED, ValidationStatus.REPAIRED)]

        sg_pred_all, sg_per_doc, sg_pred, sg_gold = _run_system(
            "CTIForge", docs, gold_triples, run_sg, compute_phase1_triplet_f1
        )
        # Cache
        cache_data = {did: [t.model_dump() for t in ts] for did, ts in sg_pred_all.items()}
        with open(sg_cache, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        # Persist the validator's error taxonomy. Without this the manuscript's
        # taxonomy claim has no artifact behind it for benchmark-scale runs --
        # error_logger.save() was previously only reached from main.py / cli.py,
        # never from the scripts that produce the paper's tables.
        pipeline.error_logger.save(out / "error_taxonomy")
    else:
        sg_cache = out / "ctiforge_cached_triples.json"
        if sg_cache.exists():
            print("  Using cached CTIForge results")
            cached = json.load(open(sg_cache))
            sg_pred_all = {did: [Triple(**td) for td in tds] for did, tds in cached.items()}
            sg_pred = []
            sg_gold = []
            sg_per_doc = []
            for doc in docs:
                doc_triples = sg_pred_all.get(doc.doc_id, [])
                doc_gold = gold_triples.get(doc.doc_id, [])
                sg_pred.extend(doc_triples)
                sg_gold.extend(doc_gold)
                doc_f1 = compute_phase1_triplet_f1(doc_triples, doc_gold)
                sg_per_doc.append({"doc_id": doc.doc_id, "pred": len(doc_triples), "gold": len(doc_gold), "f1": doc_f1["f1"]})
        else:
            sg_pred, sg_gold, sg_per_doc = [], [], []
            sg_pred_all = {}

    # --- CTI-Nexus ---
    nexus_cache = out / "nexus_cached_triples.json"
    if not args.skip_nexus:
        def run_nx(doc):
            return run_ctinexus_on_doc(doc.text, doc.doc_id,
                                       nexus_provider=args.nexus_provider,
                                       nexus_model=args.nexus_model,
                                       nexus_embedding=args.nexus_embedding,
                                       nexus_ea_model=args.nexus_ea_model)

        nx_pred_all, nx_per_doc, nx_pred, nx_gold = _run_system(
            "CTI-Nexus", docs, gold_triples, run_nx, compute_phase1_triplet_f1
        )
        cache_data = {did: [t.model_dump() for t in ts] for did, ts in nx_pred_all.items()}
        with open(nexus_cache, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    else:
        if nexus_cache.exists():
            print("  Using cached CTI-Nexus results")
            cached = json.load(open(nexus_cache))
            nx_pred_all = {did: [Triple(**td) for td in tds] for did, tds in cached.items()}
            nx_pred = []
            nx_gold = []
            nx_per_doc = []
            for doc in docs:
                doc_triples = nx_pred_all.get(doc.doc_id, [])
                doc_gold = gold_triples.get(doc.doc_id, [])
                nx_pred.extend(doc_triples)
                nx_gold.extend(doc_gold)
                doc_f1 = compute_phase1_triplet_f1(doc_triples, doc_gold)
                nx_per_doc.append({"doc_id": doc.doc_id, "pred": len(doc_triples), "gold": len(doc_gold), "f1": doc_f1["f1"]})
        else:
            nx_pred, nx_gold, nx_per_doc = [], [], []
            nx_pred_all = {}

    # --- AttacKG+ ---
    akg_cache = out / "attackkg_cached_triples.json"
    if not args.skip_attackkg:
        from openai import OpenAI
        akg_client = OpenAI()
        mitre = _setup_attackkg()

        def run_akg(doc):
            return run_attackkg_on_doc(doc.text, doc.doc_id, mitre, akg_client)

        akg_pred_all, akg_per_doc, akg_pred, akg_gold = _run_system(
            "AttacKG+", docs, gold_triples, run_akg, compute_phase1_triplet_f1
        )
        cache_data = {did: [t.model_dump() for t in ts] for did, ts in akg_pred_all.items()}
        with open(akg_cache, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    else:
        if akg_cache.exists():
            print("  Using cached AttacKG+ results")
            cached = json.load(open(akg_cache))
            akg_pred_all = {did: [Triple(**td) for td in tds] for did, tds in cached.items()}
            akg_pred = []
            akg_gold = []
            akg_per_doc = []
            for doc in docs:
                doc_triples = akg_pred_all.get(doc.doc_id, [])
                doc_gold = gold_triples.get(doc.doc_id, [])
                akg_pred.extend(doc_triples)
                akg_gold.extend(doc_gold)
                doc_f1 = compute_phase1_triplet_f1(doc_triples, doc_gold)
                akg_per_doc.append({"doc_id": doc.doc_id, "pred": len(doc_triples), "gold": len(doc_gold), "f1": doc_f1["f1"]})
        else:
            akg_pred, akg_gold, akg_per_doc = [], [], []
            akg_pred_all = {}

    # --- Compute aggregate metrics ---
    all_systems = {}

    if sg_pred:
        # Triplet / pair matching is scoped per document: pooling every document's
        # predictions and gold into flat lists lets a prediction from document A be
        # credited against a gold triple from document B. The pooled values are kept
        # alongside so the difference stays visible.
        sg_triplet = score_per_document(sg_pred_all, gold_triples, compute_phase1_triplet_f1)
        sg_pair = score_per_document(sg_pred_all, gold_triples, compute_phase1_subject_object_f1)
        sg_triplet_pooled = score_pooled(sg_pred_all, gold_triples, compute_phase1_triplet_f1)
        sg_pair_pooled = score_pooled(sg_pred_all, gold_triples, compute_phase1_subject_object_f1)
        # Entity typing is a corpus-level metric (name -> type map); left pooled.
        sg_type = compute_phase2a_entity_type_f1(sg_pred, sg_gold)
        sg_quality = compute_quality_metrics(sg_pred)
        # Schema-neutral: scored against STIX 2.1, an external standard that no
        # compared system enforces. Safe for cross-system comparison, unlike the
        # quality block above which uses CTIForge's own constraint table.
        sg_neutral = {
            "stix": stix_compliance(sg_pred),
            "normalization": normalization_loss(sg_pred),
            "evidence_presence": evidence_presence_rate(sg_pred),
        }
        all_systems["CTIForge"] = {"triplet": sg_triplet, "pair": sg_pair, "type": sg_type,
                                       "triplet_pooled": sg_triplet_pooled, "pair_pooled": sg_pair_pooled,
                                       "quality": sg_quality, "neutral": sg_neutral,
                                       "total_predicted": len(sg_pred), "per_doc": sg_per_doc, "abbr": "SG"}

    if nx_pred:
        nx_triplet = score_per_document(nx_pred_all, gold_triples, compute_phase1_triplet_f1)
        nx_pair = score_per_document(nx_pred_all, gold_triples, compute_phase1_subject_object_f1)
        nx_triplet_pooled = score_pooled(nx_pred_all, gold_triples, compute_phase1_triplet_f1)
        nx_pair_pooled = score_pooled(nx_pred_all, gold_triples, compute_phase1_subject_object_f1)
        nx_type = compute_phase2a_entity_type_f1(nx_pred, nx_gold)
        nx_quality = compute_quality_metrics(nx_pred)
        # Schema-neutral: scored against STIX 2.1, an external standard that no
        # compared system enforces. Safe for cross-system comparison, unlike the
        # quality block above which uses CTIForge's own constraint table.
        nx_neutral = {
            "stix": stix_compliance(nx_pred),
            "normalization": normalization_loss(nx_pred),
            "evidence_presence": evidence_presence_rate(nx_pred),
        }
        all_systems["CTI-Nexus"] = {"triplet": nx_triplet, "pair": nx_pair, "type": nx_type,
                                    "triplet_pooled": nx_triplet_pooled, "pair_pooled": nx_pair_pooled,
                                    "quality": nx_quality, "neutral": nx_neutral,
                                    "total_predicted": len(nx_pred), "per_doc": nx_per_doc, "abbr": "NX"}

    if akg_pred:
        akg_triplet = score_per_document(akg_pred_all, gold_triples, compute_phase1_triplet_f1)
        akg_pair = score_per_document(akg_pred_all, gold_triples, compute_phase1_subject_object_f1)
        akg_triplet_pooled = score_pooled(akg_pred_all, gold_triples, compute_phase1_triplet_f1)
        akg_pair_pooled = score_pooled(akg_pred_all, gold_triples, compute_phase1_subject_object_f1)
        akg_type = compute_phase2a_entity_type_f1(akg_pred, akg_gold)
        akg_quality = compute_quality_metrics(akg_pred)
        # Schema-neutral: scored against STIX 2.1, an external standard that no
        # compared system enforces. Safe for cross-system comparison, unlike the
        # quality block above which uses CTIForge's own constraint table.
        akg_neutral = {
            "stix": stix_compliance(akg_pred),
            "normalization": normalization_loss(akg_pred),
            "evidence_presence": evidence_presence_rate(akg_pred),
        }
        all_systems["AttacKG+"] = {"triplet": akg_triplet, "pair": akg_pair, "type": akg_type,
                                   "triplet_pooled": akg_triplet_pooled, "pair_pooled": akg_pair_pooled,
                                   "quality": akg_quality, "neutral": akg_neutral,
                                   "total_predicted": len(akg_pred), "per_doc": akg_per_doc, "abbr": "AK"}

    # --- Print comparison table ---
    sys_names = list(all_systems.keys())
    if not sys_names:
        print("\n  No systems to compare!")
        return

    print("\n" + "=" * 80)
    print(f"  HEAD-TO-HEAD COMPARISON, {len(docs)} documents")
    print(f"  Same gold data, same evaluation function")
    print("=" * 80)

    header = f"  {'Metric':<35s}"
    for name in sys_names:
        header += f" {name:>14s}"
    print(f"\n{header}")
    print(f"  {'-' * (35 + 15 * len(sys_names))}")

    metrics = [
        ("Triplet F1", lambda s: s["triplet"]["f1"]),
        ("Triplet Precision", lambda s: s["triplet"]["precision"]),
        ("Triplet Recall", lambda s: s["triplet"]["recall"]),
        ("  [legacy pooled Triplet F1]", lambda s: s.get("triplet_pooled", {}).get("f1", 0.0)),
        ("S-O Pair F1", lambda s: s["pair"]["f1"]),
        ("S-O Pair Precision", lambda s: s["pair"]["precision"]),
        ("S-O Pair Recall", lambda s: s["pair"]["recall"]),
        ("Entity Type Accuracy", lambda s: s["type"]["accuracy"]),
        ("Entity Type Micro-F1", lambda s: s["type"]["micro_f1"]),
        ("--- Schema-Neutral (comparable) ---", None),
        ("Evidence Presence", lambda s: s["neutral"]["evidence_presence"]),
        ("STIX 2.1 Compliance", lambda s: s["neutral"]["stix"]["stix_compliance"]),
        ("  STIX coverage of output", lambda s: s["neutral"]["stix"]["coverage"]),
        ("Duplicate Rate", lambda s: s["quality"]["duplicate_rate"]),
        ("--- Normalization Artifacts ---", None),
        ("Generic Relation Rate", lambda s: s["neutral"]["normalization"]["generic_relation_rate"]),
        ("'Other' Entity Type Rate", lambda s: s["neutral"]["normalization"]["other_type_rate"]),
        ("--- CTIForge-Schema (DEFINITIONAL) ---", None),
        ("Relation Specificity [SG schema]", lambda s: s["quality"]["relation_specificity"]),
        ("Entity Type Specificity [SG schema]", lambda s: s["quality"]["entity_type_specificity"]),
        ("Semantic Validity [SG schema]", lambda s: s["quality"]["semantic_validity"]),
        ("---", None),
        ("Total Predicted Triples", lambda s: s["total_predicted"]),
    ]

    for label, getter in metrics:
        if getter is None:
            # Section separator
            if label != "---":
                print(f"\n  {label}")
            continue
        row = f"  {label:<35s}"
        for name in sys_names:
            val = getter(all_systems[name])
            if isinstance(val, int):
                row += f" {val:>14d}"
            else:
                row += f" {val:>14.4f}"
        print(row)

    total_gold = len(gold_triples.get(docs[0].doc_id, [])) if docs else 0
    gold_count = sum(len(gold_triples.get(d.doc_id, [])) for d in docs)
    row = f"  {'Total Gold Triples':<35s}"
    for _ in sys_names:
        row += f" {gold_count:>14d}"
    print(row)

    # --- Per-document comparison ---
    print(f"\n  {'Document':<40s}", end="")
    for name in sys_names:
        abbr = all_systems[name]["abbr"]
        print(f" {abbr+' F1':>7s}", end="")
    print(f" {'Winner':>8s}")
    print(f"  {'-' * (40 + 8 * len(sys_names) + 8)}")

    win_counts = {name: 0 for name in sys_names}
    ties = 0

    for doc_idx in range(len(docs)):
        doc_id = docs[doc_idx].doc_id
        row = f"  {doc_id[:40]:<40s}"
        f1s = {}
        for name in sys_names:
            per_doc_list = all_systems[name]["per_doc"]
            if doc_idx < len(per_doc_list):
                f1 = per_doc_list[doc_idx]["f1"]
            else:
                f1 = 0.0
            f1s[name] = f1
            row += f" {f1:>7.3f}"

        # Determine winner
        best_f1 = max(f1s.values())
        winners = [n for n, f in f1s.items() if f >= best_f1 - 0.01]
        if len(winners) == 1 and best_f1 > 0:
            winner = all_systems[winners[0]]["abbr"]
            win_counts[winners[0]] += 1
        else:
            winner = "tie"
            ties += 1
        row += f" {winner:>8s}"
        print(row)

    print(f"\n  Win count: ", end="")
    print(", ".join(f"{all_systems[n]['abbr']}={win_counts[n]}" for n in sys_names) + f", Ties={ties}")
    print("=" * 80)

    # Save results
    results = {"num_documents": len(docs)}
    for name in sys_names:
        s = all_systems[name]
        results[name] = {
            "triplet": s["triplet"], "pair": s["pair"], "type": s["type"],
            "triplet_pooled": s.get("triplet_pooled"), "pair_pooled": s.get("pair_pooled"),
            "quality": s["quality"], "neutral": s.get("neutral"),
            "total_predicted": s["total_predicted"], "per_doc": s["per_doc"],
        }
    with open(out / "head_to_head_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}/head_to_head_results.json\n")


if __name__ == "__main__":
    main()
