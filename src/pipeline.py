"""End-to-end pipeline orchestrator for CTIForge.

Runs the five-module pipeline:
  Module A: Ingestion + Segmentation
  Module B: Neural Extraction
  Module C: Symbolic Validation
  Module D: Canonicalization + Grounding
  Module E: Cross-Document Fusion
"""

from __future__ import annotations

from pathlib import Path

from src.schema.entities import EntityType
from src.schema.graph_schema import CTIDocument, CTIGraph
from src.schema.relations import Triple, ValidationStatus
from src.ingestion.segmenter import segment_document
from src.extraction.extractor import Extractor
from src.extraction.example_retriever import FewShotExampleRetriever
from src.symbolic.validator import SymbolicValidator
from src.symbolic.error_logger import ErrorTaxonomyLogger
from src.symbolic.relation_repair import RelationRepairer
from src.grounding.canonicalizer import Canonicalizer
from src.grounding.alias_tables import AliasTable, load_mitre_aliases
from src.grounding.attack_mapping import AttackMapper
from src.extraction.link_predictor import LinkPredictor
from src.extraction.guided_reextractor import GuidedReextractor
from src.symbolic.confidence_rescorer import ConfidenceRescorer
from src.grounding.entity_alignment import EntityAligner
from src.symbolic.saliency_filter import SaliencyFilter
from src.graph.builder import build_graph
from src.fusion.merger import GraphMerger
from src.fusion.provenance import ProvenanceTracker
from src.utils.config import load_yaml_config, get_project_root
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class PipelineConfig:
    """Configuration for a pipeline run."""

    def __init__(self, config_path: str | Path = "configs/default.yaml"):
        raw = load_yaml_config(config_path)
        llm_cfg = raw.get("llm", {})
        embedding_cfg = raw.get("embedding", {})

        self.model_provider: str = llm_cfg.get("provider", "openai")
        self.model: str = self._resolve_named_model(
            provider=self.model_provider,
            cfg=llm_cfg,
            fallback="gpt-4o",
        )

        self.embedding_provider: str = embedding_cfg.get("provider", "local")
        self.embedding_model: str = self._resolve_named_model(
            provider=self.embedding_provider,
            cfg=embedding_cfg,
            fallback="sentence-transformers/all-MiniLM-L6-v2",
        )

        self.temperature: float = llm_cfg.get("temperature", 0.1)
        self.max_tokens: int = llm_cfg.get("max_tokens", 4096)
        self.retry_attempts: int = llm_cfg.get("retry_attempts", 1)
        self.cloud_timeout: int = llm_cfg.get("timeout", 120)
        self.max_rpm: float | None = llm_cfg.get("max_rpm")
        structured_output_cfg = llm_cfg.get("structured_output", {})
        self.cloud_response_format: dict | None = None
        if structured_output_cfg.get("enabled", False):
            format_type = structured_output_cfg.get("type", "json_object")
            self.cloud_response_format = {"type": format_type}
        local_llm_cfg = llm_cfg.get("local", {})
        self.local_system_prompt: str = local_llm_cfg.get(
            "system_prompt",
            "You are a CTI extraction assistant.",
        )
        self.local_do_sample: bool = local_llm_cfg.get("do_sample", False)
        self.local_max_new_tokens: int = local_llm_cfg.get("max_new_tokens", 1280)
        self.local_prompt_template: str | None = local_llm_cfg.get("prompt_template")
        self.local_max_triples: int = local_llm_cfg.get("max_triples", 10)
        self.max_chunk_chars: int = raw.get("extraction", {}).get("max_chunk_chars", 2000)
        self.overlap_chars: int = raw.get("extraction", {}).get("overlap_chars", 200)
        self.max_sentences_per_chunk: int = raw.get("extraction", {}).get(
            "max_sentences_per_chunk", 0
        )
        self.few_shot_examples: int = raw.get("extraction", {}).get("few_shot_examples", 0)
        self.max_chars_per_example: int = raw.get("extraction", {}).get(
            "max_chars_per_example", 1200
        )
        self.max_triples_per_example: int = raw.get("extraction", {}).get(
            "max_triples_per_example", 6
        )
        first_pass_cfg = raw.get("extraction", {}).get("first_pass", {})
        self.first_pass_temperature: float | None = first_pass_cfg.get("temperature")
        self.first_pass_max_tokens: int | None = first_pass_cfg.get("max_tokens")
        compact_cfg = raw.get("extraction", {}).get("compact_mode", {})
        self.enable_compact_extraction: bool = compact_cfg.get("enabled", False)
        self.compact_min_chunk_chars: int = compact_cfg.get("min_chunk_chars", 900)
        self.compact_max_triples: int = compact_cfg.get("max_triples", 8)
        self.compact_temperature: float = compact_cfg.get("temperature", 0.0)
        self.compact_max_tokens: int = compact_cfg.get("max_tokens", 1800)
        self.max_triples_per_chunk: int = raw.get("validation", {}).get("max_triples_per_chunk", 20)
        self.reject_missing_evidence: bool = raw.get("validation", {}).get(
            "reject_missing_evidence", False
        )
        self.min_entity_name_length: int = raw.get("validation", {}).get(
            "min_entity_name_length", 1
        )
        self.reject_unsupported_by_evidence: bool = raw.get("validation", {}).get(
            "reject_unsupported_by_evidence", True
        )
        self.min_confidence_threshold: float = raw.get("validation", {}).get(
            "min_confidence_threshold", 0.0
        )
        self.similarity_threshold: float = raw.get("canonicalization", {}).get(
            "similarity_threshold", 0.85
        )
        self.min_confidence: float = raw.get("fusion", {}).get("min_confidence_for_merge", 0.5)
        lp_cfg = raw.get("link_prediction", {})
        self.enable_link_prediction: bool = lp_cfg.get("enabled", True)
        self.max_lp_predictions: int = lp_cfg.get("max_predictions", 5)
        # Optional recovery stages are enabled only by explicit configuration.
        self.lp_skip_if_recovery_share_at_least: float | None = lp_cfg.get(
            "skip_if_recovery_share_at_least"
        )
        self.enable_entity_alignment: bool = raw.get("entity_alignment", {}).get("enabled", True)
        self.enable_supplement_extraction: bool = raw.get("supplement_extraction", {}).get("enabled", True)
        low_yield_cfg = raw.get("low_yield_recovery", {})
        self.enable_low_yield_recovery: bool = low_yield_cfg.get("enabled", False)
        self.low_yield_min_triples_per_chunk: int = low_yield_cfg.get("min_triples_per_chunk", 3)
        self.low_yield_max_new_triples: int = low_yield_cfg.get("max_new_triples", 8)
        doc_fallback_cfg = raw.get("document_fallback", {})
        self.enable_document_fallback: bool = doc_fallback_cfg.get("enabled", False)
        self.document_fallback_single_chunk_only: bool = doc_fallback_cfg.get(
            "single_chunk_only", True
        )
        self.document_fallback_max_raw_triples: int = doc_fallback_cfg.get(
            "max_raw_triples", 2
        )
        self.document_fallback_max_validated_triples: int = doc_fallback_cfg.get(
            "max_validated_triples", 1
        )
        self.document_fallback_max_new_triples: int = doc_fallback_cfg.get(
            "max_new_triples", 8
        )
        focus_recovery_cfg = raw.get("focused_recovery", {})
        self.enable_focused_recovery: bool = focus_recovery_cfg.get("enabled", False)
        self.focused_recovery_single_chunk_only: bool = focus_recovery_cfg.get(
            "single_chunk_only", True
        )
        self.focused_recovery_max_raw_triples: int = focus_recovery_cfg.get(
            "max_raw_triples", 2
        )
        self.focused_recovery_max_validated_triples: int = focus_recovery_cfg.get(
            "max_validated_triples", 1
        )
        self.focused_recovery_max_sentences: int = focus_recovery_cfg.get(
            "max_sentences", 6
        )
        self.focused_recovery_max_new_triples: int = focus_recovery_cfg.get(
            "max_new_triples", 8
        )
        self.enable_guided_reextraction: bool = raw.get("guided_reextraction", {}).get("enabled", True)
        self.enable_confidence_rescoring: bool = raw.get("confidence_rescoring", {}).get("enabled", True)
        self.confidence_prune_threshold: float = raw.get("confidence_rescoring", {}).get("prune_threshold", 0.35)
        saliency_cfg = raw.get("saliency_filter", {})
        self.enable_saliency_filter: bool = saliency_cfg.get("enabled", False)
        self.max_triples_per_doc: int = saliency_cfg.get("max_triples_per_doc", 20)
        self.promote_associated_with: bool = raw.get("relation_repair", {}).get(
            "promote_associated_with", False
        )
        self.require_relation_repair_entity_mentions: bool = raw.get("relation_repair", {}).get(
            "require_entity_mentions", True
        )
        self.annotations_path: Path = get_project_root() / raw.get("paths", {}).get(
            "annotations", "data/annotations/ctinexus"
        )
        self.mitre_path: Path = get_project_root() / raw.get("paths", {}).get(
            "external", "data/external"
        ) / "mitre.jsonl"

    @staticmethod
    def _resolve_named_model(provider: str, cfg: dict, fallback: str) -> str:
        """Resolve the active model from either `model` or `models[provider]`."""
        models = cfg.get("models", {})
        if isinstance(models, dict) and provider in models:
            return models[provider]
        return cfg.get("model", fallback)


class Pipeline:
    """Orchestrates the full CTIForge pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        enable_extraction: bool = True,
        enable_validation: bool = True,
        enable_canonicalization: bool = True,
        enable_fusion: bool = True,
        enable_link_prediction: bool | None = None,
        enable_entity_alignment: bool | None = None,
        enable_supplement_extraction: bool | None = None,
    ):
        self.config = config or PipelineConfig()
        self.enable_extraction = enable_extraction
        self.enable_validation = enable_validation
        self.enable_canonicalization = enable_canonicalization
        self.enable_fusion = enable_fusion
        self.enable_link_prediction = (
            enable_link_prediction if enable_link_prediction is not None
            else self.config.enable_link_prediction
        )
        self.enable_entity_alignment = (
            enable_entity_alignment if enable_entity_alignment is not None
            else self.config.enable_entity_alignment
        )
        self.enable_supplement_extraction = (
            enable_supplement_extraction if enable_supplement_extraction is not None
            else self.config.enable_supplement_extraction
        )

        # Initialize components
        self.error_logger = ErrorTaxonomyLogger()
        self.provenance = ProvenanceTracker()

        # Module B: Extractor
        if self.enable_extraction:
            self.extractor = Extractor(
                provider=self.config.model_provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                cloud_response_format=self.config.cloud_response_format,
                retry_attempts=self.config.retry_attempts,
                max_triples_per_chunk=self.config.max_triples_per_chunk,
                local_system_prompt=self.config.local_system_prompt,
                local_do_sample=self.config.local_do_sample,
                local_max_new_tokens=self.config.local_max_new_tokens,
                local_prompt_template=self.config.local_prompt_template,
                local_max_triples=self.config.local_max_triples,
                cloud_timeout=self.config.cloud_timeout,
                max_rpm=self.config.max_rpm,
            )
        else:
            self.extractor = None
        if self.config.few_shot_examples > 0 and self.config.annotations_path.exists():
            self.example_retriever = FewShotExampleRetriever(
                self.config.annotations_path,
                max_examples=self.config.few_shot_examples,
                max_chars_per_example=self.config.max_chars_per_example,
                max_triples_per_example=self.config.max_triples_per_example,
            )
        else:
            self.example_retriever = None

        # Module C: Validator
        self.validator = SymbolicValidator(
            error_logger=self.error_logger,
            reject_missing_evidence=self.config.reject_missing_evidence,
            min_entity_name_length=self.config.min_entity_name_length,
            reject_unsupported_by_evidence=self.config.reject_unsupported_by_evidence,
            min_confidence_threshold=self.config.min_confidence_threshold,
        )
        self.relation_repairer = RelationRepairer(
            promote_associated_with=self.config.promote_associated_with,
            require_entity_mentions=self.config.require_relation_repair_entity_mentions,
        )

        # Module D: Canonicalizer
        group_aliases, software_aliases = self._load_aliases()
        self.canonicalizer = Canonicalizer(
            group_aliases=group_aliases,
            software_aliases=software_aliases,
            similarity_threshold=self.config.similarity_threshold,
            error_logger=self.error_logger,
        )
        self.attack_mapper = AttackMapper(self.config.mitre_path)

        # Module E: Merger
        self.merger = GraphMerger(min_confidence=self.config.min_confidence)

        # Link prediction
        if self.enable_link_prediction and self.enable_extraction:
            self.link_predictor = LinkPredictor(
                provider=self.config.model_provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                max_predictions=self.config.max_lp_predictions,
            )
        else:
            self.link_predictor = None

        # Entity alignment
        if self.enable_entity_alignment:
            self.entity_aligner = EntityAligner(
                similarity_threshold=self.config.similarity_threshold,
                embedding_model=self.config.embedding_model,
            )
        else:
            self.entity_aligner = None

        # Guided Re-extraction (neuro-symbolic recall booster)
        if self.config.enable_guided_reextraction and self.enable_extraction:
            self.guided_reextractor = GuidedReextractor(
                provider=self.config.model_provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            self.guided_reextractor = None

        # Confidence Re-scorer (neuro-symbolic precision filter)
        if self.config.enable_confidence_rescoring:
            self.confidence_rescorer = ConfidenceRescorer(
                prune_threshold=self.config.confidence_prune_threshold,
            )
        else:
            self.confidence_rescorer = None

        # Saliency Filter (neuro-symbolic precision booster)
        if self.config.enable_saliency_filter:
            self.saliency_filter = SaliencyFilter(
                max_triples_per_doc=self.config.max_triples_per_doc,
            )
        else:
            self.saliency_filter = None

    def _load_aliases(self) -> tuple[AliasTable, AliasTable]:
        """Load alias tables from MITRE data if available."""
        mitre_path = self.config.mitre_path
        if mitre_path.exists():
            return load_mitre_aliases(mitre_path)
        logger.warning(f"MITRE data not found at {mitre_path}, using empty alias tables")
        return AliasTable(), AliasTable()

    def _should_use_compact_extraction(self, chunk: object) -> bool:
        """Use a tighter prompt only on long Mistral-style chunks."""
        if not self.config.enable_compact_extraction:
            return False
        if "mistral" not in self.config.model.lower():
            return False
        text = getattr(chunk, "text", "")
        return isinstance(text, str) and len(text) >= self.config.compact_min_chunk_chars

    def _count_preview_valid_triples(self, triples: list[Triple]) -> int:
        """Estimate how many triples would survive repair + validation."""
        preview = [triple.model_copy() for triple in triples]
        preview = self.relation_repairer.repair_triples(preview)
        preview = self.validator.validate_triples(preview)
        return sum(1 for triple in preview if triple.validation_status != ValidationStatus.REJECTED)

    def _should_run_document_fallback(
        self,
        doc: CTIDocument,
        raw_triples: list[Triple],
    ) -> bool:
        """Trigger the document fallback only on true low-yield cases."""
        if not self.config.enable_document_fallback or not self.extractor:
            return False
        if self.config.document_fallback_single_chunk_only and len(doc.chunks) != 1:
            return False
        if len(raw_triples) > self.config.document_fallback_max_raw_triples:
            return False

        preview_valid = self._count_preview_valid_triples(raw_triples)
        if preview_valid > self.config.document_fallback_max_validated_triples:
            return False

        logger.info(
            "Document fallback candidate: %s (%d raw, %d preview-valid triples)",
            doc.doc_id,
            len(raw_triples),
            preview_valid,
        )
        return True

    def _should_run_focused_recovery(
        self,
        doc: CTIDocument,
        raw_triples: list[Triple],
    ) -> bool:
        """Trigger focus-text recovery only on true low-yield cases."""
        if not self.config.enable_focused_recovery or not self.extractor:
            return False
        if self.config.focused_recovery_single_chunk_only and len(doc.chunks) != 1:
            return False
        if len(raw_triples) > self.config.focused_recovery_max_raw_triples:
            return False

        preview_valid = self._count_preview_valid_triples(raw_triples)
        if preview_valid > self.config.focused_recovery_max_validated_triples:
            return False

        logger.info(
            "Focused recovery candidate: %s (%d raw, %d preview-valid triples)",
            doc.doc_id,
            len(raw_triples),
            preview_valid,
        )
        return True

    def extract_document(self, doc: CTIDocument) -> tuple[CTIDocument, list[Triple]]:
        """Modules A and B plus link prediction: every stage that calls the LLM.

        Split out of process_document so that ablation arms can share a single
        extraction pass. Run separately, arms would differ by both the module
        under test and a fresh sampling of the model, and that resampling noise
        is the same size as the effect being measured.
        """
        """Run the pipeline on a single document.

        Returns:
            (CTIGraph, list of all triples including rejected)
        """
        logger.info(f"Processing document: {doc.doc_id} ({len(doc.text)} chars)")

        # Module A: Segment
        doc = segment_document(
            doc,
            max_chunk_chars=self.config.max_chunk_chars,
            overlap_chars=self.config.overlap_chars,
            max_sentences_per_chunk=self.config.max_sentences_per_chunk,
        )

        if not doc.chunks:
            logger.warning(f"No chunks produced for {doc.doc_id}")
            return doc, []

        # Module B: Extract
        if self.extractor:
            raw_triples = []
            for chunk in doc.chunks:
                examples = []
                if self.example_retriever:
                    examples = self.example_retriever.retrieve(
                        chunk.text,
                        exclude_doc_id=doc.doc_id,
                    )
                compact_mode = self._should_use_compact_extraction(chunk)
                raw_triples.extend(
                    self.extractor.extract_from_chunk(
                        chunk,
                        examples,
                        compact_mode=compact_mode,
                        compact_max_triples=self.config.compact_max_triples,
                        override_temperature=(
                            self.config.compact_temperature
                            if compact_mode
                            else self.config.first_pass_temperature
                        ),
                        override_max_tokens=(
                            self.config.compact_max_tokens
                            if compact_mode
                            else self.config.first_pass_max_tokens
                        ),
                    )
                )
        else:
            raw_triples = []
        primary_raw_count = len(raw_triples)
        recovered_raw_count = 0

        # Supplement extraction: run a second pass to catch missed triples
        if self.config.enable_low_yield_recovery and self.extractor:
            existing_keys = {
                (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                for t in raw_triples
            }
            for chunk in doc.chunks:
                chunk_triples = [
                    t for t in raw_triples
                    if t.source_chunk_id == chunk.chunk_id
                ]
                if len(chunk_triples) >= self.config.low_yield_min_triples_per_chunk:
                    continue

                recovered = self.extractor.recover_low_yield_chunk(
                    chunk=chunk,
                    existing_triples=chunk_triples,
                    max_recovery_triples=self.config.low_yield_max_new_triples,
                )
                if not recovered:
                    continue

                new_recovered = []
                for t in recovered:
                    key = (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                    key_swap = (t.object.lower().strip(), t.relation.value, t.subject.lower().strip())
                    if key not in existing_keys and key_swap not in existing_keys:
                        new_recovered.append(t)
                        existing_keys.add(key)
                if new_recovered:
                    recovered_raw_count += len(new_recovered)
                    logger.info(
                        "Low-yield recovery: +%d triples for chunk %s",
                        len(new_recovered),
                        chunk.chunk_id,
                    )
                    raw_triples.extend(new_recovered)

        if self._should_run_focused_recovery(doc, raw_triples):
            existing_keys = {
                (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                for t in raw_triples
            }
            focus_chunk = doc.chunks[0]
            focus_text = self.extractor.build_focus_text(
                focus_chunk.text,
                max_sentences=self.config.focused_recovery_max_sentences,
            )
            if focus_text and len(focus_text) < len(focus_chunk.text):
                focused_recovered = self.extractor.recover_from_focus_text(
                    chunk=focus_chunk,
                    focus_text=focus_text,
                    existing_triples=raw_triples,
                    max_recovery_triples=self.config.focused_recovery_max_new_triples,
                )
                if focused_recovered:
                    new_focused = []
                    for t in focused_recovered:
                        key = (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                        key_swap = (t.object.lower().strip(), t.relation.value, t.subject.lower().strip())
                        if key not in existing_keys and key_swap not in existing_keys:
                            new_focused.append(t)
                            existing_keys.add(key)
                    if new_focused:
                        logger.info(
                            "Focused recovery: +%d triples for %s",
                            len(new_focused),
                            doc.doc_id,
                        )
                        raw_triples.extend(new_focused)

        if self._should_run_document_fallback(doc, raw_triples):
            existing_keys = {
                (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                for t in raw_triples
            }
            fallback_chunk = doc.chunks[0]
            fallback_triples = self.extractor.recover_low_yield_document(
                chunk=fallback_chunk,
                existing_triples=raw_triples,
                max_fallback_triples=self.config.document_fallback_max_new_triples,
            )
            if fallback_triples:
                new_fallback = []
                for t in fallback_triples:
                    key = (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                    key_swap = (t.object.lower().strip(), t.relation.value, t.subject.lower().strip())
                    if key not in existing_keys and key_swap not in existing_keys:
                        new_fallback.append(t)
                        existing_keys.add(key)
                if new_fallback:
                    logger.info(
                        "Document fallback: +%d triples for %s",
                        len(new_fallback),
                        doc.doc_id,
                    )
                    raw_triples.extend(new_fallback)

        if self.enable_supplement_extraction and self.extractor:
            # Build set of existing (subject, relation, object) for dedup
            existing_keys = {
                (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                for t in raw_triples
            }
            for chunk in doc.chunks:
                chunk_triples = [
                    t for t in raw_triples
                    if t.source_chunk_id == chunk.chunk_id
                ]
                if chunk_triples:
                    supplement = self.extractor.supplement_extraction(
                        chunk, chunk_triples
                    )
                    # Deduplicate: only keep genuinely new triples
                    new_supplement = []
                    for t in supplement:
                        key = (t.subject.lower().strip(), t.relation.value, t.object.lower().strip())
                        # Also check swapped direction
                        key_swap = (t.object.lower().strip(), t.relation.value, t.subject.lower().strip())
                        if key not in existing_keys and key_swap not in existing_keys:
                            new_supplement.append(t)
                            existing_keys.add(key)
                    if new_supplement:
                        logger.info(
                            "Supplement: +%d new triples for chunk %s (filtered %d dupes)",
                            len(new_supplement), chunk.chunk_id,
                            len(supplement) - len(new_supplement),
                        )
                        raw_triples.extend(new_supplement)

        # Cross-chunk pronoun resolution at document level
        if len(doc.chunks) > 1:
            from src.extraction.output_parser import resolve_pronouns
            raw_triples = resolve_pronouns(raw_triples)

        logger.info(f"Extraction: {len(raw_triples)} raw triples from {len(doc.chunks)} chunks")

        # Link prediction: infer relationships between disconnected subgraphs
        if self.link_predictor and raw_triples:
            recovery_share = (
                recovered_raw_count / len(raw_triples) if recovered_raw_count and raw_triples else 0.0
            )
            if (
                self.config.lp_skip_if_recovery_share_at_least is not None
                and recovery_share >= self.config.lp_skip_if_recovery_share_at_least
            ):
                logger.info(
                    "Link prediction skipped for %s: recovery share %.2f (%d/%d raw triples, %d primary)",
                    doc.doc_id,
                    recovery_share,
                    recovered_raw_count,
                    len(raw_triples),
                    primary_raw_count,
                )
            else:
                lp_triples = self.link_predictor.predict_links(
                    triples=raw_triples,
                    doc_text=doc.text,
                    source_doc_id=doc.doc_id,
                )
                if lp_triples:
                    logger.info(f"Link prediction: +{len(lp_triples)} inferred triples")
                    raw_triples.extend(lp_triples)

        # Guided re-extraction: symbolic type constraints guide focused LLM queries
        if self.guided_reextractor and raw_triples:
            guided_triples = self.guided_reextractor.guided_reextract(
                chunks=doc.chunks,
                existing_triples=raw_triples,
                source_doc_id=doc.doc_id,
            )
            if guided_triples:
                logger.info(f"Guided re-extraction: +{len(guided_triples)} triples")
                raw_triples.extend(guided_triples)
        return doc, raw_triples

    def postprocess(
        self, raw_triples: list[Triple], doc: CTIDocument
    ) -> tuple[CTIGraph, list[Triple]]:
        """Everything downstream of extraction: validation, filtering, grounding.

        Deterministic given its input, which is what makes the shared-extraction
        ablation valid.
        """
        all_triples = raw_triples

        # Module C: Validate
        if self.enable_validation:
            # Normalize weak/free-form local relations before hard validation to
            # reduce avoidable type-pair rejections.
            all_triples = self.relation_repairer.repair_triples(all_triples)
            all_triples = self.validator.validate_triples(all_triples)

        # Confidence re-scoring: prune low-confidence triples using symbolic signals
        if self.confidence_rescorer:
            all_triples = self.confidence_rescorer.rescore_and_prune(all_triples)

        # Saliency filter
        if self.saliency_filter:
            all_triples = self.saliency_filter.filter_triples(all_triples)

        # Module D: Canonicalize
        if self.enable_canonicalization:
            all_triples = self.canonicalizer.canonicalize_triples(all_triples)
            all_triples = self._ground_attack_techniques(all_triples)

        # Entity alignment: merge similar entities using embeddings
        if self.entity_aligner:
            all_triples = self.entity_aligner.align_triples(all_triples)

        # Build entity map and graph
        entities = self.canonicalizer.build_entity_map(all_triples)
        graph = build_graph(
            entities=entities,
            triples=all_triples,
            doc_id=doc.doc_id,
            provenance=self.provenance,
        )

        return graph, all_triples

    def process_document(self, doc: CTIDocument) -> tuple[CTIGraph, list[Triple]]:
        """Run the pipeline on a single document.

        Returns:
            (CTIGraph, list of all triples including rejected)
        """
        doc, raw_triples = self.extract_document(doc)
        if not doc.chunks:
            return CTIGraph(source_documents=[doc.doc_id]), []
        return self.postprocess(raw_triples, doc)


    def _ground_attack_techniques(self, triples: list[Triple]) -> list[Triple]:
        """Attach ATT&CK technique IDs and normalize exact ATT&CK technique names."""
        grounded: list[Triple] = []
        for triple in triples:
            if triple.validation_status.value == "rejected":
                grounded.append(triple)
                continue

            updated = triple.model_copy()
            attack_ids = list(updated.attack_ids)

            if updated.subject_type == EntityType.TECHNIQUE:
                tech_id = self.attack_mapper.lookup_by_name(updated.subject)
                if tech_id:
                    attack_ids.append(tech_id)
                    updated.subject = self.attack_mapper.lookup_by_id(tech_id)["name"]

            if updated.object_type == EntityType.TECHNIQUE:
                tech_id = self.attack_mapper.lookup_by_name(updated.object)
                if tech_id:
                    attack_ids.append(tech_id)
                    updated.object = self.attack_mapper.lookup_by_id(tech_id)["name"]

            if attack_ids:
                updated.attack_ids = list(dict.fromkeys(attack_ids))

            grounded.append(updated)

        return grounded

    def process_documents(
        self, docs: list[CTIDocument]
    ) -> tuple[CTIGraph, dict[str, list[Triple]]]:
        """Run the pipeline on multiple documents and optionally fuse.

        Returns:
            (merged CTIGraph, dict of doc_id -> all triples)
        """
        doc_graphs = []
        all_doc_triples: dict[str, list[Triple]] = {}

        for doc in docs:
            graph, triples = self.process_document(doc)
            doc_graphs.append(graph)
            all_doc_triples[doc.doc_id] = triples

        # Module E: Fuse
        if self.enable_fusion and len(doc_graphs) > 1:
            merged = self.merger.merge(doc_graphs)
        elif doc_graphs:
            merged = doc_graphs[0]
        else:
            merged = CTIGraph()

        return merged, all_doc_triples

    def get_ablation_config_name(self) -> str:
        """Return a human-readable name for the current pipeline config."""
        parts = ["B"]
        if self.enable_validation:
            parts.append("C")
        if self.enable_canonicalization:
            parts.append("D")
        if self.enable_fusion:
            parts.append("E")
        return "+".join(parts)
