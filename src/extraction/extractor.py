"""LLM-based CTI triple extraction engine (Module B)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jinja2
from litellm import completion

from src.schema.graph_schema import Chunk
from src.schema.relations import Triple
from src.extraction.output_parser import parse_extraction_response
from src.utils.config import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class Extractor:
    """Extract CTI triples from text chunks using LLM prompting."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        cloud_response_format: dict[str, Any] | None = None,
        retry_attempts: int = 1,
        max_triples_per_chunk: int = 25,
        local_system_prompt: str = "You are a CTI extraction assistant.",
        local_do_sample: bool = False,
        local_max_new_tokens: int = 1280,
        local_prompt_template: str | None = None,
        prompts_dir: str | Path = "prompts",
        raw_output_dir: str | Path = "output/debug_raw_generations",
    ):
        settings = get_settings()
        self.provider = provider or "openai"
        self.model = model or settings.cti_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cloud_response_format = cloud_response_format
        self.retry_attempts = max(retry_attempts, 1)
        self.max_triples_per_chunk = max(max_triples_per_chunk, 1)
        self.local_system_prompt = local_system_prompt
        self.local_do_sample = local_do_sample
        self.local_max_new_tokens = max(local_max_new_tokens, 64)
        self.local_prompt_template = local_prompt_template
        self._local_tokenizer: Any | None = None
        self._local_model: Any | None = None
        self.raw_output_dir = Path(raw_output_dir)

        # Load Jinja2 prompt templates
        prompts_path = Path(prompts_dir)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_path)),
            autoescape=False,
        )
        self.extraction_template = self.jinja_env.get_template("extraction.jinja")
        self.compact_extraction_template = self.jinja_env.get_template("extraction_compact.jinja")
        self.supplement_template = self.jinja_env.get_template("extraction_supplement.jinja")
        self.recovery_template = self.jinja_env.get_template("extraction_recovery.jinja")
        self.document_fallback_template = self.jinja_env.get_template(
            "extraction_document_fallback.jinja"
        )
        self.local_extraction_template = None
        if self.local_prompt_template:
            self.local_extraction_template = self.jinja_env.get_template(self.local_prompt_template)

    @staticmethod
    def build_focus_text(source_text: str, max_sentences: int = 6) -> str:
        """Select a short, high-signal verbatim sentence pack from a report."""
        sentences = Extractor._split_focus_sentences(source_text)
        if not sentences:
            return ""

        scored: list[tuple[int, int, str]] = []
        for idx, sentence in enumerate(sentences):
            score = Extractor._score_focus_sentence(sentence)
            if score > 0:
                scored.append((score, idx, sentence))

        if not scored:
            scored = [
                (0, idx, sentence)
                for idx, sentence in enumerate(sentences[: max(1, min(max_sentences, 3))])
            ]

        top = sorted(scored, key=lambda item: (-item[0], item[1]))[: max(1, max_sentences)]
        ordered = [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]
        return "\n".join(ordered).strip()

    @staticmethod
    def _split_focus_sentences(source_text: str) -> list[str]:
        """Split report text into sentence-like spans for focus recovery."""
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", source_text.strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _score_focus_sentence(sentence: str) -> int:
        """Score likely benchmark-bearing sentences higher than boilerplate."""
        text = sentence.lower()
        score = 0

        if "cve-" in text or "zero-day" in text or "0-day" in text:
            score += 5
        if any(token in text for token in ("exploit", "exploiting", "abused", "vulnerability")):
            score += 4
        if any(
            token in text for token in (
                "uses", "used", "targets", "targeted", "delivers", "drops",
                "attributed", "linked to", "associated with", "located in",
                "patched", "mitigated", "communicates with",
            )
        ):
            score += 3
        if any(
            token in text for token in (
                "ransomware", "malware", "spyware", "backdoor", "campaign",
                "threat actor", "apt", "group", "hackers", "attackers",
            )
        ):
            score += 2
        if any(
            token in text for token in (
                "government", "organization", "company", "email", "credentials",
                "tokens", "infrastructure", "server", "victim", "sector",
            )
        ):
            score += 1
        if re.search(r"\b(?:apt|fin)\d+\b", text):
            score += 2
        if len(sentence) < 45:
            score -= 1

        return score

    def _uses_local_backend(self) -> bool:
        return self.provider.startswith("local_") or self.model.startswith("huggingface/")

    def _local_model_id(self) -> str:
        if self.model.startswith("huggingface/"):
            return self.model.split("/", 1)[1]
        return self.model

    def _ensure_local_backend(self) -> None:
        if self._local_model is not None and self._local_tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Local inference requires torch and transformers in the active environment"
            ) from e

        model_id = self._local_model_id()
        logger.info("Loading local extraction model: %s", model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            if hasattr(torch, "bfloat16"):
                model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            logger.warning(
                "CUDA is not available; local model %s will run on CPU and may be too slow",
                model_id,
            )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._local_tokenizer = tokenizer
        self._local_model = model

    def _build_local_prompt(self, prompt: str) -> str:
        """Use a stricter local-only prompt without affecting the cloud path."""
        return (
            f"{self.local_system_prompt}\n\n"
            "Return exactly one JSON object with this shape:\n"
            '{"triples":[{"subject":"","subject_type":"","relation":"","object":"","object_type":"","evidence_text":""}]}\n'
            "Rules:\n"
            "- Output JSON only.\n"
            "- No markdown fences.\n"
            "- No explanation.\n"
            "- Extract at most 10 of the strongest fully supported triples.\n"
            "- If nothing is supported, return {\"triples\": []}.\n\n"
            f"{prompt}\n\n"
            "JSON:"
        )

    def _build_strict_retry_prompt(self, text: str) -> str:
        """Fallback prompt for providers that returned malformed/truncated JSON."""
        return (
            "You are a CTI extraction assistant.\n\n"
            "Return exactly one valid JSON object with this shape:\n"
            '{"triples":[{"subject":"","subject_type":"","relation":"","object":"","object_type":"","evidence":""}]}\n'
            "Rules:\n"
            "- Output JSON only.\n"
            "- No markdown fences.\n"
            "- No explanation.\n"
            "- Extract at most 6 of the strongest fully supported triples.\n"
            "- Every triple must include exact supporting evidence from the text.\n"
            '- If nothing is supported, return {"triples": []}.\n\n'
            "Text to analyze:\n"
            f"{text}\n"
        )

    def _render_extraction_prompt(
        self,
        text: str,
        few_shot_examples: list[dict] | None = None,
        compact_mode: bool = False,
        compact_max_triples: int = 8,
    ) -> str:
        examples = few_shot_examples or []
        if self.provider == "local_mistral" and self.local_extraction_template is not None:
            return self.local_extraction_template.render(
                text=text,
                few_shot_examples=examples,
            )
        if compact_mode:
            return self.compact_extraction_template.render(
                text=text,
                few_shot_examples=examples,
                max_triples=max(compact_max_triples, 1),
            )
        return self.extraction_template.render(
            text=text,
            few_shot_examples=examples,
        )

    def _complete_local(self, prompt: str) -> str:
        self._ensure_local_backend()

        import torch

        tokenizer = self._local_tokenizer
        model = self._local_model
        assert tokenizer is not None
        assert model is not None

        local_prompt = self._build_local_prompt(prompt)
        messages = [
            {"role": "system", "content": self.local_system_prompt},
            {"role": "user", "content": local_prompt},
        ]

        # Preserve the original chat-template-based path for models/tokenizers
        # that define one. This worked for Qwen and should remain the preferred
        # route when available.
        #
        # Previous implementation:
        # encoded = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     return_tensors="pt",
        #     return_dict=True,
        # )
        # input_ids = encoded["input_ids"]
        # attention_mask = encoded.get("attention_mask")
        #
        # Some cached local models (for example the current Llama 3.1 tokenizer)
        # do not expose `tokenizer.chat_template`. In that case we fall back to a
        # plain instruction-style prompt while keeping the original logic intact.
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = tokenizer(
                local_prompt,
                return_tensors="pt",
            )

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        device = getattr(model, "device", None)
        if device is not None:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        generate_kwargs = {
            "max_new_tokens": min(self.local_max_new_tokens, self.max_tokens),
            "do_sample": self.local_do_sample,
            "temperature": max(self.temperature, 1e-5) if self.local_do_sample else None,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if generate_kwargs["temperature"] is None:
            generate_kwargs.pop("temperature")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _complete_cloud(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call the remote model once with consistent timeout handling."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a CTI extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "timeout": 120,
        }
        if self.cloud_response_format is not None:
            request_kwargs["response_format"] = self.cloud_response_format

        try:
            response = completion(**request_kwargs)
        except Exception as e:
            if self.cloud_response_format is None:
                raise
            logger.warning(
                "Cloud completion with response_format failed for %s; retrying without JSON mode: %s",
                self.model,
                e,
            )
            request_kwargs.pop("response_format", None)
            response = completion(**request_kwargs)
        return response.choices[0].message.content or ""

    def _complete_once(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run one extraction completion through the active backend."""
        if self._uses_local_backend():
            return self._complete_local(prompt)
        return self._complete_cloud(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _should_retry_after_parse_failure(self, raw_output: str) -> bool:
        """Retry only when the provider attempted structured output."""
        normalized = raw_output.strip().lower()
        if not normalized:
            return False
        return (
            normalized.startswith("{")
            or normalized.startswith("```")
            or '"triples"' in normalized
            or '"subject"' in normalized
        )

    def _save_raw_output(
        self,
        chunk: Chunk,
        raw_output: str,
        prompt: str,
        suffix: str = "raw",
    ) -> None:
        """Persist prompt/output pairs for debugging parse failures."""
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        safe_chunk = re.sub(r"[^A-Za-z0-9_.-]+", "_", chunk.chunk_id)
        (self.raw_output_dir / f"{safe_chunk}.{suffix}.prompt.txt").write_text(
            prompt,
            encoding="utf-8",
        )
        (self.raw_output_dir / f"{safe_chunk}.{suffix}.output.txt").write_text(
            raw_output,
            encoding="utf-8",
        )

    def extract_from_chunk(
        self,
        chunk: Chunk,
        few_shot_examples: list[dict] | None = None,
        compact_mode: bool = False,
        compact_max_triples: int = 8,
        override_temperature: float | None = None,
        override_max_tokens: int | None = None,
    ) -> list[Triple]:
        """Extract triples from a single text chunk."""
        prompt = self._render_extraction_prompt(
            text=chunk.text,
            few_shot_examples=few_shot_examples,
            compact_mode=compact_mode,
            compact_max_triples=compact_max_triples,
        )

        raw_output = ""
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw_output = self._complete_once(
                    prompt,
                    temperature=override_temperature,
                    max_tokens=override_max_tokens,
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM call failed for chunk %s on attempt %s/%s: %s",
                    chunk.chunk_id,
                    attempt,
                    self.retry_attempts,
                    e,
                )

        if not raw_output:
            logger.error(f"LLM call failed for chunk {chunk.chunk_id}: {last_error}")
            return []

        triples = parse_extraction_response(
            raw_response=raw_output,
            source_doc_id=chunk.doc_id,
            source_chunk_id=chunk.chunk_id,
            source_text=chunk.text,
        )
        if raw_output and not triples:
            self._save_raw_output(chunk, raw_output, prompt)
            if self._should_retry_after_parse_failure(raw_output):
                retry_prompt = self._build_strict_retry_prompt(chunk.text)
                try:
                    retry_output = self._complete_once(
                        retry_prompt,
                        temperature=0.0,
                        max_tokens=min(self.max_tokens, 1800),
                    )
                except Exception as e:
                    logger.warning(
                        "Strict retry failed for chunk %s: %s",
                        chunk.chunk_id,
                        e,
                    )
                    retry_output = ""

                if retry_output:
                    triples = parse_extraction_response(
                        raw_response=retry_output,
                        source_doc_id=chunk.doc_id,
                        source_chunk_id=chunk.chunk_id,
                        source_text=chunk.text,
                    )
                    if triples:
                        logger.info(
                            "Chunk %s: recovered %d triples from strict retry",
                            chunk.chunk_id,
                            len(triples),
                        )
                    else:
                        self._save_raw_output(
                            chunk,
                            retry_output,
                            retry_prompt,
                            suffix="retry",
                        )
        triples = triples[: self.max_triples_per_chunk]

        logger.info(
            "Chunk %s: extracted %d triples from %d chars%s",
            chunk.chunk_id,
            len(triples),
            len(chunk.text),
            " [compact]" if compact_mode else "",
        )
        return triples

    def supplement_extraction(
        self,
        chunk: Chunk,
        existing_triples: list[Triple],
    ) -> list[Triple]:
        """Run a supplementary extraction pass to find missed triples."""
        prompt = self.supplement_template.render(
            text=chunk.text,
            existing_triples=existing_triples,
        )

        try:
            raw_output = self._complete_once(prompt)
        except Exception as e:
            logger.warning(f"Supplement extraction failed for {chunk.chunk_id}: {e}")
            return []

        new_triples = parse_extraction_response(
            raw_response=raw_output,
            source_doc_id=chunk.doc_id,
            source_chunk_id=chunk.chunk_id,
            source_text=chunk.text,
        )
        if raw_output and not new_triples:
            self._save_raw_output(chunk, raw_output, prompt, suffix="supplement")

        if new_triples:
            logger.info(
                f"Supplement pass for {chunk.chunk_id}: found {len(new_triples)} additional triples"
            )

        return new_triples

    def recover_low_yield_chunk(
        self,
        chunk: Chunk,
        existing_triples: list[Triple],
        max_recovery_triples: int = 8,
    ) -> list[Triple]:
        """Run a targeted second pass when the first extraction yielded too little."""
        prompt = self.recovery_template.render(
            text=chunk.text,
            existing_triples=existing_triples,
            max_recovery_triples=max(max_recovery_triples, 1),
        )

        try:
            raw_output = self._complete_once(
                prompt,
                temperature=0.0,
                max_tokens=min(self.max_tokens, 2200),
            )
        except Exception as e:
            logger.warning("Low-yield recovery failed for %s: %s", chunk.chunk_id, e)
            return []

        recovered = parse_extraction_response(
            raw_response=raw_output,
            source_doc_id=chunk.doc_id,
            source_chunk_id=chunk.chunk_id,
            source_text=chunk.text,
        )
        if raw_output and not recovered:
            self._save_raw_output(chunk, raw_output, prompt, suffix="recovery")
            return []

        for triple in recovered:
            triple.confidence = min(triple.confidence, 0.9)

        if recovered:
            logger.info(
                "Recovery pass for %s: found %d candidate triples",
                chunk.chunk_id,
                len(recovered),
            )

        return recovered

    def recover_low_yield_document(
        self,
        chunk: Chunk,
        existing_triples: list[Triple],
        max_fallback_triples: int = 8,
    ) -> list[Triple]:
        """Retry a single hard document with a stricter, precision-first prompt."""
        prompt = self.document_fallback_template.render(
            text=chunk.text,
            existing_triples=existing_triples,
            max_fallback_triples=max(max_fallback_triples, 1),
        )

        try:
            raw_output = self._complete_once(
                prompt,
                temperature=0.0,
                max_tokens=min(self.max_tokens, 2200),
            )
        except Exception as e:
            logger.warning("Document fallback failed for %s: %s", chunk.chunk_id, e)
            return []

        recovered = parse_extraction_response(
            raw_response=raw_output,
            source_doc_id=chunk.doc_id,
            source_chunk_id=chunk.chunk_id,
            source_text=chunk.text,
        )
        if raw_output and not recovered:
            self._save_raw_output(chunk, raw_output, prompt, suffix="doc_fallback")
            return []

        for triple in recovered:
            triple.confidence = min(triple.confidence, 0.88)

        if recovered:
            logger.info(
                "Document fallback for %s: found %d candidate triples",
                chunk.chunk_id,
                len(recovered),
            )

        return recovered

    def recover_from_focus_text(
        self,
        chunk: Chunk,
        focus_text: str,
        existing_triples: list[Triple],
        max_recovery_triples: int = 8,
    ) -> list[Triple]:
        """Retry extraction on a short verbatim sentence pack from the report."""
        if not focus_text.strip():
            return []

        prompt = self.recovery_template.render(
            text=focus_text,
            existing_triples=existing_triples,
            max_recovery_triples=max(max_recovery_triples, 1),
        )

        try:
            raw_output = self._complete_once(
                prompt,
                temperature=0.0,
                max_tokens=min(self.max_tokens, 1600),
            )
        except Exception as e:
            logger.warning("Focused recovery failed for %s: %s", chunk.chunk_id, e)
            return []

        recovered = parse_extraction_response(
            raw_response=raw_output,
            source_doc_id=chunk.doc_id,
            source_chunk_id=chunk.chunk_id,
            source_text=chunk.text,
        )
        if raw_output and not recovered:
            self._save_raw_output(chunk, raw_output, prompt, suffix="focus_recovery")
            return []

        for triple in recovered:
            triple.confidence = min(triple.confidence, 0.88)

        if recovered:
            logger.info(
                "Focused recovery for %s: %d candidate triples from %d focus chars",
                chunk.chunk_id,
                len(recovered),
                len(focus_text),
            )

        return recovered

    def extract_from_document(
        self,
        chunks: list[Chunk],
        few_shot_examples: list[dict] | None = None,
    ) -> list[Triple]:
        """Extract triples from all chunks of a document."""
        all_triples = []
        for chunk in chunks:
            triples = self.extract_from_chunk(chunk, few_shot_examples)
            all_triples.extend(triples)
        return all_triples
