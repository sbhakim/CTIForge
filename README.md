# CTIForge

Builds cyber threat intelligence knowledge graphs from unstructured threat
reports, with a deterministic validation layer sitting between the LLM extractor
and the graph.

The motivation is practical. LLM extractors treat their own output as final, so a
triple that lands in the graph carries no record of what was checked or repaired
along the way. When the graph is later wrong, there is no way to tell which
decision produced the error. CTIForge runs every candidate triple through a rule
cascade and logs each action, so the rule that admitted any given edge stays
recoverable from the output.

## Pipeline

Five stages, each independently toggleable for ablation:

- **A — Ingestion.** Loads reports, splits them into paragraph-level chunks with
  stable IDs and character offsets back into the source.
- **B — Extraction.** LLM prompting for typed triples with evidence spans.
  Optional few-shot retrieval and link prediction over disconnected subgraphs.
- **C — Validation.** Type-pair constraints with auto-swap repair for reversed
  arguments, IOC format normalisation, evidence alignment, placeholder
  detection. Every action lands in an error taxonomy log.
- **D — Grounding.** Alias-based canonicalisation and MITRE ATT&CK technique
  grounding.
- **E — Fusion.** Cross-document merge that keeps per-triple provenance.

## Setup

```bash
conda create -n cti python=3.11
conda activate cti
pip install -r requirements.txt
cp .env.example .env      # then add a key for whichever provider you use
```

Extraction goes through LiteLLM, so OpenAI, Anthropic, Google, DeepSeek,
OpenRouter, and local HuggingFace models all work. `configs/` has a starting
point for each.

## Data

The evaluation corpora are third-party and not redistributed here.

- CTI-Nexus: 149 annotated reports, from https://github.com/peng-gao-lab/CTINexus
- CTIKG: 255 annotated sentences, from the CTIKG release
- MITRE ATT&CK technique, group, and software records, used for grounding

Put them where the configs expect them: `data/annotations/ctinexus/` and
`data/external/mitre.jsonl`.

## Running

```bash
python main.py --config configs/default.yaml info

python main.py --config configs/default.yaml \
  extract data/annotations/ctinexus/<report>.json -o output/single

python main.py --config configs/default.yaml evaluate --max-docs 20 -o output/eval

python main.py --config configs/default.yaml ablation --max-docs 20 -o output/ablation
```

`scripts/` wraps the common cases.

Note that `eval_head_to_head.py` caches extracted triples per system. After
changing pipeline code, delete `<output-dir>/ctiforge_cached_triples.json` or you
will silently score the previous run.

## Evaluation

| Script | What it does |
|---|---|
| `eval_ctinexus_decomposed.py` | Triplet and subject–object metrics on CTI-Nexus |
| `eval_ctikg_benchmark.py` | Comparison against CTIKG on its 255-sentence set |
| `eval_head_to_head.py` | Matched-backbone comparison with CTI-Nexus and AttacKG+ |
| `eval_protocol_spread.py` | One prediction set scored under seven matchers |
| `eval_multisystem_spread.py` | Protocol sensitivity across systems; finds rank reversals |
| `eval_matcher_vs_human.py` | Matcher agreement against human-adjudicated judgements |
| `eval_error_taxonomy.py` | Aggregates validator logs into per-category tables |
| `structural_metrics.py` | Evidence presence, schema compliance, duplicate rate |

Two things about the harness are easy to get wrong.

**Matching is scoped per document.** Pooling a whole corpus into flat prediction
and gold lists before matching lets a prediction from one report be credited
against a different report's annotations. On our 149-document set that inflates
true positives by about 6%. Results carry both scopes; the one you want is tagged
`scope: per_document`.

**Not every metric supports cross-system comparison.**
`src/evaluation/schema_neutral.py` separates measures that depend on no
participant's ontology (STIX 2.1 relationship compliance, evidence presence,
duplicate rate) from measures scored against CTIForge's own constraint table. The
latter are definitional — the validator enforces that table at extraction time, so
CTIForge cannot score badly on them — and comparing another system against them
is not meaningful.

## Tests

```bash
pytest tests/ -q
```

Covers the validator, canonicaliser, schema contracts, per-document scoping,
schema-neutral scoring, and the evaluation harnesses.

## Layout

```
src/
  ingestion/     loaders, paragraph segmentation
  extraction/    extractor, output parser, few-shot retrieval, link prediction
  symbolic/      validator, relation repair, saliency filter, IOC rules, error logger
  grounding/     canonicaliser, alias tables, entity alignment, ATT&CK mapping
  fusion/        cross-document merge with provenance
  graph/         construction and export
  evaluation/    metrics, per-document scoping, schema-neutral scoring
  schema/        14 entity types, 12 relation types, type-pair constraints
prompts/         Jinja2 templates for extraction, recovery, link prediction
configs/         provider and pipeline configs
tests/
scripts/
```

## License

MIT. See `LICENSE`.

The evaluation corpora are third-party and carry their own terms; this licence
covers only the code in this repository.

## Contact

Please open an issue for project questions or contact Safayat Bin Hakim at
safayat DOT b DOT hakim AT gmail DOT com.
