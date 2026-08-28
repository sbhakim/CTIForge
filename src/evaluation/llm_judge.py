"""LLM-judge protocol, using GRID's calibrated judge prompt.

No lexical or embedding matcher we tested agrees with human judgement more than
about 70% of the time; GRID's LLM judge reaches 86%. That makes the judge the
only human-aligned reference protocol available, so it gets its own column in the
multi-system spread table.

IMPORTANT, what "86.0%" is attached to. That figure was measured for a specific
pair: GRID's `grid_judge_fav` prompt AND the GPT-5.4-mini model, at temperature
0.1 with medium reasoning effort. This module reuses the prompt verbatim (writing
our own would just add an eighth arbitrary protocol rather than a calibrated
reference), but the model is configurable. Running a different model is a
deviation whose effect on human agreement is UNMEASURED and must be reported as
such, do not carry the 86.0% figure over to a judge you have not calibrated.

Cost discipline follows GRID's own lesson: their task-bank supervision cost ~$60
offline and was reusable, while the equivalent online LLM-as-judge cost ~$942.
Every response here is cached to disk keyed by model + prompt + payload, so a
re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

GRID_SRC = Path(__file__).resolve().parents[3] / "Codes" / "ProjectGRID" / "src"

# Approximate per-1M-token USD rates, for dry-run estimation only.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),
}


def load_grid_prompts() -> tuple[str, str]:
    """Return (precision_prompt, recall_prompt) from GRID's released source."""
    if str(GRID_SRC) not in sys.path:
        sys.path.insert(0, str(GRID_SRC))
    try:
        import tools_prompt_nano as t
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import GRID's judge prompts from {GRID_SRC}.\n"
            f"Expected the ProjectGRID clone alongside this repo. ({exc})"
        )
    bundle = t.get_judge_prompt_bundle("grid_judge_fav")
    return bundle["precision_prompt"], bundle["recall_prompt"]


def _fmt_edges(edges, prefix: str) -> str:
    return json.dumps(
        [{"index": f"{prefix}_{i}", "sub": s, "rel": r, "obj": o}
         for i, (s, r, o) in enumerate(edges)],
        indent=2, ensure_ascii=False,
    )


def _fmt_entities(edges) -> str:
    names = []
    for s, _, o in edges:
        for n in (s, o):
            if n and n not in names:
                names.append(n)
    return json.dumps([{"name": n} for n in names], indent=2, ensure_ascii=False)


def build_payloads(preds, golds, article: str) -> tuple[str, str]:
    """Assemble judge inputs in GRID's exact section layout."""
    p_prompt, r_prompt = load_grid_prompts()
    pred_fmt = _fmt_edges(preds, "predict_relationship")
    gt_fmt = _fmt_edges(golds, "truth_relationship")
    ents = _fmt_entities(golds)
    article_ctx = article if article else "N/A"

    precision_content = f"""{p_prompt}
--- Prediction Relations to Evaluate ---
{pred_fmt}

--- Ground Truth Reference Entities ---
{ents}

--- Ground Truth Reference Relations ---
{gt_fmt}

--- Article (Context) ---
{article_ctx}
"""
    recall_content = f"""{r_prompt}
--- Ground Truth Entities ---
{ents}

--- Ground Truth Relations ---
{gt_fmt}

--- Prediction Pool Relations ---
{pred_fmt}

--- Article (Context) ---
{article_ctx}
"""
    return precision_content, recall_content


def _cache_key(model: str, content: str) -> str:
    return hashlib.sha256(f"{model}\x00{content}".encode()).hexdigest()[:32]


def parse_verdicts(response: str) -> tuple[int, int]:
    """Return (n_positive, n_total) from the judge's JSON verdict list.

    The judge emits a list of {"index": ..., "result": "TP"/"FP"/"FN", ...}.
    Mirrors GRID's counting: positives are entries whose result is TP.
    """
    if not response or not response.strip():
        return 0, 0
    text = response.strip()
    if "```" in text:
        parts = [p for p in text.split("```") if p.strip()]
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("["):
                text = p
                break
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return 0, 0
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return 0, 0
    if not isinstance(items, list):
        return 0, 0
    results = [i for i in items if isinstance(i, dict) and "result" in i]
    tp = sum(1 for i in results if str(i["result"]).strip().upper().startswith("TP"))
    return tp, len(results)


class CachedJudge:
    """Judge wrapper with a disk cache. A repeated run costs nothing."""

    def __init__(self, model: str = "gpt-4o-mini",
                 cache_dir: str | Path = "output/judge_cache",
                 temperature: float = 0.1, max_tokens: int = 16384,
                 dry_run: bool = False):
        self.model = model
        self.cache_dir = Path(cache_dir) / model.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.stats = {"hits": 0, "calls": 0, "in_chars": 0, "out_chars": 0, "errors": 0}

    def _cached(self, key: str) -> str | None:
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())["response"]
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _store(self, key: str, content: str, response: str) -> None:
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"model": self.model, "content_sha": key,
                        "response": response}, ensure_ascii=False)
        )

    def ask(self, content: str) -> str:
        key = _cache_key(self.model, content)
        hit = self._cached(key)
        if hit is not None:
            self.stats["hits"] += 1
            return hit

        self.stats["in_chars"] += len(content)
        if self.dry_run:
            self.stats["calls"] += 1
            return ""

        from litellm import completion
        try:
            resp = completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            out = resp.choices[0].message.content or ""
        except Exception as exc:  # network/quota/parse, do not poison the cache
            self.stats["errors"] += 1
            print(f"    [judge error] {type(exc).__name__}: {str(exc)[:160]}")
            return ""

        self.stats["calls"] += 1
        self.stats["out_chars"] += len(out)
        self._store(key, content, out)
        return out

    def estimate_cost(self) -> dict:
        """Rough USD estimate from characters seen (~4 chars/token)."""
        in_tok = self.stats["in_chars"] / 4
        out_tok = self.stats["out_chars"] / 4 if self.stats["out_chars"] else self.stats["calls"] * 800
        pin, pout = PRICES.get(self.model, (None, None))
        if pin is None:
            return {"input_tokens": int(in_tok), "output_tokens": int(out_tok), "usd": None}
        return {
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "usd": round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 4),
            "usd_if_prefix_cached": round(in_tok / 1e6 * pin * 0.1 + out_tok / 1e6 * pout, 4),
        }


def judge_documents(judge: CachedJudge, items, workers: int = 6) -> list[dict]:
    """Judge many documents concurrently.

    Sequential judging of 965 calls takes hours; a small pool cuts it to minutes.
    Workers are kept modest to stay clear of rate limits, and CachedJudge.ask
    already swallows per-call errors, so one failure cannot abort the batch.
    Cache writes are one file per key, so concurrent writes do not collide.
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(
            lambda it: judge_document(judge, it[0], it[1], it[2]), items))


def judge_document(judge: CachedJudge, preds, golds, article: str) -> dict:
    """Score one document. Returns TP counts for precision and recall sides."""
    p_content, r_content = build_payloads(preds, golds, article)
    p_tp, p_n = parse_verdicts(judge.ask(p_content))
    r_tp, r_n = parse_verdicts(judge.ask(r_content))
    return {
        "precision_tp": p_tp, "precision_judged": p_n, "n_pred": len(preds),
        "recall_tp": r_tp, "recall_judged": r_n, "n_gold": len(golds),
    }
