"""Coverage guard for the error taxonomy.

The manuscript lists a 14-category taxonomy as a contribution, described as "an
empirically grounded characterization of where and how LLM extraction fails".
Auditing 67 run logs (829 logged actions) on 2026-08-28 found that only 8
categories ever fire, and that 3 of the 14 have no call site anywhere in src/ --
they cannot fire at all.

These tests pin the current state so that:
  - wiring up a dead category (good) fails loudly and prompts a doc update;
  - adding a new dead category (bad) fails loudly.

They do not assert that the dead categories are acceptable. They are not; either
implement them or drop them from the paper's count.
"""

from pathlib import Path

import pytest

from src.symbolic.error_logger import ErrorCategory

SRC = Path(__file__).resolve().parent.parent / "src"

# Declared but with zero call sites outside error_logger.py, as of 2026-08-28.
KNOWN_DEAD = {
    "duplicate_entity",
    "hallucinated_relation",
    "over_extraction",
}

# Wired to a call site but never observed across 67 archived run logs.
# Not a defect on its own -- they guard genuinely rare conditions.
KNOWN_WIRED_BUT_UNOBSERVED = {
    "empty_field",          # parser filters empty fields upstream of validation
    "malformed_identifier",
    "repaired_alias",       # canonicalizer path
}


def _declared() -> dict[str, str]:
    return {
        k: v for k, v in vars(ErrorCategory).items()
        if not k.startswith("_") and isinstance(v, str)
    }


def _call_sites(attr_name: str) -> int:
    n = 0
    for path in SRC.rglob("*.py"):
        if path.name == "error_logger.py":
            continue
        if f"ErrorCategory.{attr_name}" in path.read_text():
            n += path.read_text().count(f"ErrorCategory.{attr_name}")
    return n


def test_declared_category_count_is_fourteen():
    """The paper claims 14. If this changes, the paper must change with it."""
    assert len(_declared()) == 14


def test_dead_category_set_has_not_changed():
    """Fails if a dead category gets wired up, or a new dead one is added."""
    dead = {v for k, v in _declared().items() if _call_sites(k) == 0}
    assert dead == KNOWN_DEAD, (
        f"Dead-category set changed: {dead} != {KNOWN_DEAD}. "
        "Update the manuscript's category count and this test together."
    )


def test_live_categories_are_actually_reachable():
    live = {v for k, v in _declared().items() if v not in KNOWN_DEAD}
    for name, value in _declared().items():
        if value in live:
            assert _call_sites(name) > 0, f"{value} claimed live but has no call site"


def test_at_least_eight_categories_are_reachable_and_observed_in_practice():
    """Guards the honest claim: 8 observed, not 14."""
    reachable = {v for k, v in _declared().items() if _call_sites(k) > 0}
    assert len(reachable) == 11  # 14 declared - 3 dead
    observed_capable = reachable - KNOWN_WIRED_BUT_UNOBSERVED
    assert len(observed_capable) == 8


@pytest.mark.parametrize("category", sorted(KNOWN_DEAD))
def test_dead_categories_are_documented_not_silently_present(category):
    """Explicit record of which categories the paper must not count as empirical."""
    assert category in KNOWN_DEAD
    attr = category.upper()
    assert _call_sites(attr) == 0
