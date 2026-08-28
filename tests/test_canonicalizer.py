"""Tests for canonicalization (Module D)."""

import json
from pathlib import Path

import pytest
from src.schema.entities import EntityType, Entity
from src.schema.relations import Triple, RelationType, ValidationStatus
from src.grounding.alias_tables import AliasTable, load_mitre_aliases
from src.grounding.canonicalizer import Canonicalizer
from src.grounding.attack_mapping import AttackMapper


class TestAliasTable:
    def test_add_and_resolve(self):
        table = AliasTable()
        table.add("APT29", ["Cozy Bear", "The Dukes"])
        assert table.resolve("Cozy Bear") == "APT29"
        assert table.resolve("the dukes") == "APT29"
        assert table.resolve("APT29") == "APT29"

    def test_resolve_unknown(self):
        table = AliasTable()
        assert table.resolve("unknown_actor") is None

    def test_get_aliases(self):
        table = AliasTable()
        table.add("APT29", ["Cozy Bear", "The Dukes"])
        aliases = table.get_aliases("APT29")
        assert "Cozy Bear" in aliases
        assert "The Dukes" in aliases

    def test_load_mitre_aliases_handles_nested_ctiarena_format(self, tmp_path: Path):
        path = tmp_path / "mitre.jsonl"
        entry = {
            "type": "tool",
            "contents": json.dumps(
                {"name": "Mimikatz", "x_mitre_aliases": ["mimikatz.exe", "kiwi"]}
            ),
        }
        path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        _, software_aliases = load_mitre_aliases(path)
        assert software_aliases.resolve("kiwi") == "Mimikatz"


class TestAttackMapper:
    def test_loads_ctiarena_style_technique_entries(self, tmp_path: Path):
        path = tmp_path / "mitre.jsonl"
        entry = {
            "mitre_id": "T1059",
            "title": "Command and Scripting Interpreter",
            "contents": json.dumps(
                {
                    "name": "Command and Scripting Interpreter",
                    "description": "Technique description",
                    "kill_chain_phases": [
                        {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                    ],
                }
            ),
            "metadata": {"source": "mitre_ttp"},
        }
        path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        mapper = AttackMapper(path)
        assert mapper.lookup_by_name("Command and Scripting Interpreter") == "T1059"
        assert mapper.lookup_by_id("T1059")["tactics"] == ["execution"]

    def test_fuzzy_lookup_matches_normalized_name(self, tmp_path: Path):
        path = tmp_path / "mitre.jsonl"
        entry = {
            "mitre_id": "T1059",
            "title": "Command and Scripting Interpreter",
            "contents": json.dumps({"name": "Command and Scripting Interpreter"}),
            "metadata": {"source": "mitre_ttp"},
        }
        path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        mapper = AttackMapper(path)
        assert mapper.lookup_by_name("Command & Scripting Interpreter") == "T1059"


class TestCanonicalizer:
    def _make_triple(self, subject="APT29", object_="Cobalt Strike") -> Triple:
        return Triple(
            event_id="test_t0",
            subject=subject,
            subject_type=EntityType.THREAT_ACTOR,
            relation=RelationType.USES,
            object=object_,
            object_type=EntityType.TOOL,
            evidence_text="test evidence",
            source_doc_id="doc1",
            source_chunk_id="doc1_chunk0",
            validation_status=ValidationStatus.VALIDATED,
        )

    def test_alias_resolution(self):
        group_aliases = AliasTable()
        group_aliases.add("APT29", ["Cozy Bear", "The Dukes"])

        canon = Canonicalizer(group_aliases=group_aliases)
        result = canon.canonicalize_name("Cozy Bear", EntityType.THREAT_ACTOR)
        assert result == "APT29"

    def test_canonicalize_triples(self):
        group_aliases = AliasTable()
        group_aliases.add("APT29", ["Cozy Bear"])

        canon = Canonicalizer(group_aliases=group_aliases)
        triple = self._make_triple(subject="Cozy Bear")
        results = canon.canonicalize_triples([triple])
        assert results[0].subject == "APT29"

    def test_rejected_triples_skipped(self):
        canon = Canonicalizer()
        triple = self._make_triple()
        triple.validation_status = ValidationStatus.REJECTED
        results = canon.canonicalize_triples([triple])
        assert results[0].subject == "APT29"  # unchanged

    def test_build_entity_map(self):
        canon = Canonicalizer()
        triples = [
            self._make_triple(subject="APT29", object_="Cobalt Strike"),
            self._make_triple(subject="APT29", object_="Mimikatz"),
        ]
        entities = canon.build_entity_map(triples)
        # APT29 should appear once (deduplicated)
        apt29_id = Entity.make_node_id(EntityType.THREAT_ACTOR, "APT29")
        assert apt29_id in entities
        assert len(entities[apt29_id].source_mentions) >= 1

    def test_cache_is_type_aware(self):
        group_aliases = AliasTable()
        software_aliases = AliasTable()
        group_aliases.add("BlackCat Actor", ["BlackCat"])
        software_aliases.add("BlackCat Malware", ["BlackCat"])

        canon = Canonicalizer(
            group_aliases=group_aliases,
            software_aliases=software_aliases,
        )
        assert canon.canonicalize_name("BlackCat", EntityType.THREAT_ACTOR) == "BlackCat Actor"
        assert canon.canonicalize_name("BlackCat", EntityType.MALWARE) == "BlackCat Malware"
