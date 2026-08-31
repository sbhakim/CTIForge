"""Parser robustness against malformed LLM output."""

import json

from src.extraction.output_parser import parse_extraction_response


def test_null_fields_are_skipped_not_crashed():
    """A JSON null must not kill the run.

    `dict.get(k, default)` returns the default only when the key is absent; a
    key present with an explicit null yields None. Gemma-2-9B emitted
    {"object": null} and the resulting None.strip() aborted a 149-document
    extraction at document 6.
    """
    payload = json.dumps({"triples": [
        {"subject": "APT28", "relation": "uses", "object": None},
        {"subject": None, "relation": "uses", "object": "X-Agent"},
        {"subject": "APT28", "relation": None, "object": "X-Agent",
         "subject_type": None, "object_type": None},
        {"subject": "APT29", "relation": "uses", "object": "WellMess"},
    ]})
    triples = parse_extraction_response(payload, source_doc_id="d0",
                                   source_chunk_id="c0", source_text="")
    # The two rows with a null subject/object are dropped; the other two survive.
    assert len(triples) == 2
    assert {t.subject for t in triples} == {"APT28", "APT29"}
