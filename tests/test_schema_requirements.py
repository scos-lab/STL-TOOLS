"""Cross-statement requirement rule + identity resolver hook (agent-comms engine).

These cover the ``require { ... }`` block and the ``resolvers=`` parameter that let a
profile demand, e.g., that a ``merge`` is valid only when an independent,
registry-resolved ``verify`` (outcome=pass) exists — the structural fix for the
"gate required from a role bound to nobody" deadlock.
"""
from pathlib import Path

import pytest

from stl_parser import parse
from stl_parser.schema import load_schema, validate_against_schema

FIXTURE = Path(__file__).parent / "fixtures" / "agent_review_req.stl.schema"

MERGE_ONLY = '[Review:Verifier_Lane] -> [Review:WorkItem_T1] ::mod(action="merge")'
INDEP_VERIFY = (
    '[Review:Verifier_C] -> [Review:Revision_abc] '
    '::mod(action="verify", outcome="pass", verifier="claude", claimant="codex")'
)
SELF_VERIFY = (
    '[Review:Verifier_C] -> [Review:Revision_abc] '
    '::mod(action="verify", outcome="pass", verifier="codex", claimant="codex")'
)

RESOLVER = {"identity": lambda who: who in {"claude", "codex"}}


@pytest.fixture
def schema():
    return load_schema(str(FIXTURE))


def _validate(text, schema, resolvers=None):
    return validate_against_schema(parse(text), schema, resolvers=resolvers)


def test_require_block_parses(schema):
    assert len(schema.requirements) == 1
    r = schema.requirements[0]
    assert (r.trigger_action, r.binding_action, r.binding_outcome) == ("merge", "verify", "pass")
    assert r.independent is True
    assert r.resolver == "identity"


def test_merge_without_verify_fails_closed(schema):
    res = _validate(MERGE_ONLY, schema, resolvers=RESOLVER)
    assert not res.is_valid
    assert any(e.code == "E612" for e in res.errors)


def test_merge_with_independent_resolved_verify_passes(schema):
    res = _validate(f"{INDEP_VERIFY}\n{MERGE_ONLY}", schema, resolvers=RESOLVER)
    assert res.is_valid, [e.message for e in res.errors]


def test_missing_resolver_fails_closed(schema):
    # verify present and independent, but no resolver supplied -> unresolvable gate.
    res = _validate(f"{INDEP_VERIFY}\n{MERGE_ONLY}", schema, resolvers=None)
    assert not res.is_valid
    assert any("resolver" in e.message for e in res.errors)


def test_self_verify_is_not_independent(schema):
    res = _validate(f"{SELF_VERIFY}\n{MERGE_ONLY}", schema, resolvers=RESOLVER)
    assert not res.is_valid
    assert any(e.code == "E612" for e in res.errors)


def test_unknown_identity_rejected(schema):
    other = {"identity": lambda who: who == "someone_else"}
    res = _validate(f"{INDEP_VERIFY}\n{MERGE_ONLY}", schema, resolvers=other)
    assert not res.is_valid
