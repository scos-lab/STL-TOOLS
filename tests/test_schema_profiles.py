# -*- coding: utf-8 -*-
"""Tests for schema profiles, typed edge rules and composite validation.

Adapted from Scorpse's ``test_software_schemas.py`` (fork of
scos-lab/semantic-tension-language, branch feature/software-schema-hardening).
The original tests exercised his software schema family under ``docs/schemas``;
that family is not part of stl-parser, so these tests use the minimal generic
fixtures under ``tests/fixtures/profiles/`` that exercise the same engine
behaviour: profile manifests, namespaced routing, cross-namespace references,
typed edge rules, and composite (strictest-wins) graph constraints.
"""

from pathlib import Path

import pytest

from stl_parser import (
    Anchor,
    Modifier,
    ParseResult,
    Statement,
    load_profile,
    load_schema,
    parse,
    parse_file,
    validate_against_profiles,
    validate_against_schema,
)
from stl_parser.errors import STLSchemaError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "profiles"
PROFILE_MANIFEST = FIXTURE_DIR / "composite.stl.profile"
EXAMPLE_DOCUMENT = FIXTURE_DIR / "example.stl"


def statement(source, target, relation=None, namespace=None, target_namespace=None, **extra):
    custom = dict(extra)
    if relation is not None:
        custom["relation"] = relation
    return Statement(
        source=Anchor(name=source, namespace=namespace),
        target=Anchor(name=target, namespace=target_namespace),
        modifiers=Modifier(custom=custom),
    )


def codes(result):
    return [error.code for error in result.errors]


# ========================================
# PROFILE MANIFEST LOADING
# ========================================


class TestLoadProfile:
    def test_manifest_loads_all_namespaces(self):
        profiles = load_profile(str(PROFILE_MANIFEST))

        assert set(profiles) == {"Core", "Delivery", "Operations"}
        assert profiles["Core"].name == "CoreFixture"
        assert all(schema.namespace == key for key, schema in profiles.items())

    def test_manifest_rejects_missing_manifest_file(self, tmp_path):
        with pytest.raises(STLSchemaError) as exc_info:
            load_profile(str(tmp_path / "absent.stl.profile"))

        assert exc_info.value.code == "E400"

    def test_manifest_rejects_missing_include(self, tmp_path):
        manifest = tmp_path / "missing.stl.profile"
        manifest.write_text("profile Missing v1.0 { include: [absent] }", encoding="utf-8")

        with pytest.raises(STLSchemaError, match="not found"):
            load_profile(str(manifest))

    def test_manifest_rejects_duplicate_namespaces(self, tmp_path):
        manifest = tmp_path / "duplicate.stl.profile"
        manifest.write_text("profile Duplicate v1.0 { include: [one, two] }", encoding="utf-8")
        schema_text = 'schema Example v1.0 { namespace "Same" }'
        (tmp_path / "one.stl.schema").write_text(schema_text, encoding="utf-8")
        (tmp_path / "two.stl.schema").write_text(schema_text, encoding="utf-8")

        with pytest.raises(STLSchemaError, match="Duplicate profile namespace"):
            load_profile(str(manifest))

    def test_manifest_rejects_schema_without_namespace(self, tmp_path):
        manifest = tmp_path / "anon.stl.profile"
        manifest.write_text("profile Anon v1.0 { include: [anon] }", encoding="utf-8")
        (tmp_path / "anon.stl.schema").write_text("schema Anon v1.0 { }", encoding="utf-8")

        with pytest.raises(STLSchemaError, match="has no namespace"):
            load_profile(str(manifest))

    @pytest.mark.parametrize(
        "text",
        [
            "profile Broken v1.0 { includes: [core] }",  # wrong key
            "profile Broken v1.0 { include: [core] ",  # unterminated
            "schema NotAProfile v1.0 { }",  # not a profile manifest
        ],
    )
    def test_manifest_rejects_malformed_text(self, tmp_path, text):
        manifest = tmp_path / "broken.stl.profile"
        manifest.write_text(text, encoding="utf-8")

        with pytest.raises(STLSchemaError) as exc_info:
            load_profile(str(manifest))

        assert exc_info.value.code == "E602"


# ========================================
# SINGLE-SCHEMA CONTRACT OF EACH FIXTURE
# ========================================

SCHEMA_CASES = [
    ("core.stl.schema", "Service_API", "Component_Auth", "contains"),
    ("delivery.stl.schema", "Build_42", "Artifact_API", "produces"),
    ("operations.stl.schema", "Metric_Latency", "Service_API", "observes"),
]


class TestFixtureSchemaContracts:
    @pytest.mark.parametrize(("filename", "source", "target", "relation"), SCHEMA_CASES)
    def test_accepts_its_contract(self, filename, source, target, relation):
        schema = load_schema(str(FIXTURE_DIR / filename))

        result = validate_against_schema(
            ParseResult(statements=[statement(source, target, relation)]), schema
        )

        assert result.is_valid is True, codes(result)

    @pytest.mark.parametrize(("filename", "source", "target", "relation"), SCHEMA_CASES)
    def test_rejects_missing_relation(self, filename, source, target, relation):
        schema = load_schema(str(FIXTURE_DIR / filename))

        result = validate_against_schema(
            ParseResult(statements=[statement(source, target)]), schema
        )

        assert result.is_valid is False
        assert "E607" in codes(result)

    @pytest.mark.parametrize(("filename", "source", "target", "relation"), SCHEMA_CASES)
    def test_rejects_unknown_relation(self, filename, source, target, relation):
        schema = load_schema(str(FIXTURE_DIR / filename))

        result = validate_against_schema(
            ParseResult(statements=[statement(source, target, "unknown")]), schema
        )

        assert result.is_valid is False
        assert "E603" in codes(result)  # enum violation
        assert "E611" in codes(result)  # no edge rule admits the triple

    @pytest.mark.parametrize(("filename", "source", "target", "relation"), SCHEMA_CASES)
    def test_rejects_unknown_source_prefix(self, filename, source, target, relation):
        schema = load_schema(str(FIXTURE_DIR / filename))

        result = validate_against_schema(
            ParseResult(statements=[statement("Unknown_Thing", target, relation)]), schema
        )

        assert result.is_valid is False
        assert "E606" in codes(result)

    def test_edge_rule_rejects_wrong_direction(self):
        schema = load_schema(str(FIXTURE_DIR / "core.stl.schema"))
        # Both anchors satisfy the patterns; only the edge rule can reject this.
        result = validate_against_schema(
            ParseResult(statements=[statement("Component_Auth", "Service_API", "contains")]),
            schema,
        )

        assert codes(result) == ["E611"]

    def test_string_pattern_is_enforced(self):
        schema = load_schema(str(FIXTURE_DIR / "delivery.stl.schema"))
        good = statement("Build_42", "Artifact_API", "produces", commit="abc1234")
        bad = statement("Build_42", "Artifact_API", "produces", commit="not-a-sha")

        assert validate_against_schema(ParseResult(statements=[good]), schema).is_valid
        assert codes(validate_against_schema(ParseResult(statements=[bad]), schema)) == ["E603"]


# ========================================
# COMPOSITE VALIDATION ACROSS PROFILES
# ========================================


class TestCompositeProfiles:
    @pytest.fixture
    def profiles(self):
        return load_profile(str(PROFILE_MANIFEST))

    def test_example_document_validates_across_all_profiles(self, profiles):
        document = parse_file(str(EXAMPLE_DOCUMENT))
        assert document.is_valid is True, [error.message for error in document.errors]

        result = validate_against_profiles(document, profiles)

        assert result.is_valid is True, [error.message for error in result.errors]
        assert result.schema_name == "CompositeProfiles"
        assert result.schema_version == "Core:v1.0,Delivery:v1.0,Operations:v1.0"
        assert {stmt.source.namespace for stmt in document.statements} == set(profiles)

    def test_cross_namespace_target_is_validated_by_its_own_profile(self, profiles):
        # Delivery source, Core target: the target must satisfy Core's target pattern.
        ok = statement(
            "Deployment_Prod", "Service_API", "deploys_to",
            namespace="Delivery", target_namespace="Core",
        )
        assert validate_against_profiles(ParseResult(statements=[ok]), profiles).is_valid

        wrong = statement(
            "Deployment_Prod", "Artifact_API", "deploys_to",
            namespace="Delivery", target_namespace="Core",
        )
        result = validate_against_profiles(ParseResult(statements=[wrong]), profiles)
        assert "E606" in codes(result)

    def test_unnamespaced_source_routes_by_unique_prefix(self, profiles):
        document = ParseResult(statements=[statement("Metric_Latency", "Service_API", "observes")])

        assert validate_against_profiles(document, profiles).is_valid

    def test_unknown_prefix_cannot_be_routed(self, profiles):
        document = ParseResult(statements=[statement("Unknown_Thing", "Service_API", "observes")])

        assert codes(validate_against_profiles(document, profiles)) == ["E610"]

    def test_strictest_chain_limit_wins(self, profiles):
        # Delivery declares max_chain_length 5, Operations 8, Core none:
        # 5 applies to the whole composite graph.
        document = parse_file(str(EXAMPLE_DOCUMENT))
        extra = parse(
            '[Core:Component_Auth] -> [Core:Component_Billing] ::mod(relation="calls")\n'
            '[Core:Component_Billing] -> [Core:Endpoint_Invoice] ::mod(relation="exposes")\n'
        )
        document = ParseResult(statements=document.statements + extra.statements)

        result = validate_against_profiles(document, profiles)

        assert codes(result) == ["E608"]
        assert "6 > 5" in result.errors[0].message

    def test_cycle_rejected_if_any_profile_forbids_cycles(self, profiles):
        # Only Core sets allow_cycles: false; it applies to the whole composite graph.
        document = parse_file(str(EXAMPLE_DOCUMENT))
        extra = parse('[Core:Component_Auth] -> [Core:Service_API] ::mod(relation="calls")')
        document = ParseResult(statements=document.statements + extra.statements)

        result = validate_against_profiles(document, profiles)

        assert codes(result) == ["E609"]

    def test_per_profile_statement_counts_apply_to_routed_subset(self, profiles):
        profiles["Operations"].constraints.min_statements = 1
        core_only = ParseResult(statements=[
            statement(
                "Service_API", "Component_Auth", "contains",
                namespace="Core", target_namespace="Core",
            ),
        ])

        result = validate_against_profiles(core_only, profiles)

        assert codes(result) == ["E605"]
        assert "Operations" in result.errors[0].message

        with_operations = ParseResult(statements=core_only.statements + [
            statement(
                "Metric_Latency", "Service_API", "observes",
                namespace="Operations", target_namespace="Core",
            ),
        ])
        assert validate_against_profiles(with_operations, profiles).is_valid
