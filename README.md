# STL Parser (v1.9.0)

**A comprehensive Python toolkit for Semantic Tension Language (STL) — parse, build, validate, query, diff, stream, and repair structured knowledge.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

`stl-parser` provides feature parity with the JSON ecosystem for STL documents:

| Capability | Functions | JSON Equivalent |
|-----------|-----------|-----------------|
| **Parse & Serialize** | `parse()`, `to_json()`, `to_stl()`, `to_rdf()` | `json.loads()` / `json.dumps()` |
| **Build Programmatically** | `stl()`, `stl_doc()`, `StatementBuilder` | Manual dict construction |
| **Schema Validation** | `load_schema()`, `validate_against_schema()` | JSON Schema |
| **LLM Output Repair** | `clean()`, `repair()`, `validate_llm_output()` | N/A (STL-unique) |
| **Query & Filter** | `find()`, `filter_statements()`, `select()`, `stl_pointer()` | `jq` / JSONPath |
| **Diff & Patch** | `stl_diff()`, `stl_patch()`, `diff_to_text()` | `json-diff` / `json-patch` |
| **Streaming I/O** | `STLEmitter`, `STLReader`, `stream_parse()` | NDJSON streaming |
| **Graph Analysis** | `STLGraph`, `STLAnalyzer`, `extract_chains()` | N/A (STL-unique) |
| **Confidence Decay** | `effective_confidence()`, `decay_report()` | N/A (STL-unique) |
| **CLI** | 11 commands (validate, convert, query, chain, diff, ...) | `jq` CLI |

## Installation

### From PyPI

```bash
pip install stl-parser
```

### From Source (Development)

```bash
git clone https://github.com/scos-lab/STL-TOOLS.git
cd STL-TOOLS
pip install -e ".[dev]"
```

## Quick Start

### Parse STL

```python
from stl_parser import parse

result = parse('[Einstein] -> [Theory_Relativity] ::mod(confidence=0.98, rule="empirical")')
stmt = result.statements[0]
print(stmt.source.name)          # "Einstein"
print(stmt.modifiers.confidence)  # 0.98
```

### Build Programmatically

```python
from stl_parser import stl

stmt = stl("[Rain]", "[Flooding]").mod(confidence=0.85, rule="causal", strength=0.8).build()
print(str(stmt))  # [Rain] -> [Flooding] ::mod(confidence=0.85, rule="causal", strength=0.8)
```

### Build a Multi-Statement Document

```python
from stl_parser import stl, stl_doc

doc = stl_doc(
    stl("[Rain]", "[Flooding]").mod(rule="causal", confidence=0.85),
    stl("[Flooding]", "[Evacuation]").mod(rule="causal", confidence=0.90),
)
print(len(doc.statements))  # 2
```

### Validate Against a Schema

```python
from stl_parser import parse, load_schema, validate_against_schema

schema = load_schema("docs/schemas/causal.stl.schema")
result = parse('[Rain] -> [Flooding] ::mod(rule="causal", confidence=0.85, strength=0.8)')
validation = validate_against_schema(result, schema)
print(f"Valid: {validation.is_valid}")
```

### Clean LLM Output

LLMs (especially small local models) generate imperfect STL. The 3-stage `clean -> repair -> parse` pipeline auto-fixes common issues:

```python
from stl_parser import validate_llm_output

# Spaces in anchors, wrong arrow, missing ::, typo, value out of range — all fixed
result = validate_llm_output("[Big Problem] -> [Smart Solution] mod(confience=1.2, strenght=0.3)")
print(result.statements[0])
# [Big_Problem] -> [Smart_Solution] ::mod(confidence=1.0, strength=0.3)

print(f"Valid: {result.is_valid}, Repairs: {len(result.repairs)}")
# Valid: True, Repairs: 6
```

**Auto-repair capabilities:**

| LLM Error | Example | Repaired To |
|-----------|---------|-------------|
| Spaces in anchors | `[Heavy Rain]` | `[Heavy_Rain]` |
| Wrong arrow | `=>`, `➡`, `—>` | `->` |
| Missing `::` prefix | `mod(...)` | `::mod(...)` |
| Missing brackets | `Rain -> [Flood]` | `[Rain] -> [Flood]` |
| Modifier typos | `confience`, `strenght` | `confidence`, `strength` |
| Value out of range | `confidence=1.5` | `confidence=1.0` |
| Unquoted strings | `rule=causal` | `rule="causal"` |
| Multi-word strings | `location=San Francisco` | `location="San Francisco"` |
| Bare list values | `data=1,2,3, bins=5` | `data="1,2,3", bins=5` |
| JSON arrays | `items=["a","b"]` | `items="[\"a\",\"b\"]"` |
| JSON objects | `area={"w": 20}` | `area="{\"w\": 20}"` |
| Tuples | `teams=("A","B")` | `teams="(\"A\",\"B\")"` |
| Boolean case | `flag=True` | `flag=true` |
| Code fences | `` ```stl ... ``` `` | Extracted content |
| Multi-line statements | Split across lines | Merged |
| Illegal anchor chars | `[Café & Bar]` | `[Caf_and_Bar]` |
| Overlong anchor names | 139 chars | Truncated at 64 (word boundary) |
| Unclosed quotes | `description="text, ts="` | `description="text", ts="` |
| Orphan keys (no value) | `confidence=0.9, description,` | `confidence=0.9` |
| Incomplete `::mod` | `::mod ::mod(...)` | `::mod(...)` |
| Missing `]` on anchor | `[Name ::mod(` | `[Name] ::mod(` |
| `=` in anchor | `[Cue=Value]` | `[Cue_Value]` |
| Quoted numerics | `confidence="0.95"` | `confidence=0.95` |
| Empty numeric fields | `confidence=""` | (removed) |

The repair pipeline uses a smart tokenizer (`_split_mod_pairs`) that tracks bracket/brace/paren nesting depth and identifies `key=value` boundaries correctly — even when values contain commas, spaces, or nested structures.

This makes STL a reliable LLM-to-program communication protocol — small models (0.6B-14B) that struggle with JSON tool calling can generate STL naturally, and the parser handles the rest. In BFCL benchmarks, qwen2.5:7b achieves **100% tool call accuracy** with STL (pure text generation) vs 97% with JSON (framework-assisted).

### Query Statements

```python
from stl_parser import parse, find, find_all, filter_statements, select

result = parse("""
[Rain] -> [Flooding] ::mod(rule="causal", confidence=0.85)
[Sun] -> [Evaporation] ::mod(rule="causal", confidence=0.90)
[Wind] -> [Erosion] ::mod(rule="causal", confidence=0.70)
""")

# Find first match ("source" resolves to source.name)
stmt = find(result, source="Rain")

# Filter returns a new ParseResult
high_conf = filter_statements(result, confidence__gte=0.85)
print(len(high_conf.statements))  # 2

# Extract a single field from all statements
names = select(result, field="source")   # ["Rain", "Sun", "Wind"]
confs = select(result, field="confidence")  # [0.85, 0.9, 0.7]
```

### Diff & Patch

```python
from stl_parser import parse, stl_diff, stl_patch

before = parse('[A] -> [B] ::mod(confidence=0.8)')
after  = parse('[A] -> [B] ::mod(confidence=0.9)')

diff = stl_diff(before, after)
print(diff.summary)  # {added: 0, removed: 0, modified: 1}

patched = stl_patch(before, diff)
```

### Streaming I/O

```python
from stl_parser import STLEmitter, stream_parse

# Write statements to a file
with STLEmitter("output.stl") as emitter:
    emitter.emit("[A]", "[B]", confidence=0.9, rule="causal")
    emitter.emit("[C]", "[D]", confidence=0.8, rule="logical")

# Stream-parse a file
for stmt in stream_parse("output.stl"):
    print(f"{stmt.source.name} -> {stmt.target.name}")
```

### Extract Chains

Discover transitive relationships by extracting all directed chains from STL:

```python
from stl_parser import parse, extract_chains, STLGraph

result = parse("""
[Rain] -> [Flooding] ::mod(rule="causal", confidence=0.85)
[Flooding] -> [Evacuation] ::mod(rule="causal", confidence=0.90)
[Evacuation] -> [Shelter] ::mod(rule="causal", confidence=0.80)
""")

chains = extract_chains(result, min_length=1)
print(STLGraph.format_chains(chains))
# Chain 1: [Rain] → [Flooding] → [Evacuation] → [Shelter]
```

### Confidence Decay

```python
from stl_parser import parse, effective_confidence

# Statement with a timestamp
result = parse('[Fact_X] -> [Conclusion_Y] ::mod(confidence=0.95, timestamp="2020-01-01T00:00:00Z")')
stmt = result.statements[0]

# Compute decayed confidence (exponential half-life)
decayed = effective_confidence(stmt, half_life_days=365)
print(f"Original: {stmt.modifiers.confidence}, Decayed: {decayed:.4f}")
```

## CLI Reference

The `stl` command provides 11 subcommands:

```bash
# Validate an STL file
stl validate input.stl

# Parse and output as JSON
stl parse input.stl --json

# Convert to JSON, RDF/Turtle, JSON-LD, N-Triples
stl convert input.stl --to json --output output.json
stl convert input.stl --to rdf --format turtle --output output.ttl

# Graph analysis (nodes, edges, centrality, cycles)
stl analyze input.stl

# Extract directed chains
stl chain input.stl
stl chain input.stl --min 1 --format json

# Build a statement from CLI
stl build "[Rain]" "[Flooding]" --mod "rule=causal,confidence=0.85"

# Clean and repair LLM output
stl clean messy_output.txt --show-repairs
stl clean messy_output.txt --schema domain.stl.schema

# Validate against a domain schema
stl schema-validate input.stl --schema medical.stl.schema

# Query with filters and field selection
stl query input.stl --where "rule=causal,confidence__gte=0.8" --select "source.name,target.name"

# Semantic diff between two files
stl diff before.stl after.stl
stl diff before.stl after.stl --format json

# Apply a diff patch
stl patch base.stl changes.json --output patched.stl
```

## Architecture

```
stl_parser/                    # 19 modules
│
├── Core Layer
│   ├── grammar.py             # Lark EBNF grammar definition
│   ├── parser.py              # parse(), parse_file() — core parser with multi-line merge
│   ├── models.py              # Pydantic v2 models (Anchor, Modifier, Statement, ParseResult)
│   ├── validator.py           # Structural and semantic validation
│   ├── errors.py              # Error hierarchy (E001-E969, W001-W099)
│   └── _utils.py              # Shared utilities (sanitize_anchor_name, etc.)
│
├── Serialization Layer
│   ├── serializer.py          # to_json(), to_dict(), to_stl(), to_rdf(), from_json()
│   ├── builder.py             # stl(), stl_doc(), StatementBuilder — programmatic construction
│   └── emitter.py             # STLEmitter — thread-safe streaming writer
│
├── Analysis Layer
│   ├── graph.py               # STLGraph — graph construction, chain extraction
│   ├── analyzer.py            # STLAnalyzer — centrality, cycles, statistics
│   └── decay.py               # Confidence decay with configurable half-life
│
├── Query Layer
│   ├── query.py               # find(), filter_statements(), select(), stl_pointer()
│   ├── diff.py                # stl_diff(), stl_patch(), diff_to_text()
│   └── reader.py              # stream_parse(), STLReader — streaming input
│
├── Validation Layer
│   ├── schema.py              # load_schema(), validate_against_schema(), .stl.schema format
│   └── llm.py                 # clean(), repair(), validate_llm_output() — 3-stage pipeline (21 repairs)
│
└── Interface Layer
    └── cli.py                 # Typer CLI with 11 commands
```

## Schema Ecosystem

Six domain-specific schemas are included in `docs/schemas/`:

| Schema | Domain | Key Constraints |
|--------|--------|----------------|
| `tcm.stl.schema` | Traditional Chinese Medicine | Unicode anchors, `domain` required |
| `scientific.stl.schema` | Scientific research | `source` required, confidence 0.3-1.0 |
| `causal.stl.schema` | Causal inference | `strength` required, rule must be "causal" |
| `historical.stl.schema` | Historical knowledge | `time` + `source` required, multi-script |
| `medical.stl.schema` | Medical/clinical | Prefixed anchors (Symptom_, Drug_, ...) |
| `legal.stl.schema` | Legal reasoning | Prefixed anchors (Law_, Regulation_, ...) |

Create custom schemas using `docs/schemas/_template.stl.schema`.

## Development

### Setup

```bash
git clone https://github.com/scos-lab/semantic-tension-language.git
cd semantic-tension-language/parser
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/                          # Run all 540+ tests
pytest tests/ --cov=stl_parser         # With coverage report
pytest tests/test_builder.py           # Single module
pytest tests/ -k "test_query"          # By name pattern
```

### Code Quality

```bash
black stl_parser/          # Format
ruff stl_parser/           # Lint
mypy stl_parser/           # Type check
```

## Documentation

- [STL Core Specification v1.0](../spec/stl-core-spec-v1.0.md)
- [STL Core Specification Supplement](../spec/stl-core-spec-v1.0-supplement.md)
- [Project Index](../docs/PROJECT_INDEX.md) — Module reference and architecture overview
- [Schema Ecosystem](../docs/schemas/) — Domain-specific schema library

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 — see [LICENSE](../LICENSE).

## Citation

```bibtex
@software{stl_parser_2025,
  author = {SCOS-Lab},
  title = {STL Parser: A Comprehensive Toolkit for Semantic Tension Language},
  year = {2025},
  version = {1.9.0},
  url = {https://github.com/scos-lab/semantic-tension-language}
}
```
