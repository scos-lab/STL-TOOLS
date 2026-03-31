# Changelog

## v1.9.0 (2026-03-31)

### Chain Extraction — Visualize Node Relationships

New feature: extract and display all directed chains from STL statements,
making transitive node relationships immediately visible to LLMs and humans.

**New: `STLGraph.extract_chains(min_length=2)`** — finds all maximal directed
paths in the graph, returning chains like `[A] → [B] → [C] → [D]`.

**New: `STLGraph.format_chains(chains)`** — formats chains as readable text.

**New: `extract_chains(parse_result)` convenience function** — top-level API.

**New CLI: `stl chain <file> [--min N] [--format text|json]`** — extract and
display chains from any STL file.

### LLM Repair Pipeline — 3 New Auto-Repair Functions

Strengthened the `validate_llm_output()` pipeline to handle real-world LLM output
quality issues discovered in the LongMemEval dataset (940 STL files).

**8 new auto-repair functions in the LLM pipeline (`validate_llm_output`):**

| Function | Fixes |
|----------|-------|
| `_fix_anchor_illegal_chars()` | `&` → `_and_`, `$` → removed, `'` → removed, `+` → `_plus_`, `%` → `_pct_`, multi-colon ratios |
| `_fix_anchor_length()` | Truncates names >64 chars at underscore boundary |
| `_fix_broken_anchor_bracket()` | `[Name ::mod(` → `[Name] ::mod(`, removes empty anchors |
| `_fix_anchor_equals()` | `[Cue=Value]` → `[Cue_Value]` |
| `_fix_incomplete_mod()` | `::mod ::mod(...)` → `::mod(...)`, removes bare `::mod` |
| `_fix_unclosed_quotes()` | Inserts missing `"` before next `key=` in modifier values |
| `_fix_orphan_keys()` | Removes modifier keys without `=value` (e.g. stray `description`) |
| `_fix_quoted_numerics()` | `confidence="0.95"` → `confidence=0.95`, removes `confidence=""` |

**New: `STLGraph.from_networkx(graph)`** — factory method to wrap an existing
NetworkX DiGraph/MultiDiGraph, enabling external systems (e.g. STG) to reuse
STLGraph's analysis methods without converting to ParseResult first.

### STG Integration

`propagate` command now auto-displays chains with full STL edge details:
- Activated subgraph → `STLGraph.from_networkx()` → `extract_chains()`
- Each chain shows node activation flow + every edge's STL statement
- Deduplication removes chains with >70% node overlap

**Impact on LongMemEval (940 STL files):**
- Parse success rate: 77.4% → **100.0%**
- All 212 originally broken files auto-repaired
- Chain extraction: 10,417 → **13,253** chains (+27.2%)

### Bugfix: stl_parser path conflict

Removed stale `stl_parser` 1.7.0 copy in `website factory/.../src/` that was
shadowing the editable install via `_cortex.pth`. The `stl` CLI now correctly
loads the latest version.

## v1.8.3 (2026-03-27)

### Smart Mod Tokenizer — LLM Tool Call Support

Rewrote `_fix_unquoted_strings()` in the LLM repair pipeline with a smart tokenizer
that correctly handles complex `::mod()` values from small LLMs.

**New: `_split_mod_pairs()`** — splits mod content by tracking `[]{}()` nesting depth
and identifying `key=value` boundaries, instead of naively splitting on commas.

**Fixes:**
- Unquoted multi-word strings: `location=San Francisco` → `location="San Francisco"`
- Bare comma-separated lists: `values=1,2,3, bins=5` → `values="1,2,3", bins=5`
- JSON arrays as values: `items=["a","b"]` → properly quoted
- JSON objects as values: `area={"w": 20}` → properly quoted
- Tuples as values: `teams=("A","B")` → properly quoted
- Boolean case normalization: `True/False` → `true/false`
- Str-typed modifier fields with numeric values: `time=5` → `time="5"`
  (prevents Pydantic type rejection for fields like `time`, `value`, `duration`)
- Mid-value apostrophes: `text=It's great` no longer triggers false quote mode
- `_fix_anchor_spaces()` no longer modifies brackets inside `::mod()` values

**Impact:** BFCL Simple benchmark (400 cases) with qwen2.5:7b improved from 83% → 100%.

## v1.8.2 (2026-03-26)

- `_fix_anchor_spaces()`: `[Heavy Rain]` → `[Heavy_Rain]`
- `_fix_single_quotes()`: `name='test'` → `name="test"`
- Repair ordering: typo fix before clamp
- `stltoolcall.py`: single-anchor tool call detection, trailing junk removal

## v1.8.1 (2026-03-24)

- Initial LLM repair pipeline (`clean` → `repair` → `parse`)
- Arrow normalization, bracket fixing, modifier prefix fixing
- Typo correction, value clamping
