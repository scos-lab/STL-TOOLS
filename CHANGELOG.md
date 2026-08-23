# Changelog

## v1.11.0 (2026-08-24)

### Cross-statement requirement rule + identity resolver hook (contributed by Scorpse)

- New `require { action / binding / outcome / independent / resolver }` schema block
  (`RequirementRule`): a statement with `action == <trigger>` is valid only if the
  document also contains a binding statement (`action == <binding>`, optional
  `outcome`), optionally independent (`verifier != claimant`) and passing a named
  identity resolver. Unsatisfiable requirements **fail closed** with error `E612`
  naming exactly what is missing — the structural fix for the "gate required from a
  role bound to nobody" deadlock reported on discussion #8.
- `validate_against_schema` / `validate_against_profiles` gain an optional
  `resolvers={name: fn}` parameter; the identity registry stays external to the
  engine (orchestrator-agnostic).
- Follow-up (maintainers): registered `E612_SCHEMA_REQUIREMENT` in the public
  error-code registry with message + suggestion; widened the "every code has a
  public message" test to cover it.

## v1.10.2 (2026-08-24)

- `STLGraph.extract_chains` gains optional bounds: `max_paths` (stop after N
  distinct chains), `cutoff` (maximum path length passed to the enumerator) and
  `time_budget_s` (wall-clock budget); `graph.last_chains_truncated` reports
  whether a bound cut the enumeration short. Default behaviour is unchanged
  (unbounded). Motivation: simple-path enumeration is combinatorial on dense
  subgraphs — a multi-seed recall query over a ~10k-edge STG graph spun for
  minutes in `all_simple_paths`; callers can now bound the work.

## v1.10.1 (2026-08-21)

- Fix: `validate_datetime_format` rejected `Z`-suffixed timestamps on Python 3.9
  and 3.10 (`datetime.fromisoformat` only accepts a trailing `Z` from 3.11); the
  `Z` is now normalised to `+00:00` before parsing, matching `decay.py` and the
  schema engine. Present since 1.7.0; surfaced by the first CI run that actually
  executed the test suite (see the 1.10.0 CI note).
- CI: matrix `fail-fast: false`, so one interpreter's failure no longer cancels
  the rest of the matrix.

## v1.10.0 (2026-08-21)

### CI

- `ci.yml`: the lint step invoked `ruff stl_parser/`, which newer ruff rejects
  ("unrecognized subcommand"); with fail-fast across the matrix this cancelled
  every job, so **tests had not actually run in CI since v1.8.x**. Now
  `ruff check`; lint / black / mypy / Codecov steps are `continue-on-error`
  (pre-existing lint baseline of ~110 ruff findings is tracked separately),
  so the job's colour reflects the test suite. GitHub Actions bumped
  (setup-python v5, upload-artifact v4, codecov-action v4).

### Schema Engine Hardening (contributed by Scorpse)

Ported from Scorpse's fork of `scos-lab/semantic-tension-language`
(branch `feature/software-schema-hardening`), with permission. Design notes
and original commits by Scorpse (_AdrianTeo).

- **Fail-closed schema parsing**: unknown top-level blocks, anchor keys,
  constraint keys, and modifier field types now raise `E602` instead of being
  silently skipped.
- **Strict primitives**: `integer` fields accept only Python ints (not bool or
  float); `boolean` fields must be bool; `datetime` fields must be ISO 8601
  strings. String fields accept a regex constraint
  (`identifier: string(/[A-Z]+-[0-9]+/)`), matched with `re.fullmatch`.
- **Typed edge rules**: repeatable
  `edge { source: [...] relation: [...] target: [...] }` blocks
  (`SchemaEdgeRule`). When a schema declares edge rules, every statement must
  match at least one source-type / relation / target-type triple (`E611`);
  schemas without edge blocks behave exactly as before.
- **Graph constraints enforced**: `max_chain_length` (`E608`) and
  `allow_cycles: false` (`E609`) are now enforced (previously parsed but not
  checked).
- **Profiles**: `load_profile(path)` loads a `.stl.profile` manifest into a
  `Dict[namespace, STLSchema]` (missing files, missing or duplicate namespaces
  are errors); `validate_against_profiles(parse_result, profiles)` routes each
  statement by source namespace or unique anchor-prefix match, validates
  cross-namespace targets against every registered profile, applies per-profile
  `min/max_statements` to the routed subset, and applies the strictest
  composite graph constraints (cycles rejected if any profile forbids them,
  smallest `max_chain_length` wins).
- **Pydantic fidelity**: `to_pydantic` maps enum to `Literal`, datetime to
  `datetime`, integer to strict `int`, string pattern to a `pattern`
  constraint; `from_pydantic` recognises `Literal` and `datetime` annotations.
- **New error codes** `E604`-`E611` with messages and suggestions. Existing
  schema validation paths now emit the specific codes: document count `E605`,
  anchor namespace/pattern `E606`, missing required field `E607`, field type
  mismatch `E604` (range/enum/pattern violations remain `E603`).
- `load_schema()` on a missing `.schema` / `.stl.schema` path raises `E400`
  file-not-found instead of parsing the path string as schema text.
- `W002` ("many digits") no longer fires for structured anchor names that
  contain `_` or `-` (generated identifiers).
- `analyzer.py`: add the missing `import re` (reachable `NameError` in anchor
  type inference).
- `__version__` is read from the installed distribution metadata, falling back
  to the project version for uninstalled source checkouts.
- New public exports: `load_profile`, `validate_against_profiles`.

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
