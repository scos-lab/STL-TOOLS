# Changelog

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
