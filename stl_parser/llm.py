# -*- coding: utf-8 -*-
"""
STL LLM — LLM Output Cleaning & Auto-Repair

Clean, repair, and validate raw LLM-generated STL output.
Three-stage pipeline: clean → repair → parse, with optional schema validation.

Compiled from: docs/stlc/stl_llm_v1.0.0.stlc.md

Usage:
    >>> from stl_parser.llm import validate_llm_output, clean, repair, prompt_template
    >>> result = validate_llm_output(raw_llm_text)
    >>> print(f"Valid: {result.is_valid}, Repairs: {len(result.repairs)}")
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import Statement, ParseResult, ParseError, ParseWarning
from .parser import parse
from ._utils import (
    extract_stl_fences,
    merge_multiline_statements,
    is_stl_line,
)
from .errors import STLLLMError, ErrorCode


# ========================================
# DATA MODELS
# ========================================


class RepairAction(BaseModel):
    """Records a single repair action applied to the text."""

    type: str = Field(description="Repair type identifier")
    line: Optional[int] = Field(None, description="Line number where repair was applied")
    original: str = Field(description="Original text before repair")
    repaired: str = Field(description="Text after repair")
    description: str = Field(description="Human-readable description of the repair")


class LLMValidationResult(BaseModel):
    """Result of validating LLM-generated STL output."""

    statements: List[Statement] = Field(default_factory=list)
    errors: List[ParseError] = Field(default_factory=list)
    warnings: List[ParseWarning] = Field(default_factory=list)
    is_valid: bool = True
    repairs: List[RepairAction] = Field(default_factory=list)
    cleaned_text: str = ""
    original_text: str = ""
    schema_result: Optional[Any] = Field(None, description="SchemaValidationResult if schema provided")


# ========================================
# ARROW NORMALIZATION
# ========================================

# Common LLM arrow variants to normalize
_ARROW_VARIANTS = [
    ("\u2794", "->"),   # ➔
    ("\u27a1", "->"),   # ➡
    ("\u279c", "->"),   # ➜
    ("\u2b95", "->"),   # ⮕
    ("\uff1d\uff1e", "->"),  # ＝＞ (fullwidth =>)
    ("=>", "->"),
    ("- >", "->"),
    ("\u2014>", "->"),  # —> (em-dash arrow)
    ("\u2013>", "->"),  # –> (en-dash arrow)
]


# ========================================
# COMMON TYPO CORRECTIONS
# ========================================

_MODIFIER_TYPOS = {
    "confience": "confidence",
    "confindence": "confidence",
    "confidece": "confidence",
    "strenght": "strength",
    "stength": "strength",
    "timestap": "timestamp",
    "timestmp": "timestamp",
    "authro": "author",
    "auther": "author",
    "soruce": "source",
    "souce": "source",
    "certianty": "certainty",
    "certinty": "certainty",
    "conditionalty": "conditionality",
    "intensty": "intensity",
}

# Fields with [0.0, 1.0] range
_CLAMPABLE_FIELDS = {"confidence", "certainty", "strength", "intensity"}

# Modifier fields typed as Optional[str] — if a numeric value is assigned,
# Pydantic will reject it. The repair pipeline must quote numeric values
# for these keys so they become strings.
_STR_TYPED_MODIFIER_FIELDS = {
    'time', 'duration', 'frequency', 'tense', 'location', 'domain',
    'scope', 'necessity', 'source', 'author', 'timestamp', 'version',
    'emotion', 'value', 'valence', 'alignment', 'cause', 'effect',
    'conditionality', 'intent', 'focus', 'perspective', 'mood',
    'modality', 'rule',
}


# ========================================
# CLEAN STAGE
# ========================================


def clean(raw_text: str) -> Tuple[str, List[RepairAction]]:
    """Clean raw LLM output text for STL parsing.

    Applies: fence extraction, arrow normalization, whitespace fixing,
    multi-line merging, prose stripping.

    Args:
        raw_text: Raw LLM output text

    Returns:
        Tuple of (cleaned_text, list of repair actions)

    Example:
        >>> text, repairs = clean("```stl\\n[A] => [B]\\n```")
        >>> print(text)
        [A] -> [B]
    """
    repairs = []
    text = raw_text

    # 1. Extract from code fences if present
    if re.search(r"```(?:stl)?", text):
        fenced, _meta = extract_stl_fences(text)
        if fenced.strip():
            repairs.append(RepairAction(
                type="strip_fence",
                original=text[:80] + "..." if len(text) > 80 else text,
                repaired=fenced[:80] + "..." if len(fenced) > 80 else fenced,
                description="Extracted STL from code fences",
            ))
            text = fenced

    # 2. Normalize arrows
    for variant, replacement in _ARROW_VARIANTS:
        if variant in text:
            new_text = text.replace(variant, replacement)
            if new_text != text:
                repairs.append(RepairAction(
                    type="normalize_arrow",
                    original=variant,
                    repaired=replacement,
                    description=f"Replaced arrow variant '{variant}' with '{replacement}'",
                ))
                text = new_text

    # 3. Fix whitespace
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        fixed = re.sub(r"  +", " ", line).rstrip()
        if fixed != line:
            repairs.append(RepairAction(
                type="fix_whitespace",
                original=line[:60],
                repaired=fixed[:60],
                description="Normalized whitespace",
            ))
        fixed_lines.append(fixed)
    text = "\n".join(fixed_lines)

    # 4. Merge multi-line statements
    merged = merge_multiline_statements(text)
    if merged != text:
        repairs.append(RepairAction(
            type="merge_multiline",
            original="(multi-line statements)",
            repaired="(merged to single lines)",
            description="Merged multi-line statements",
        ))
        text = merged

    # 5. Strip prose lines (keep STL and comments)
    lines = text.split("\n")
    stl_lines = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        detection = is_stl_line(line)
        if detection["is_stl"] or detection["type"] == "comment":
            stl_lines.append(line)
        elif detection["type"] in ("natural_language", "markdown_link", "markdown_list", "markdown_quote"):
            repairs.append(RepairAction(
                type="strip_prose",
                line=i + 1,
                original=line[:60],
                repaired="(removed)",
                description=f"Removed non-STL line ({detection['type']})",
            ))

    text = "\n".join(stl_lines)

    return text, repairs


# ========================================
# REPAIR STAGE
# ========================================


def repair(text: str) -> Tuple[str, List[RepairAction]]:
    """Repair common structural errors in STL text.

    Applies: bracket fixing, string quoting, modifier prefix fixing,
    value clamping, typo correction.

    Args:
        text: Cleaned STL text that may still have structural errors

    Returns:
        Tuple of (repaired_text, list of repair actions)

    Example:
        >>> text, repairs = repair("[A] -> [B] mod(confidence=1.5)")
        >>> print(text)
        [A] -> [B] ::mod(confidence=1.0)
    """
    repairs = []
    lines = text.split("\n")
    repaired_lines = []

    for i, line in enumerate(lines):
        original = line

        # 1. Fix spaces in anchor names: [Heavy Rain] → [Heavy_Rain]
        line = _fix_anchor_spaces(line, i + 1, repairs)

        # 2. Sanitize illegal characters in anchor names: & → _and_, $ → removed, etc.
        line = _fix_anchor_illegal_chars(line, i + 1, repairs)

        # 3. Truncate overlong anchor names (>64 chars)
        line = _fix_anchor_length(line, i + 1, repairs)

        # 4. Fix broken anchor brackets: [Name ::mod( → [Name] ::mod(
        line = _fix_broken_anchor_bracket(line, i + 1, repairs)

        # 5. Fix = in anchor names: [Cue=Daily_Commute] → [Cue_Daily_Commute]
        line = _fix_anchor_equals(line, i + 1, repairs)

        # 6. Fix incomplete ::mod without () — e.g. "::mod ::mod(...)"
        line = _fix_incomplete_mod(line, i + 1, repairs)

        # 6. Fix missing :: prefix on mod()
        line = _fix_mod_prefix(line, i + 1, repairs)

        # 7. Fix missing brackets on anchors
        line = _fix_missing_brackets(line, i + 1, repairs)

        # 8. Fix unclosed quotes in modifier values
        line = _fix_unclosed_quotes(line, i + 1, repairs)

        # 9. Remove orphan keys (key without =value) in modifiers
        line = _fix_orphan_keys(line, i + 1, repairs)

        # 10. Fix unquoted string values in modifiers
        line = _fix_unquoted_strings(line, i + 1, repairs)

        # 11. Fix common typos in modifier keys (before clamp, so corrected keys get clamped)
        line = _fix_typos(line, i + 1, repairs)

        # 12. Fix numeric fields that got quoted: confidence="0.95" → confidence=0.95
        line = _fix_quoted_numerics(line, i + 1, repairs)

        # 13. Clamp out-of-range numeric values
        line = _clamp_values(line, i + 1, repairs)

        repaired_lines.append(line)

    return "\n".join(repaired_lines), repairs


# Character substitutions for anchor names
_ANCHOR_CHAR_SUBS = {
    "&": "_and_",
    "+": "_plus_",
    "$": "",
    "%": "_pct_",
    "'": "",
    "\u2019": "",   # right single quote '
    "\u2018": "",   # left single quote '
    "#": "_num_",
    "@": "_at_",
    "!": "",
    "?": "",
    "(": "",
    ")": "",
    "/": "_",
    "\\": "_",
    ",": "_",
    ".": "_",
    ";": "_",
}

# Maximum allowed anchor name length
_MAX_ANCHOR_LEN = 64


def _fix_anchor_illegal_chars(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Sanitize illegal characters in anchor names.

    Replaces characters not allowed in STL identifiers (like &, $, ', +)
    with safe alternatives. Only operates on brackets BEFORE ::mod().
    """
    mod_idx = line.find("::mod(")
    if mod_idx >= 0:
        anchor_part = line[:mod_idx]
        mod_part = line[mod_idx:]
    else:
        anchor_part = line
        mod_part = ""

    def _sanitize(m: re.Match) -> str:
        name = m.group(1)
        original_name = name

        # Fix multiple colons (ratio patterns like 1:1:1) — keep only
        # single namespace colon, replace others with underscore
        colon_count = name.count(':')
        if colon_count > 1:
            # Replace all colons with underscore (namespace will be lost,
            # but multiple colons are invalid anyway)
            name = name.replace(':', '_')

        # Apply character substitutions
        for char, replacement in _ANCHOR_CHAR_SUBS.items():
            if char in name:
                name = name.replace(char, replacement)

        # Clean up resulting artifacts: multiple underscores, leading/trailing underscores
        name = re.sub(r"_+", "_", name).strip("_")

        # Ensure name starts with a word character (not digit-only after cleanup)
        if name and not re.match(r"[A-Za-z_\u4e00-\u9fff\u0600-\u06ff]", name):
            name = "_" + name

        if name != original_name:
            repairs.append(RepairAction(
                type="fix_anchor_chars",
                line=line_num,
                original=f"[{original_name}]",
                repaired=f"[{name}]",
                description=f"Sanitized illegal chars in anchor: [{original_name}] → [{name}]",
            ))

        return f"[{name}]"

    anchor_part = re.sub(r"\[([^\]]+)\]", _sanitize, anchor_part)
    return anchor_part + mod_part


def _fix_anchor_length(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Truncate anchor names exceeding 64 characters.

    Truncates at the last underscore boundary before the limit to avoid
    cutting words mid-way. Preserves ::mod() portion untouched.
    """
    mod_idx = line.find("::mod(")
    if mod_idx >= 0:
        anchor_part = line[:mod_idx]
        mod_part = line[mod_idx:]
    else:
        anchor_part = line
        mod_part = ""

    def _truncate(m: re.Match) -> str:
        name = m.group(1)
        if len(name) <= _MAX_ANCHOR_LEN:
            return m.group(0)

        # Try to truncate at last underscore before limit
        truncated = name[:_MAX_ANCHOR_LEN]
        last_underscore = truncated.rfind("_")
        if last_underscore > _MAX_ANCHOR_LEN // 2:
            truncated = truncated[:last_underscore]

        repairs.append(RepairAction(
            type="truncate_anchor",
            line=line_num,
            original=f"[{name}]",
            repaired=f"[{truncated}]",
            description=f"Truncated anchor name from {len(name)} to {len(truncated)} chars",
        ))
        return f"[{truncated}]"

    anchor_part = re.sub(r"\[([^\]]+)\]", _truncate, anchor_part)
    return anchor_part + mod_part


def _fix_broken_anchor_bracket(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix anchor names missing their closing bracket.

    Handles cases where ``]`` was lost (often due to LLM truncation):
    - ``[Name ::mod(...)`` → ``[Name] ::mod(...)``
    - ``-> [ ::mod(...)`` → removes the empty anchor (drops the line)
    - ``:: ::mod(...)`` → ``::mod(...)``
    """
    # Pattern 1: :: ::mod( (with possible spaces)
    if re.search(r'::\s+::mod\(', line):
        new_line = re.sub(r'::\s+::mod\(', '::mod(', line)
        if new_line != line:
            repairs.append(RepairAction(
                type="fix_broken_anchor",
                line=line_num,
                original=":: ::mod(",
                repaired="::mod(",
                description="Removed stray :: before ::mod(",
            ))
            line = new_line

    # Pattern 2: [Name ::mod( → [Name] ::mod(
    match = re.search(r'\[([^\]\[]+?)\s+(::mod\()', line)
    if match:
        name = match.group(1).strip()
        if name and re.match(r'^[\w\-\u4e00-\u9fff]+$', name):
            new_line = line[:match.start()] + f'[{name}] {match.group(2)}' + line[match.end():]
            repairs.append(RepairAction(
                type="fix_broken_anchor",
                line=line_num,
                original=f"[{name} ::mod(",
                repaired=f"[{name}] ::mod(",
                description=f"Inserted missing ] after anchor name '{name}'",
            ))
            line = new_line

    # Pattern 3: -> [ ::mod( → empty anchor, drop the whole statement
    if re.search(r'->\s*\[\s*::mod\(', line) or re.search(r'->\s*\[\s*\]', line):
        repairs.append(RepairAction(
            type="fix_broken_anchor",
            line=line_num,
            original=line[:60],
            repaired="(removed)",
            description="Removed statement with empty anchor name",
        ))
        return "# (removed: empty anchor)"

    return line


def _fix_anchor_equals(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Replace = in anchor names with underscore.

    Handles cases like ``[Cue=Daily_Commute]`` → ``[Cue_Daily_Commute]``.
    Only operates on brackets BEFORE ``::mod()``.
    """
    mod_idx = line.find("::mod(")
    if mod_idx >= 0:
        anchor_part = line[:mod_idx]
        mod_part = line[mod_idx:]
    else:
        anchor_part = line
        mod_part = ""

    def _fix_eq(m: re.Match) -> str:
        name = m.group(1)
        if "=" not in name:
            return m.group(0)
        fixed = name.replace("=", "_")
        repairs.append(RepairAction(
            type="fix_anchor_equals",
            line=line_num,
            original=f"[{name}]",
            repaired=f"[{fixed}]",
            description=f"Replaced = in anchor: [{name}] → [{fixed}]",
        ))
        return f"[{fixed}]"

    anchor_part = re.sub(r"\[([^\]]+)\]", _fix_eq, anchor_part)
    return anchor_part + mod_part


def _fix_incomplete_mod(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Remove incomplete ``::mod`` fragments without parentheses.

    Handles LLM-generated lines like ``[A] -> [B] ::mod ::mod(timestamp=...)``,
    where the first ``::mod`` has no ``()``. Removes the bare fragment.
    """
    # Pattern: ::mod followed by space/whitespace then ::mod( — remove the first one
    pattern = r'::mod\s+::mod\('
    if re.search(pattern, line):
        new_line = re.sub(r'::mod\s+(::mod\()', r'\1', line)
        if new_line != line:
            repairs.append(RepairAction(
                type="fix_incomplete_mod",
                line=line_num,
                original="::mod ::mod(",
                repaired="::mod(",
                description="Removed incomplete ::mod fragment (no parentheses)",
            ))
            return new_line

    # Also handle: ::mod without ( at end of anchor section (before nothing or newline)
    # e.g. "[A] -> [B] ::mod" (completely empty mod)
    pattern2 = r'::mod\s*$'
    if re.search(pattern2, line):
        new_line = re.sub(pattern2, '', line).rstrip()
        if new_line != line:
            repairs.append(RepairAction(
                type="fix_incomplete_mod",
                line=line_num,
                original="::mod",
                repaired="(removed)",
                description="Removed trailing ::mod with no content",
            ))
            return new_line

    return line


def _fix_orphan_keys(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Remove orphan modifier keys that have no ``=value``.

    Handles lines like ``::mod(confidence=0.95, description, timestamp="...")``,
    where ``description`` appears without ``=value``. The orphan key is removed
    to prevent ``_split_mod_pairs`` from merging it into the previous value.
    """
    mod_match = re.search(r'::mod\((.+)\)\s*$', line)
    if not mod_match:
        return line

    mod_content = mod_match.group(1)
    prefix = line[:mod_match.start()]

    # Find orphan keys: a comma-separated identifier not followed by =
    # Pattern: ", identifier," or ", identifier)" where identifier has no =
    fixed = re.sub(
        r',\s*([a-z_]+)\s*(?=,|\))',
        lambda m: (
            m.group(0) if '=' in mod_content[mod_content.find(m.group(1)):mod_content.find(m.group(1)) + len(m.group(1)) + 2]
            else ''
        ),
        mod_content,
    )

    # Simpler approach: split on comma, check each part for key=value pattern
    parts = []
    removed = False
    # Use _split_mod_pairs-style logic but simpler
    raw_parts = mod_content.split(',')
    i = 0
    while i < len(raw_parts):
        part = raw_parts[i].strip()
        if not part:
            i += 1
            continue

        # Check if this part has = in it (key=value)
        if '=' in part:
            # Collect continuation parts if value has commas inside quotes
            full_part = part
            # Count quotes to check if we're in an unclosed string
            while full_part.count('"') % 2 != 0 and i + 1 < len(raw_parts):
                i += 1
                full_part += ',' + raw_parts[i]
            parts.append(full_part)
        else:
            # Orphan key — no = sign. Remove it.
            removed = True
            repairs.append(RepairAction(
                type="remove_orphan_key",
                line=line_num,
                original=part,
                repaired="(removed)",
                description=f"Removed orphan modifier key '{part}' (no value assigned)",
            ))
        i += 1

    if removed:
        new_mod = ", ".join(parts)
        return prefix + "::mod(" + new_mod + ")"

    return line


def _fix_unclosed_quotes(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix unclosed quote in modifier string values.

    Detects lines where a modifier string value (e.g. description="...")
    is missing its closing quote, causing the next key=value pair to be
    absorbed into the string. Fixes by inserting a closing quote before
    the next ``key=`` pattern.

    Common LLM error:
        description="text with comma, timestamp="2023-01-01"
        → description="text with comma", timestamp="2023-01-01"
    """
    mod_match = re.search(r'::mod\(', line)
    if not mod_match:
        return line

    mod_start = mod_match.end()
    mod_content = line[mod_start:]

    # Quick check: if quotes are balanced, nothing to fix
    quote_count = mod_content.count('"')
    if quote_count % 2 == 0:
        return line

    # Find unclosed quote: an opening " whose content runs into a `key=` pattern
    # Pattern: value starts with ", has no closing ", but then we see `, identifier=`
    # Fix: insert " before the `, identifier=` part
    fixed = re.sub(
        r'(="[^"]*?)(,\s*)((?:timestamp|confidence|rule|strength|source|author|'
        r'domain|description|certainty|lesson|cause|effect|time|intent|version|'
        r'conditionality|focus|perspective|mood|modality|emotion|intensity|'
        r'valence|priority|scope|frequency|duration|tense|location|necessity|'
        r'alignment|value|path)\s*=)',
        r'\1"\2\3',
        mod_content,
    )

    if fixed != mod_content:
        new_line = line[:mod_start] + fixed
        repairs.append(RepairAction(
            type="fix_unclosed_quote",
            line=line_num,
            original=line[mod_start:mod_start + 60] + "...",
            repaired=fixed[:60] + "...",
            description="Inserted missing closing quote in modifier value",
        ))
        return new_line

    return line


def _fix_anchor_spaces(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix spaces in anchor names: [Heavy Rain] → [Heavy_Rain].

    LLMs frequently generate natural-language anchor names with spaces.
    STL anchors only allow [A-Za-z0-9_-:], so spaces must be replaced.

    Only fixes brackets BEFORE ::mod() — brackets inside mod values
    (e.g. JSON arrays) must not be touched.
    """
    # Split at ::mod( to only process the anchor part
    mod_idx = line.find("::mod(")
    if mod_idx >= 0:
        anchor_part = line[:mod_idx]
        mod_part = line[mod_idx:]
    else:
        anchor_part = line
        mod_part = ""

    def _replace_spaces(m: re.Match) -> str:
        name = m.group(1)
        if " " not in name:
            return m.group(0)
        fixed = re.sub(r"\s+", "_", name.strip())
        repairs.append(RepairAction(
            type="fix_anchor_spaces",
            line=line_num,
            original=f"[{name}]",
            repaired=f"[{fixed}]",
            description=f"Replaced spaces in anchor: [{name}] → [{fixed}]",
        ))
        return f"[{fixed}]"

    anchor_part = re.sub(r"\[([^\]]+)\]", _replace_spaces, anchor_part)
    return anchor_part + mod_part


def _fix_mod_prefix(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix missing :: prefix on mod()."""
    # Match 'mod(' not preceded by '::'
    pattern = r"(?<!:)(?<!:)\bmod\("
    if re.search(pattern, line):
        new_line = re.sub(pattern, "::mod(", line)
        if new_line != line:
            repairs.append(RepairAction(
                type="fix_mod_prefix",
                line=line_num,
                original="mod(",
                repaired="::mod(",
                description="Added :: prefix to mod()",
            ))
            return new_line
    return line


def _fix_missing_brackets(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix missing brackets around anchor names."""
    # Pattern: word(s) -> [Target] or [Source] -> word(s)
    # Only fix if there's an arrow and at least one bracket pair

    if "->" not in line and "\u2192" not in line:
        return line

    arrow = "->" if "->" in line else "\u2192"
    parts = line.split(arrow, 1)
    if len(parts) != 2:
        return line

    left = parts[0].strip()
    right_full = parts[1]

    # Check right side for ::mod
    right_parts = right_full.split("::mod", 1)
    right = right_parts[0].strip()
    mod_suffix = "::mod" + right_parts[1] if len(right_parts) > 1 else ""

    changed = False

    # Fix left side if no brackets
    if left and not left.startswith("["):
        # Only fix if it looks like an anchor name (word chars, possibly with namespace)
        if re.match(r"^[\w:]+$", left):
            new_left = f"[{left}]"
            repairs.append(RepairAction(
                type="add_brackets",
                line=line_num,
                original=left,
                repaired=new_left,
                description=f"Added brackets to source anchor '{left}'",
            ))
            left = new_left
            changed = True

    # Fix right side if no brackets
    if right and not right.startswith("["):
        if re.match(r"^[\w:]+$", right):
            new_right = f"[{right}]"
            repairs.append(RepairAction(
                type="add_brackets",
                line=line_num,
                original=right,
                repaired=new_right,
                description=f"Added brackets to target anchor '{right}'",
            ))
            right = new_right
            changed = True

    if changed:
        if mod_suffix:
            return f"{left} {arrow} {right} {mod_suffix}"
        return f"{left} {arrow} {right}"

    return line


def _fix_unquoted_strings(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Normalize and quote values in ::mod().

    Uses a smart tokenizer that respects bracket/brace/paren nesting and
    identifies key=value boundaries by looking for `identifier=` patterns
    at depth 0. Handles:
    - Unquoted multi-word strings: location=San Francisco → location="San Francisco"
    - Bare list values: values=1,2,3, bins=5 → values="1,2,3", bins=5
    - JSON arrays/objects: items=["a","b"] → items='["a","b"]'
    - Single quotes: name='test' → name="test"
    - Boolean normalization: True → true
    - Unclosed mod parentheses
    """
    # Find ::mod(...) content
    mod_match = re.search(r'::mod\((.+)\)\s*$', line)
    if not mod_match:
        # Try unclosed mod: ::mod(... without closing )
        mod_match = re.search(r'::mod\((.+)$', line)
        if not mod_match:
            return line
        # Close it
        line = line.rstrip() + ")"
        mod_match = re.search(r'::mod\((.+)\)\s*$', line)
        if not mod_match:
            return line

    mod_content = mod_match.group(1)
    prefix = line[:mod_match.start()]

    # Smart-tokenize into key=value pairs
    pairs = _split_mod_pairs(mod_content)
    if not pairs:
        return line

    new_pairs = []
    changed = False

    for key, val in pairs:
        val = val.strip()
        original_pair = f"{key}={val}"

        # Already double-quoted → keep
        if val.startswith('"') and val.endswith('"'):
            new_pairs.append(original_pair)
            continue

        # Single-quoted → convert to double
        if val.startswith("'") and val.endswith("'"):
            fixed_val = '"' + val[1:-1] + '"'
            repairs.append(RepairAction(
                type="fix_single_quotes",
                line=line_num,
                original=original_pair,
                repaired=f'{key}={fixed_val}',
                description=f"Converted single quotes to double quotes for '{key}'",
            ))
            new_pairs.append(f'{key}={fixed_val}')
            changed = True
            continue

        # JSON array [...] → quote the whole thing
        if val.startswith('[') and val.endswith(']'):
            inner = val[1:-1].replace('"', '\\"')
            fixed = f'{key}="[{inner}]"'
            repairs.append(RepairAction(
                type="quote_json_array",
                line=line_num,
                original=original_pair,
                repaired=fixed,
                description=f"Quoted JSON array value for '{key}'",
            ))
            new_pairs.append(fixed)
            changed = True
            continue

        # JSON object {...} → quote
        if val.startswith('{') and val.endswith('}'):
            inner = val[1:-1].replace('"', '\\"')
            fixed = f'{key}="{{{inner}}}"'
            repairs.append(RepairAction(
                type="quote_json_object",
                line=line_num,
                original=original_pair,
                repaired=fixed,
                description=f"Quoted JSON object value for '{key}'",
            ))
            new_pairs.append(fixed)
            changed = True
            continue

        # Tuple (...) → quote
        if val.startswith('(') and val.endswith(')'):
            inner = val[1:-1].replace('"', '\\"')
            fixed = f'{key}="({inner})"'
            repairs.append(RepairAction(
                type="quote_tuple",
                line=line_num,
                original=original_pair,
                repaired=fixed,
                description=f"Quoted tuple value for '{key}'",
            ))
            new_pairs.append(fixed)
            changed = True
            continue

        # Pure numeric — but quote if it's a str-typed Modifier field
        # (e.g. time=5 → time="5", otherwise Pydantic rejects int for Optional[str])
        if re.match(r'^-?\d+\.?\d*$', val):
            if key in _STR_TYPED_MODIFIER_FIELDS:
                fixed = f'{key}="{val}"'
                repairs.append(RepairAction(
                    type="quote_reserved_field",
                    line=line_num,
                    original=original_pair,
                    repaired=fixed,
                    description=f"Quoted numeric value for str-typed field '{key}'",
                ))
                new_pairs.append(fixed)
                changed = True
            else:
                new_pairs.append(original_pair)
            continue

        # Boolean — normalize to lowercase
        if val.lower() in ('true', 'false'):
            if val != val.lower():
                repairs.append(RepairAction(
                    type="fix_boolean_case",
                    line=line_num,
                    original=original_pair,
                    repaired=f'{key}={val.lower()}',
                    description=f"Normalized boolean case for '{key}': {val} → {val.lower()}",
                ))
                changed = True
            new_pairs.append(f'{key}={val.lower()}')
            continue

        # Contains comma → bare list value, quote it
        if ',' in val:
            fixed = f'{key}="{val}"'
            repairs.append(RepairAction(
                type="quote_list_value",
                line=line_num,
                original=original_pair,
                repaired=fixed,
                description=f"Quoted comma-separated list value for '{key}'",
            ))
            new_pairs.append(fixed)
            changed = True
            continue

        # Unquoted string (has spaces, special chars, or is just text)
        fixed = f'{key}="{val}"'
        repairs.append(RepairAction(
            type="quote_string",
            line=line_num,
            original=original_pair,
            repaired=fixed,
            description=f"Quoted unquoted string value for '{key}'",
        ))
        new_pairs.append(fixed)
        changed = True

    if changed:
        return prefix + "::mod(" + ", ".join(new_pairs) + ")"

    # Even if no changes, reconstruct with proper tokenization
    # (fixes cases where original comma splitting was ambiguous)
    reconstructed = prefix + "::mod(" + ", ".join(f"{k}={v.strip()}" for k, v in pairs) + ")"
    if reconstructed != line:
        return reconstructed

    return line


def _split_mod_pairs(content: str) -> List[Tuple[str, str]]:
    """Split mod() content into (key, value) pairs, respecting nesting.

    Instead of naively splitting on commas, identifies key=value boundaries
    by looking for `identifier=` patterns at bracket-depth 0. Handles:
    - Nested brackets: items=["a","b"] stays as one value
    - Nested braces: area={"w": 20} stays as one value
    - Nested parens: teams=("A","B") stays as one value
    - Quoted strings with commas: desc="a, b" stays as one value
    - Bare lists: values=1,2,3, bins=5 → values="1,2,3" and bins=5
    - Mid-value apostrophes: text=It's great → not confused as quote start

    Returns:
        List of (key, value) tuples.
    """
    # Find all key= positions at depth 0
    key_positions = []  # (start_of_key, end_of_equals, key_name)

    i = 0
    depth_bracket = 0   # []
    depth_brace = 0     # {}
    depth_paren = 0     # ()
    in_quote = False
    quote_char = None

    while i < len(content):
        c = content[i]

        # Track quotes — only enter quote mode if the quote follows '='
        # (possibly with whitespace). This prevents mid-value apostrophes
        # like "It's" from triggering quote mode.
        if c in ('"', "'") and (i == 0 or content[i-1] != '\\'):
            if not in_quote:
                # Check if this quote is right after = (value start)
                j = i - 1
                while j >= 0 and content[j] == ' ':
                    j -= 1
                if j >= 0 and content[j] == '=':
                    in_quote = True
                    quote_char = c
                # else: stray quote in unquoted value — ignore
            elif c == quote_char:
                in_quote = False
            i += 1
            continue

        if in_quote:
            i += 1
            continue

        # Track nesting
        if c == '[':
            depth_bracket += 1
        elif c == ']':
            depth_bracket = max(0, depth_bracket - 1)
        elif c == '{':
            depth_brace += 1
        elif c == '}':
            depth_brace = max(0, depth_brace - 1)
        elif c == '(':
            depth_paren += 1
        elif c == ')':
            depth_paren = max(0, depth_paren - 1)

        # At depth 0, look for key= pattern
        if depth_bracket == 0 and depth_brace == 0 and depth_paren == 0:
            if c == '=' and i > 0:
                # Walk back to find key name start
                j = i - 1
                while j >= 0 and content[j] == ' ':
                    j -= 1
                key_end = j + 1
                while j >= 0 and re.match(r'[\w]', content[j]):
                    j -= 1
                key_start = j + 1

                if key_start < key_end:
                    key_name = content[key_start:key_end]
                    key_positions.append((key_start, i + 1, key_name))

        i += 1

    if not key_positions:
        return []

    # Extract values: from end_of_equals to start_of_next_key (minus comma/space)
    pairs = []
    for idx, (key_start, val_start, key_name) in enumerate(key_positions):
        if idx + 1 < len(key_positions):
            val_end = key_positions[idx + 1][0]
            # Strip trailing comma and whitespace
            raw_val = content[val_start:val_end].rstrip()
            if raw_val.endswith(','):
                raw_val = raw_val[:-1].rstrip()
        else:
            raw_val = content[val_start:].rstrip()

        pairs.append((key_name, raw_val))

    return pairs


def _fix_quoted_numerics(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix numeric modifier fields that were incorrectly quoted.

    Handles cases like ``confidence="0.95"`` → ``confidence=0.95``
    and ``confidence=""`` → removes the empty field.
    Also cleans trailing commas inside quotes: ``confidence="0.95,"`` → ``confidence=0.95``.
    """
    for field in _CLAMPABLE_FIELDS:
        # Match field="..." pattern
        pattern = rf'({field}\s*=\s*)"([^"]*)"'
        match = re.search(pattern, line)
        if match:
            prefix = match.group(1)
            inner = match.group(2).strip().rstrip(',').strip()

            if not inner:
                # Empty value — remove the whole field
                # Remove "field="" " and any surrounding comma
                line = re.sub(rf',?\s*{field}\s*=\s*""', '', line)
                line = re.sub(rf'{field}\s*=\s*""\s*,?\s*', '', line)
                repairs.append(RepairAction(
                    type="remove_empty_numeric",
                    line=line_num,
                    original=f'{field}=""',
                    repaired="(removed)",
                    description=f"Removed empty {field} field",
                ))
            elif re.match(r'^-?\d+\.?\d*$', inner):
                # Valid number inside quotes — unquote
                new_val = f'{prefix}{inner}'
                line = line[:match.start()] + new_val + line[match.end():]
                repairs.append(RepairAction(
                    type="unquote_numeric",
                    line=line_num,
                    original=f'{field}="{match.group(2)}"',
                    repaired=f'{field}={inner}',
                    description=f"Unquoted numeric value for '{field}'",
                ))

    # Clean up leading comma after removed fields: ::mod(, key=...) → ::mod(key=...)
    line = re.sub(r'::mod\(\s*,\s*', '::mod(', line)

    return line


def _clamp_values(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Clamp out-of-range numeric values for known [0.0, 1.0] fields."""
    for field in _CLAMPABLE_FIELDS:
        pattern = rf"({field}\s*=\s*)(-?\d+\.?\d*)"
        match = re.search(pattern, line)
        if match:
            val = float(match.group(2))
            if val > 1.0:
                clamped = "1.0"
                repairs.append(RepairAction(
                    type="clamp_value",
                    line=line_num,
                    original=f"{field}={match.group(2)}",
                    repaired=f"{field}={clamped}",
                    description=f"Clamped {field} from {val} to {clamped}",
                ))
                line = line[:match.start()] + f"{field}={clamped}" + line[match.end():]
            elif val < 0.0:
                clamped = "0.0"
                repairs.append(RepairAction(
                    type="clamp_value",
                    line=line_num,
                    original=f"{field}={match.group(2)}",
                    repaired=f"{field}={clamped}",
                    description=f"Clamped {field} from {val} to {clamped}",
                ))
                line = line[:match.start()] + f"{field}={clamped}" + line[match.end():]
    return line


def _fix_typos(line: str, line_num: int, repairs: List[RepairAction]) -> str:
    """Fix common typos in modifier keys."""
    # Only fix within ::mod() context
    mod_match = re.search(r"::mod\((.+)\)", line)
    if not mod_match:
        return line

    mod_content = mod_match.group(1)
    new_content = mod_content

    for typo, correct in _MODIFIER_TYPOS.items():
        pattern = rf"\b{typo}\b"
        if re.search(pattern, new_content):
            repairs.append(RepairAction(
                type="fix_typo",
                line=line_num,
                original=typo,
                repaired=correct,
                description=f"Fixed typo: '{typo}' → '{correct}'",
            ))
            new_content = re.sub(pattern, correct, new_content)

    if new_content != mod_content:
        prefix = line[:mod_match.start()]
        suffix = line[mod_match.end():]
        return prefix + "::mod(" + new_content + ")" + suffix

    return line


# ========================================
# MAIN PIPELINE
# ========================================


def validate_llm_output(
    raw_text: str,
    schema=None,
) -> LLMValidationResult:
    """Full pipeline: clean → repair → parse → optional schema validation.

    Args:
        raw_text: Raw LLM output text
        schema: Optional STLSchema for schema validation

    Returns:
        LLMValidationResult with statements, errors, repairs

    Example:
        >>> result = validate_llm_output("```stl\\n[A] => [B]\\n```")
        >>> print(result.is_valid)
        True
    """
    # Stage 1: Clean
    cleaned_text, clean_repairs = clean(raw_text)

    # Stage 2: Repair
    repaired_text, repair_repairs = repair(cleaned_text)

    # Stage 3: Parse
    parse_result = parse(repaired_text)

    # Stage 4: Optional schema validation
    schema_result = None
    if schema is not None:
        try:
            from .schema import validate_against_schema
            schema_result = validate_against_schema(parse_result, schema)
        except ImportError:
            pass

    # Compile result
    all_repairs = clean_repairs + repair_repairs

    return LLMValidationResult(
        statements=parse_result.statements,
        errors=parse_result.errors,
        warnings=parse_result.warnings,
        is_valid=parse_result.is_valid and (schema_result is None or schema_result.is_valid),
        repairs=all_repairs,
        cleaned_text=repaired_text,
        original_text=raw_text,
        schema_result=schema_result,
    )


# ========================================
# PROMPT TEMPLATE
# ========================================

_BASE_TEMPLATE = """Generate valid STL (Semantic Tension Language) statements using this syntax:

[Source_Anchor] -> [Target_Anchor] ::mod(key=value, key=value, ...)

Rules:
- Anchors: Use [BracketedNames] with PascalCase or underscore_separation
- Arrows: Use -> between source and target
- Modifiers: Optional, prefixed with ::mod(...)
- String values must be quoted: rule="causal"
- Numeric values: confidence=0.95 (no quotes)
- Boolean values: deterministic=true (no quotes)

Common modifier keys:
- confidence: float [0.0-1.0] — how certain is this relation
- rule: string — "causal", "logical", "empirical", "definitional"
- source: string — reference URI or citation
- author: string — who established this relation
- timestamp: string — ISO 8601 datetime
- strength: float [0.0-1.0] — causal strength
- domain: string — knowledge domain

Example:
[Theory_Relativity] -> [Prediction_TimeDilation] ::mod(rule="logical", confidence=0.99)
"""


def prompt_template(schema=None) -> str:
    """Generate an STL instruction prompt for LLMs.

    Args:
        schema: Optional STLSchema to add schema-specific constraints

    Returns:
        Prompt template string

    Example:
        >>> prompt = prompt_template()
        >>> # Use as system message for LLM
    """
    template = _BASE_TEMPLATE

    if schema is not None:
        template += "\n\nSchema-specific constraints:\n"
        template += f"Schema: {schema.name} v{schema.version}\n"

        if schema.namespace:
            template += f"- Default namespace: {schema.namespace}\n"

        if schema.modifier.required_fields:
            template += f"- Required modifier fields: {', '.join(schema.modifier.required_fields)}\n"

        for field_name, fc in schema.modifier.field_constraints.items():
            if fc.type == "enum" and fc.enum_values:
                template += f"- {field_name}: must be one of {fc.enum_values}\n"
            elif fc.type in ("float", "integer"):
                parts = [f"- {field_name}: {fc.type}"]
                if fc.min_value is not None:
                    parts.append(f"min={fc.min_value}")
                if fc.max_value is not None:
                    parts.append(f"max={fc.max_value}")
                template += " ".join(parts) + "\n"

        if schema.source_anchor.pattern:
            template += f"- Source anchor names must match: /{schema.source_anchor.pattern}/\n"

        if schema.target_anchor.pattern:
            template += f"- Target anchor names must match: /{schema.target_anchor.pattern}/\n"

    return template
