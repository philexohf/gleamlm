# Coding Standards

## General

- Python 3.10+ target.
- 4-space indentation, LF line endings, UTF-8.
- Lines wrapped at 100 characters where practical.

## Formatting

Automated via **ruff** (`ruff format`). No manual formatting decisions.

## Linting

Enforced by **ruff** with rules: E/W/F (pycodestyle + pyflakes), I (import order), N (naming), UP (pyupgrade), B (bugbear), SIM (simplify).

Imports are grouped: stdlib, third-party, first-party (`gleamlm`). Sorted by `ruff` (isort-compatible).

## Type annotations

- The shared library `gleamlm/` requires full type annotations on all public functions and methods. Checked by **mypy** in strict mode.
- Uses `from __future__ import annotations` for forward references.
- Wrapper projects (`gleamlm-nano/`, scripts, tools) should add annotations gradually.

## Naming

| Kind | Convention |
|------|-----------|
| Modules | `lower_with_under.py` |
| Classes | `PascalCase` |
| Functions / methods | `lower_with_under()` |
| Constants | `UPPER_WITH_UNDER` |
| Private members | `_leading_underscore` |

## Testing

- `pytest` with fixtures in `conftest.py`.
- Unit tests in `tests/` cover the core `gleamlm/` library.
- Model-specific smoke tests live in their variant directories.

## No-comment rule

Don't add comments that explain *what* the code does --- the code should be self-documenting. Comments are for *why*.
