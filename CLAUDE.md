# Claude Code Instructions

## Spelling: American English everywhere

- Use American English in everything you write: code, identifiers, comments, docstrings, docs, commit messages, PR and issue text. "Color", "neighbor", "meter", "center", "gray", "labeled", "canceled", "catalog", "penalize".
- The codebase has hundreds of older British spellings ("colour", "neighbour", "metres", "centre"). Don't copy them from the code around you. Keep one only where renaming would break an interface (a CLI flag, a JSON key, a file name).
- CI checks the lines a PR adds (`scripts/check_spelling.sh`, codespell's British-to-American list); run it before pushing.

## Pull Requests

- Don't commit directly to `main`. Always go through a pull request.
- Don't merge PRs unless explicitly told to.
- Don't include issue numbers in PR titles. Put them in the body instead.
- Use GitHub keyword phrases like "Fixes #123" or "Resolves #234" to make PRs automatically close issues.
- Don't hard-wrap PR bodies, issue bodies or comments: GitHub renders each newline. Hard-wrap commit messages and code comments only.
- Put figures in issues and PRs via the `issue-images` branch (see the `issue-images` skill), never as local paths.

## Running things locally

- Run one heavy job at a time on the laptop, with the default (serial) `--num-workers`. Never run two OCR or fit jobs at once; it runs out of memory.
- Never put a recursive S3 listing (`aws s3 ls --recursive`) in a polling loop: listings are billed as egress.
- The Mac's filesystem is case-insensitive; Batch's Linux is not. Build file paths from a file's real stem, never from a normalized (lowercased, zero-stripped) page key: `p100l.jpg` finds `p100L.jpg` locally and nothing on Batch (#647).
- Skills in `.claude/skills/` cover recurring procedures: `background-jobs` (launching and waiting on long jobs on macOS), `truth-ab` (A/B runs on OIM-truth volumes), `local-volume` (pulling a corpus item into data/), `issue-images`.

## Coding guidelines

- Avoid abbrevs: `const startDate = r.tags['start_date']`, not `const sd = ...`.
  - Exceptions: single-letter variables are fine if they are short-lived. `i` is always fine as an index.
- Don't sweat access control. There's no need for `private` declarations or `_`-prefixed variables. Use `varname`, not `_varname`.
- Factor out helper functions where reasonable. Write at least a one-line documentation comment (`//` in TypeScript, `#` in Python) for internal-only functions, and full jsdoc/docstrings for exported functions.
- Write unit tests for all public functions.
- Functions should never take more than four positional arguments. To avoid this factor out dataclasses/interfaces, use keyword-only arguments or an options object, or group x/y parameters into  `(x, y)` tuples (or lat/lng, width/height, etc.).
- All boolean variables (function parameters, command-line flags) should default to `false`.

### Python

This project uses `uv` for package management and running scripts. The `uv` binary can be found in `~/.local/bin/uv`.

Write type hints for all function parameters, and local variables where pyright cannot infer a type. Write return types for functions, unless they're very complicated and can be inferred.

Provide generic type parameters where types take them: don't use plain `dict`, use `dict[str, str]` if that's what the type is.

Don't include explicit type annotations where they would exactly match what pyright infers. For example:

```diff
- lines: list[str] = ["a", "b", "c"]
+ lines = ["a", "b", "c"]
- tags: dict[str, str] = {"key1": "value1", "key2": val2}
+ tags = {"key1": "value1", "key2": val2}
```

Assume all functions are importable. There's no need to `_`-prefix functions.

### TypeScript

Use the following format for JSDoc comments:

```ts
/** one liner */

/**
 * One line summary.
 *
 * details
 * more details
 */
```

## After every change

### Python files

Run ruff to format and lint after editing any `.py` file, and run `pyright` to check for type errors:

```
uv run ruff format <file>
uv run ruff check --fix <file>
uv run pyright
```

Read these gates' full output: never pipe them through `tail` or `head`, which once hid two real findings and failed CI. Before pushing, run them repo-wide as CI does: `uv run ruff format --check`, `uv run ruff check`, `uv run pyright`, `uv run pytest`.

### TypeScript / frontend files

Run prettier after editing any file under `app/`:

```
cd app && npx prettier --write <file>
```
