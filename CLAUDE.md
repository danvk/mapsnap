# Claude Code Instructions

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

Run ruff to format and lint after editing any `.py` file:

```
uv run ruff format <file>
uv run ruff check --fix <file>
```

### TypeScript / frontend files

Run prettier after editing any file under `app/`:

```
cd app && npx prettier --write <file>
```
