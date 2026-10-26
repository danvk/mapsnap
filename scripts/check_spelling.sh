#!/bin/bash
# Fail when a branch adds British spellings; CLAUDE.md asks for American English.
#
#   scripts/check_spelling.sh [<base>]     # default: origin/main
#
# Checks only the lines the branch adds (against its merge base with <base>), with
# codespell's built-in British-to-American list, so the codebase's hundreds of
# older spellings don't fail every PR. A legacy spelling that an interface needs
# (a CLI flag, a JSON key) can be added to IGNORE. CLAUDE.md is skipped: it
# quotes the British forms on purpose.
set -euo pipefail
BASE=${1:-origin/main}
IGNORE=mitre
added=$(git diff "$(git merge-base "$BASE" HEAD)" HEAD -U0 -- \
  '*.py' '*.md' '*.sh' '*.ts' '*.tsx' '*.yml' '*.toml' ':!CLAUDE.md' \
  | grep '^+' | grep -v '^+++' || true)
[ -n "$added" ] || exit 0
if ! printf '%s\n' "$added" | uvx codespell --builtin en-GB_to_en-US --ignore-words-list "$IGNORE" -; then
  echo "British spellings in the lines above; CLAUDE.md asks for American English." >&2
  exit 1
fi
