---
name: issue-images
description: "Put an image (figure, screenshot, plot, before/after) into a GitHub issue, PR description, or comment for danvk/mapsnap, by committing it to the repo's issue-images branch and embedding its raw URL. Use whenever an issue or PR you're filing or updating rests on a figure, or the user asks to add an image to one. `gh` cannot upload images, so never leave a figure as a local path instead."
---

# Images in mapsnap issues and PRs

`gh` can't attach images. Instead, the repo keeps an orphan branch, `issue-images`, holding nothing but figures. Issues and PRs embed those figures by their `raw.githubusercontent.com` URL.

## Steps

1. **Use a worktree**, so the main checkout and its branch are untouched:

   ```bash
   cd ~/github/mapsnap
   git fetch -q origin issue-images
   git worktree add ../mapsnap-issue-images origin/issue-images -B issue-images
   ```

   If `../mapsnap-issue-images` already exists, run `git -C ../mapsnap-issue-images pull --ff-only` there instead.

2. **Put the files in a topic directory**, named for the subject rather than the issue number. Look at what's there (`git ls-tree -d --name-only origin/issue-images`) and reuse a fitting directory. Examples: `content-regions/ny1923-p26-regions.png`, `cutline-model/5-regression.jpg`. Give the files descriptive names.

   Keep each file under a few MB. A JPEG at quality ~85 is fine for map imagery; keep PNG for line art and plots.

3. **Commit and push** with the branch's message style:

   ```bash
   cd ../mapsnap-issue-images
   git add <dir>
   git commit -m "Issue images: <topic> (#N)"
   git push origin issue-images
   ```

4. **Embed and verify.** The URL is

   ```
   https://raw.githubusercontent.com/danvk/mapsnap/issue-images/<dir>/<file>
   ```

   Use it as `![alt text](URL)` in the issue body or comment (`gh issue edit N --body-file …`, `gh issue comment N --body …`, or the PR equivalents). Check that it resolves before saying it's done:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" <URL>   # expect 200
   ```

   Put a one-line caption above each image saying what to look at.

## Notes

- Never push figures to `main` or a feature branch, and never rewrite `issue-images` history: existing issues link to its files.
- Keep the scripts that made a figure out of this branch. Mention where they live in the issue text.
- Leave the worktree in place for next time.
