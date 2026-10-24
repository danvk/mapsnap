---
name: background-jobs
description: "Launch, watch, and wait on long-running local jobs on Dan's Mac (zsh, macOS, Apple Silicon): nohup/background launches, Monitor or until-loops, pgrep checks, memory checks. Use whenever starting a job that outlives one Bash call, or writing a loop that waits on one or polls its progress."
---

# Long-running local jobs on macOS

Every trap below has failed silently on this machine. A broken monitor looks exactly like a healthy job doing nothing, so each one has cost real time.

## Launching

- **Anything over 10 minutes leaves the Bash tool.** Either use `run_in_background: true`, or `nohup bash script.sh > log 2>&1 &` and watch the log.
- **Write the loop as a script file and run it with `bash`**, or wrap it in `bash -c '…'`. The default shell is zsh, and zsh behaves differently (see below).
- **Never edit a running bash script.** Bash reads scripts incrementally, so an in-place edit corrupts the interpreter mid-loop. Copy the script to a new name and launch that.
- **Never rewrite a list file a running `while read … < list` loop is reading.** Truncating it ends the loop after the current item; relaunch from a new file.
- **Make loops skip finished items** (a done-marker or output file per item), so a restart costs nothing.
- **One heavy job at a time.** Never run two OCR or fit jobs at once, and use the default `--num-workers` (serial); the laptop runs out of memory otherwise.

## Check that a worker really exists

Within a minute of launching, confirm the worker process exists and has memory. Don't trust the launch.

```bash
ps -axo pid,rss,command | grep "[.]venv/bin/python"   # the [.] keeps grep from matching itself
```

## zsh traps (use bash for loops)

- **zsh doesn't word-split a plain variable.** `logs=$(…); for log in $logs` runs once, with every line glued together. (`for x in $(cmd)` does split, which hides the bug.)
- **zsh reserves some names.** `path`, `status`, `keymaps`, `signals` and `options` are read-only or special. Assigning `path=…` clobbers `PATH` for the rest of the command, so every later command is "not found". Use other names.
- **zsh errors on unmatched globs** (`nomatch`): `ls p*.foo` with no matches is an error, not an empty list.

## pgrep and waiting

- **macOS `pgrep` has no `-c`.** Use `pgrep -f pattern | wc -l | tr -d ' '`.
- **`pgrep -f` matches the waiting command itself.** Any pattern that appears in your own `bash -c '…'` or `until` command matches its argv, so the wait never ends; it once "waited" 10 minutes after the job had finished. The same goes for `pkill -f`. Instead, wait on:
  - a done-marker file or a log line, e.g. `until grep -q "ALL DONE" log; do sleep 60; done`;
  - the worker's PID: `while kill -0 "$PID" 2>/dev/null; do sleep 30; done`;
  - something only the worker's command line contains, e.g. `ps -axo command | grep "[.]venv/bin/mapsnap georef"`.
- **`grep -c` exits 1 when it counts zero.** A trailing `grep -c FAILED log` makes a successful wait report failure. Append `|| true`, or test the count.
- **Give every wait loop an exit that can't hang**, e.g. on an empty variable or a marker that will never appear. Add a timeout, or a check that the worker is still alive.

## Memory

- **`vm_stat` undercounts free memory**: compressed and cached pages are reclaimable but not counted. It reported 2–3 GB free when 82% was free. Use `memory_pressure | grep "free percentage"`.
- **Measure a process's peak with `/usr/bin/time -l`** or `footprint`. MPS (the GPU) adds native memory that Batch's Linux workers never see; use `--no-gpu` when sizing for Batch.

## Polling remote state

- **Never put a recursive S3 listing (`aws s3 ls --recursive`) in a polling loop.** Listing responses count as egress; one 5-minute loop cost about $26 in September. Poll a marker prefix non-recursively, SQS attributes, or Batch job status.
- **AWS sessions expire every 10–20 minutes locally.** An empty `aws s3 ls` usually means re-authenticate (`aws login`), not "nothing there".

## Before calling a job stuck

Check the source directly: the log tail, `ps` for a live worker with memory, output files' timestamps. A monitor reporting zeros is more often a broken monitor than a stuck job.
