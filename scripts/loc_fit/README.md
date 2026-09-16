# Running the CPU chain on EC2

`loc-fit` runs split → craft (derive-only) → adjacency → keymap → ocr → fit over
one LoC item at a time, pulling items from a queue. It needs `loc-craft` to have
been through the item first: an item whose CRAFT boxes are missing is *not
ready*, not failed, and goes back to the queue.

The EC2 machinery — AMI resolution, spot zone rotation, IAM, bootstrap,
self-termination — is shared with `loc-craft` and lives in `../loc_craft/`.
Only the job, the queue and the run tag differ, so `launch.sh` here delegates
there. (The shared directory is named for the job that needed it first; treat it
as "the fleet scripts".)

## A run is a queue, a tag, and a directory

One `--run-tag` names all three:

| | |
|---|---|
| S3 | `<item>/runs/<tag>/` — this run's outputs, separate from every other run's |
| SQS | `mapsnap-fit-<tag>` — one queue per run |
| provenance | the `run` line in each manifest and annotation page |

The tag is typed once, in `launch.sh`, and travels in the queue message. A
worker takes the run from the work rather than from its own flags, so the
directory the outputs land in and the run recorded inside them cannot disagree.
`--run-tag` on a worker is an assertion against the message, and a mismatch
stops it.

**One queue per run, always.** SQS has no selective receive: a worker cannot
decline a message meant for another run, and merely passing one over counts
against `maxReceiveCount`. Three of those dead-letter a message nothing ever
worked on. Filtering by tag in the consumer looks reasonable and quietly
destroys the other run's queue.

## The run book

Cut the release first, so the tag names real code:

```bash
git tag v1.3 && git push origin v1.3
```

A pilot — a few hundred items, its own directory, colliding with nothing:

```bash
scripts/loc_fit/launch.sh --run-tag v1.3-pilot --fill --limit 200 --shards 2
```

`--limit` takes the manifest's first N, which is one state's worth. To choose
the items instead, fill from a manifest of your own — any subset of the mirror
manifest's rows is a valid one, and the workers keep the full manifest, since
the queue names the items and the manifest only resolves them:

```bash
scripts/loc_fit/launch.sh --run-tag v1.3-pilot --fill \
  --manifest ~/Documents/mapsnap/loc-fit-sample-200.tsv \
  --max-receives 10 --shards 3
```

`--max-receives` is worth raising while `loc-craft` is still running. An item
whose boxes are not all there yet is *released*, not retired, and a release
counts as a receive: at the default 3, an item waiting on the GPU pass
dead-letters after three passes rather than waiting for it.

The full pass:

```bash
scripts/loc_fit/launch.sh --run-tag v1.3 --fill --shards 8 --workers 2
```

More workers on a pass already running — same command without `--fill`, since
the queue exists and is filled:

```bash
scripts/loc_fit/launch.sh --run-tag v1.3 --shards 4
```

Watch it drain:

```bash
uv run mapsnap work-queue status --url "$QUEUE_URL"
```

Instances terminate themselves when the queue is empty and upload their log
first. `--dry-run` prints every command without running any of them.

## Reusing OCR

OCR is about two thirds of both the output bytes and the CPU, and a re-run often
does not need it — a change to georef, snap or reconcile leaves the reads alone.
`--ocr-from` borrows them from an earlier run:

```bash
scripts/loc_fit/launch.sh --run-tag v1.4 --ocr-from v1.3 --fill --shards 8
```

It is refused per item when that run read against a different county extract
(the manifest records `centerlines_sha`), because a read is a match between a
page's text and an extract, and the 0-buffer re-cut changed every one of them.
`ocr --resume` then covers the model half: it re-reads any page whose recognizer
weights differ, and any page the earlier run did not have — a newly split panel,
say. So the borrow is safe by construction, and silently degrades to a full
re-read rather than producing stale matches.

## What it costs

Measured over 31 volumes: **133 KB a page** of run output, so a full corpus pass
is about **55 GB** — roughly a dollar a month in S3 Standard. Keeping runs apart
is therefore nearly free. The compute is not: at the pilot's ~15 s/page, a full
pass is on the order of 1,800 worker-hours, and OCR is the bulk of it. Size the
pilot deliberately and plan on one full run, not iterating one.

The stable half of each item — images, CRAFT boxes, P(road) maps — is written
once at the item root by `loc-craft` and shared by every run. Treat it as
append-only: if it is ever regenerated in place, earlier runs stop describing
reproducible inputs, because their recorded inputs changed underneath them.

## Comparing two runs

Two runs at the same commit are the determinism check. Three fields differ by
construction and have to be normalized out first:

- `created` / `modified` on every annotation (wall clock)
- `generated` in the report card (the date)
- `run` in the report card (the tag itself)

and one by accident: name-hit text in `candidates.jsonl` is hash-order random
between processes unless `PYTHONHASHSEED` is pinned.

```bash
aws s3 sync --dryrun \
  "s3://mapsnap-sanborn/by-state/indiana/1904/sanborn02404_004/runs/v1.3/" \
  "s3://mapsnap-sanborn/by-state/indiana/1904/sanborn02404_004/runs/v1.3-rerun/"
```

Deleting a run is one command, which is the other half of why runs are kept
apart:

```bash
aws s3 rm --recursive --exclude '*' --include '*/runs/v1.3-pilot/*' \
  s3://mapsnap-sanborn/by-state/
```
