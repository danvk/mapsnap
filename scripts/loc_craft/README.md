# Corpus CRAFT + P(road) pass

Caches the two image-only sidecars for all 432,293 pages and 4,080 raw key-map
sheets in the Sanborn mirror: `<stem>.boxes.json` from CRAFT and
`<stem>.roadprob.jpg` from the road UNet. Both depend on nothing but the image
and a model, so they are computed once and live beside the images for good;
everything downstream (split, ocr, keymap, fit) varies per run and stays out of
this pass.

`mapsnap loc-craft` does the work for one **shard**: the items whose id hashes
to its number. Shards are static, so there is no queue and no coordinator, and a
worker killed mid-shard is replaced by launching the same shard again — finished
items are skipped by their sidecars in S3.

## One-time setup (admin session)

```sh
aws login
scripts/loc_craft/iam-setup.sh
```

Creates the `mapsnap-craft` instance role (read the bucket, write `by-state/*`
and `_craft/*`), EC2's Spot service-linked role, and a managed policy granting
the `mapsnap-mirror` user launch rights, after which it can read _and raise_
quotas without an admin session. The grant is a managed policy rather than an
inline one because IAM caps a user's _aggregate_ inline policy size at 2048
bytes, which this plus the sizing benchmark's grant exceeds. Quotas are counted in vCPUs and G-family spot and on-demand are
separate pools:

```sh
for code in L-3819A6DF L-DB2E81BA; do   # G+VT spot, G+VT on-demand
  AWS_PROFILE=mapsnap aws service-quotas get-service-quota --region us-west-2 \
    --service-code ec2 --quota-code $code --query '[Quota.QuotaName,Quota.Value]' --output text
  AWS_PROFILE=mapsnap aws service-quotas list-requested-service-quota-change-history-by-quota \
    --region us-west-2 --service-code ec2 --quota-code $code \
    --query 'RequestedQuotas[].[Created,DesiredValue,Status,CaseId]' --output text
done
```

At 8 spot + 8 on-demand vCPUs, four `g6.xlarge` run at once (two per pool). A
request whose history still says `CASE_OPENED` is being worked by a human, and a
second request for the same quota is refused while it is open; a partial grant
(0 -> 8 when 32 was asked) does not close the case. Either reply on that case in
Support Center with the usage history, or ask for a smaller step, which is more
often granted automatically:

```sh
AWS_PROFILE=mapsnap aws service-quotas request-service-quota-increase --region us-west-2 \
  --service-code ec2 --quota-code L-3819A6DF --desired-value 16
```

## Pilot first

```sh
git push origin HEAD                                  # instances clone the launched ref
scripts/loc_craft/launch.sh --shards 64 --only 0 --extra-args "--limit 50"
```

One instance, 50 items, about 20 minutes. Read its log (below) for the
pages-per-hour line, then divide: 35,114 items at that rate is the whole corpus.
Each shard is walked in a fixed shuffled order (`--seed`), so a `--limit` sample
is a representative mix of eras and formats rather than the lowest item ids --
the corpus's first item is an 1867 Boston atlas of unsplit two-page spreads that
tiles into four and takes twelve minutes on its own.
The pilot is also what says whether the per-page numbers from the sizing
benchmark hold on real volumes rather than Hudson.

## Full run

```sh
scripts/loc_craft/launch.sh --shards 4 --on-demand-from 2 --workers 2
```

Four shards, the first two on spot and the last two on demand, which is the
whole G-family quota at 8 vCPUs each. Each instance terminates itself when its
shard is done. With more quota, raise `--shards` to match.

Spot launches order zones by current price but **rotate the starting zone per
shard**, so the fleet spreads across pools instead of piling into the cheapest
one. Ordering by price alone put all 30 key-map instances in us-west-2d on
2026-09-14, and one pool reclamation at 19:04:44 took 29 of them five minutes
after launch, before any had recorded a single item. The few dollars a cheap
zone saves are not worth a correlated total loss. Each shard still falls through
the other zones when its first has no capacity; the order is printed at launch.

Spot reclamation is the normal failure of this fleet, not an exception. Treat an
instance terminating as no evidence at all about whether its work finished:
check the outputs in the bucket. `StateTransitionReason` distinguishes them --
"Service initiated" is a reclamation, "User initiated" is a clean self-shutdown.

Re-partitioning later is safe, because an item is skipped on the strength of its
sidecars in S3 rather than on which shard claimed it. To grow the fleet mid-run,
terminate the running instances and relaunch with the larger `--shards`: only
the items actually in flight are repeated. Running two partitions at once is the
thing to avoid, since their shards overlap and both would compute the same
items.

Watch:

```sh
AWS_PROFILE=mapsnap aws ec2 describe-instances --region us-west-2 \
  --filters Name=tag:project,Values=mapsnap-craft Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[Tags[?Key==`shard`].Value|[0],InstanceId,LaunchTime]' --output table

AWS_PROFILE=mapsnap aws s3 ls s3://mapsnap-sanborn/_craft/logs/
AWS_PROFILE=mapsnap aws s3 cp s3://mapsnap-sanborn/_craft/logs/<name>.log - | tail -20
```

Each item logs a line with its counts, the running items-per-hour rate and an
ETA for the shard. Logs land in S3 every two minutes.

If a spot instance is reclaimed, launch its shard again:

```sh
scripts/loc_craft/launch.sh --shards 4 --only 1
```

## Identifying the key maps

`mapsnap loc-keymaps` is the same shape on the Standard (CPU) quota, and answers
the question the raw mirror cannot: which sheet of each volume is its key map.
The mirror kept full-resolution copies of the page-0 family and lettered sheets
only, so a volume whose key map is in the page-1 family has no raw sheet yet.

```sh
scripts/loc_craft/launch.sh --job loc-keymaps --shards 32
```

Most items never load a model: an item with fewer pages than the coverage floor
cannot have a detectable key map, and one with an unsplit page 0 is a key map by
convention, recorded from file names with nothing downloaded. Only the rest, about
11,185 of 35,158, download their one to four candidate pages and run the CNN
localizer and CRNN reader, at about 65 s each on a c6i.2xlarge.

It writes `keymaps.json` into each item's prefix, which is what `mapsnap keymap`
reads, so detection never repeats. Re-run with `--force` after retraining the
number models, or with a lower `--min-distinct` if the floor of
[#405](https://github.com/danvk/mapsnap/issues/405) changes.

The closing summary counts the key maps found that are **not** in the mirror's raw
set: those are the sheets a later pass has to fetch as JP2s and convert.

## Fetching the key-map sheets the mirror skipped

`mapsnap loc-keymaps` names each volume's key map; the ones outside the page-0
family and the lettered sheets have no full-resolution copy, because the mirror
kept raw sheets only for those. `mapsnap loc-raw` fetches them: JP2 from the
source mirror, decoded at full resolution, uploaded to the item's `raw/` prefix.

```sh
uv run mapsnap loc-raw --build-list keymaps.tsv          # sweep the records
scripts/loc_craft/launch.sh --job loc-raw --shards 4 \
  --extra-args "--list keymaps.tsv --mirror http://host:port"
```

Run it on an instance rather than a laptop: the original mirror took two days
because a home uplink caps near 2.5 MB/s, and in-region the upload is free.

**Use few shards.** The source mirror is somebody's machine, measured near
20 MB/s in total, which at about 7 MB a sheet is already close to 10,000 sheets
an hour. More shards crowd each other and the person hosting it. Tell them before
a run of this size.

## Fitting: the CPU chain

Once `loc-craft` has left a volume with its CRAFT boxes and P(road) maps, the
rest of the pipeline is CPU work: `split`, `adjacency`, `keymap`, `ocr` and
`fit`. All of it is scoped to the volume -- the key map is confirmed against the
volume's page set, adjacency needs every sheet, georef needs the volume's
reference scale, reconcile is one joint decision per volume -- and an LoC item
is a volume, so `mapsnap loc-fit` runs the whole chain per item.

```sh
# The item -> county mapping the workers read, once
aws s3 cp ~/Documents/mapsnap/loc-counties/items.tsv      s3://mapsnap-sanborn/_craft/
aws s3 cp ~/Documents/mapsnap/loc-counties/city-items.tsv s3://mapsnap-sanborn/_craft/

# Its own queue, with a lease long enough for the biggest volume: the corpus
# tops out at 167 sheets, about 28 minutes of chain on one worker, so 30 min
# would hand a still-running item to a second worker. Three hours is safe.
uv run mapsnap work-queue create --name mapsnap-fit --visibility 10800
FIT_QUEUE=<the url it prints>
uv run mapsnap work-queue fill --url "$FIT_QUEUE"

scripts/loc_craft/launch.sh --job loc-fit --shards 8 --workers 4 \
  --extra-args "--queue $FIT_QUEUE --counties s3://mapsnap-sanborn/_craft/items.tsv s3://mapsnap-sanborn/_craft/city-items.tsv"
```

`--job loc-fit` picks `c6i.2xlarge`; the county extracts are read straight
from `osm-by-county/` (`load_centerlines` takes the `.pbf` directly). Start
with `--workers 4` on an 8-vCPU box and watch memory before going higher: each
worker is a full ocr plus fit process.

Three things about what it uploads and what it skips:

- **The candidate files are kept.** `artifacts/*/candidates.jsonl` is
  nominally a cache, but it is the only record of what snap and street-solve
  considered and rejected, and regenerating it means re-running the search.
  Madison p20__3 made the case: its provenance said snap offered no hypothesis
  and the reason was in a file that had not been kept. Measured at 10.7 KB a
  page over the truth volumes, so about 4.4 GB across the corpus, roughly
  \$0.10 a month. `artifacts/reconcile/` is still dropped, since its verdicts
  restate what the per-page provenance records already carry.
- **Panel images stay on the worker.** `make_iiif_georef` builds every page's
  image URL from its parent and split pages share the parent's canvas, so
  nothing reads `p209__1.jpg`; it reads the parent plus `p209.panels.json`. The
  panel image, its boxes and its P(road) crop are deterministic in the parent
  and the rings and cost 0.35 vCPU-s a page to re-cut, so the chain re-runs
  `split` locally every time rather than storing ~50 GB every later pass would
  re-download.
- **Not ready is not failed.** An item whose boxes are not all present is
  waiting on the GPU pass; it is released back to the queue, not retired, so
  filling the fit queue before craft finishes is safe.
- **Canvases point at LoC.** A mirrored volume has no reference annotation
  page, but its `metadata.json` carries everything a canvas needs, so `fit`
  reads that and addresses LoC's own image servers -- nothing has to be hosted
  and viewers get full-resolution tiles. The service id is the item's
  `storage_dir` with `/` as `:` plus the sheet's `stem`; checked against
  Columbus 1951 vol 3's real LoC manifest, all 102 derived ids match exactly.
  Page keys come from the sheet's `key` rather than being parsed back out of
  the URL, which would lowercase the suffix of the 10,882 corpus sheets keyed
  `p5S` and drop them from the annotation. Scoring is unaffected: the
  LoC-pointing file and the manifest-based one both score Columbus at 88.8,
  matching the archived run.

## Keeping a fleet alive overnight

Launches use one-time spot requests, so a reclaimed instance stays dead until
something relaunches it. That is not rare: 29 of 30 key-map instances went in a
single second on 2026-09-14, and a `loc-raw` shard went at 20:44 the same day.

```sh
scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --workers 2 --on-demand-from 2
scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --watch          # loop every 15 min
```

The instance type follows the job unless `--instance-type` overrides it: only
`loc-craft` takes a GPU, and the rest go to `c6i.2xlarge`. Defaulting a CPU job
to a GPU type sends it at the 8-vCPU G-family quota, where it fails with
`MaxSpotInstanceCountExceeded` while 256 vCPUs of Standard spot sit idle.

It relaunches any shard that is neither finished nor running. A shard counts as
finished when it has written `_craft/done/<job>-of-<shards>-shard-<n>`, which
`bootstrap.sh` does only on a clean exit -- from EC2 alone, a finished shard and
a reclaimed one look the same, because the instance is simply gone.

One-shot by default so it can live in cron and survive a laptop restart, which a
`--watch` loop in a terminal does not:

```
*/15 * * * * cd ~/github/mapsnap && scripts/loc_craft/supervise.sh --job loc-craft --shards 4 --workers 2 >> /tmp/supervise.log 2>&1
```

A fleet launched before the marker existed writes none, so the supervisor will
relaunch each of its shards once more after they finish. That run re-lists the
shard, finds everything done, exits cleanly and writes the marker, which stops
the cycle -- half an hour and a dollar or two per shard, once.

`--dry-run` reports what a sweep would launch without launching it.

## Spreading a fleet across regions

GPU quota is granted per region, and 8 vCPUs is only two `g6.xlarge`. A second
region roughly doubles the fleet, and cross-region S3 is both cheap and
transparent: the corpus reads about 324 GB and writes about 100 GB of sidecars,
which at $0.02/GB is under $10 for a full pass, and the CLI follows the bucket's
region on its own, so nothing needs configuring on the instance.

The one thing that must line up is the partition. A worker owns shard
`SHARD * WORKERS + w` of `SHARDS * WORKERS`, so two fleets are disjoint only if
they launch with the same `--shards` and `--workers` and take different
`--only`. Running a second fleet on the _same_ partition does not merely
duplicate a little work: every worker walks its shard in the same seeded
shuffle, so the newcomer skips the finished prefix, catches up to the running
fleet and then starts each item at the same moment it does. Nothing claims an
item that is in flight; only finished ones are skipped.

Six shards split four/two, GPU quota being 8 spot vCPUs in each region:

```sh
# us-west-2: shards 0-3, the last two on demand (spot quota is 2 instances)
scripts/loc_craft/launch.sh --shards 6 --only 0-3 --workers 2 --on-demand-from 2

# us-east-2: shards 4-5, both spot
scripts/loc_craft/launch.sh --shards 6 --only 4-5 --workers 2 --region us-east-2
```

Each region then supervises its own shards. Without `--own`, every supervisor
would relaunch the other region's shards into its own region:

```
*/15 * * * * cd ~/github/mapsnap && scripts/loc_craft/supervise.sh --job loc-craft --shards 6 --own 0-3 --workers 2 --on-demand-from 2 >> /tmp/supervise-west.log 2>&1
*/17 * * * * cd ~/github/mapsnap && scripts/loc_craft/supervise.sh --job loc-craft --shards 6 --own 4-5 --workers 2 --region us-east-2 >> /tmp/supervise-east.log 2>&1
```

Re-partitioning a running job is safe: completion lives in the S3 sidecars, not
in shard bookkeeping, so a new partition skips what is already done and loses
only the items in flight when the old instances are terminated.

## Running from a queue instead of shards

Static sharding makes each instance's share a launch-time decision. Resizing the
fleet means tearing it down and re-partitioning, and a shard whose instance
never launched is a hole nobody fills -- on 2026-09-15 three of six shards could
not launch for lack of g6 capacity, leaving half the corpus unworked while four
instances ran.

With a queue the workers are identical: start one or twenty, in any region, at
any moment, and each takes the next item. Nothing has to agree on anything.

```sh
uv run mapsnap work-queue create --name mapsnap-craft     # queue + dead-letter
uv run mapsnap work-queue fill --url "$QUEUE" --limit 200 # a pilot
uv run mapsnap work-queue status --url "$QUEUE"
```

Then launch instances with the URL instead of a shard:

```sh
scripts/loc_craft/launch.sh --shards 1 --workers 2 \
  --extra-args "--queue $QUEUE"
```

`--shards 1` is vestigial here: the worker ignores its shard number once
`--queue` is given. Launch as many instances as capacity allows, whenever it
allows, and add more later without touching the ones already running.

### What the visibility timeout does

Receiving an item hides it from other consumers for `--visibility` seconds and
deleting it retires it. That is a **lease, not a delivery interval**: throughput
is unrelated to it, and a worker takes its next item the moment it asks. The
only thing the timeout governs is how long a dead worker's item waits before
another worker may take it, so it has to exceed the longest an item can take --
the corpus's worst are 90-page volumes carrying a key-map sheet tiled at native
resolution, about 57 s per page, hence the 30-minute default.

An item that keeps killing its worker is not retried forever: after
`--max-receives` attempts it moves to `mapsnap-craft-dead`, where it can be
looked at. One bad TIFF stalled four `loc-raw` shards before this existed.

### Rollout

The queue needs IAM permissions that predate it, so re-run the setup first:

```sh
aws login                                   # admin session, expires every 12 h
scripts/loc_craft/iam-setup.sh
```

Prove it on a small slice before cutting a fleet over: fill with `--limit 200`,
run a single instance, and check `status` reaches zero. Nothing is at risk if it
misbehaves -- the work is idempotent either way, and the shard path still works.

## Checking the result

`mapsnap loc-craft --dry-run` lists what each item still needs without computing
anything, so a full sweep says whether the corpus is complete:

```sh
uv run mapsnap loc-craft --shards 1 --shard 0 --dry-run | tail -20
```

That is one S3 listing per item (~35k requests, a couple of cents) and takes
about half an hour; it prints nothing for items that are done.

## Sizing

From the 2026-09-13 benchmark, per page on an L4 (`g6.xlarge`): CRAFT 1.54 s,
P(road) 0.24 s; a raw key-map sheet's tiled CRAFT is 56.8 s. That is roughly 278
instance-hours for the corpus with one worker process per instance:

| fleet                                      | wall time |
| ------------------------------------------ | --------- |
| 4 x g6.xlarge (8 spot + 8 on-demand vCPUs) | ~2.9 days |
| 8 x g6.xlarge (32 vCPUs)                   | ~1.5 days |

The driver downloads the next item while the current one computes, so the S3
round trip does not idle the GPU; the first pilot showed untiled items running
at 2.4-3.7 s/page against 1.54 s of compute, about a third of the time waiting.

`--workers N` runs N driver processes per instance, each on a sub-shard, so
CRAFT's CPU post-processing on one item overlaps another's GPU work. `ocr`
gained 2x that way on the same hardware, but it is unmeasured for this pass.
Measure it with two pilot instances on disjoint shards:

```sh
scripts/loc_craft/launch.sh --shards 4 --only 0 --workers 1 --extra-args "--limit 25"
scripts/loc_craft/launch.sh --shards 4 --only 1 --workers 2 --extra-args "--limit 25"
```

`--limit` is per driver process, so the second instance does about twice the
items; compare **pages per hour** from each run's closing summary, not items or
wall time, since items vary from 1 to 100+ pages. With `--workers 2` the log
holds two interleaved processes, so add their two figures together.
Transfer is free (same region) and small: ~412 GB down, ~83 GB of sidecars up.
