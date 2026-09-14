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
the `mapsnap-mirror` user launch rights, after which it can read *and raise*
quotas without an admin session. The grant is a managed policy rather than an
inline one because IAM caps a user's *aggregate* inline policy size at 2048
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
scripts/loc_craft/launch.sh --job loc-keymaps --instance-type c6i.2xlarge --shards 32
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
scripts/loc_craft/launch.sh --job loc-raw --instance-type c6i.2xlarge --shards 4 \
  --extra-args "--list keymaps.tsv --mirror http://host:port"
```

Run it on an instance rather than a laptop: the original mirror took two days
because a home uplink caps near 2.5 MB/s, and in-region the upload is free.

**Use few shards.** The source mirror is somebody's machine, measured near
20 MB/s in total, which at about 7 MB a sheet is already close to 10,000 sheets
an hour. More shards crowd each other and the person hosting it. Tell them before
a run of this size.

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

| fleet | wall time |
| --- | --- |
| 4 x g6.xlarge (8 spot + 8 on-demand vCPUs) | ~2.9 days |
| 8 x g6.xlarge (32 vCPUs) | ~1.5 days |

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
