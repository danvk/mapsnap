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
and `_craft/*`), EC2's Spot service-linked role, and launch rights for the
`mapsnap-mirror` user. Quotas are counted in vCPUs and G-family spot and
on-demand are separate pools; check what you have with:

```sh
for code in L-3819A6DF L-DB2E81BA; do   # G+VT spot, G+VT on-demand
  aws service-quotas get-service-quota --region us-west-2 --service-code ec2 \
    --quota-code $code --query '[Quota.QuotaName,Quota.Value]' --output text
done
```

At 8 spot + 8 on-demand vCPUs, four `g6.xlarge` run at once (two per pool).

## Pilot first

```sh
git push origin HEAD                                  # instances clone the launched ref
scripts/loc_craft/launch.sh --shards 64 --only 0 --extra-args "--limit 50"
```

One instance, 50 items, about 20 minutes. Read its log (below) for the
items-per-hour line, then divide: 35,114 items at that rate is the whole corpus.
The pilot is also what says whether the per-page numbers from the sizing
benchmark hold on real volumes rather than Hudson.

## Full run

```sh
scripts/loc_craft/launch.sh --shards 4 --on-demand-from 2
```

Four shards, the first two on spot and the last two on demand, which is the
whole G-family quota. Each instance terminates itself when its shard is done.
With more quota, raise `--shards` to match.

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

`--workers N` runs N driver processes per instance, each on a sub-shard, so
CRAFT's CPU post-processing on one item overlaps another's GPU work. `ocr`
gained 2x that way on the same hardware, but it is unmeasured for this pass:
run the pilot once at `--workers 1` and once at `--workers 2` and compare the
items-per-hour line before committing the full fleet.
Transfer is free (same region) and small: ~412 GB down, ~83 GB of sidecars up.
