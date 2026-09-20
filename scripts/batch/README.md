# Running the corpus fit on AWS Batch

The replacement for `scripts/loc_craft/{bootstrap,launch,supervise,shards}.sh`
and the SQS worker loop (#448): the image built from the repo's `Dockerfile`
runs one item per Batch job, an array job maps over an item list, and Batch
owns scheduling, retries, spot replacement, logs and completion.

```
scripts/batch/setup.sh        one-time: ECR repo, roles, compute environment, queue, job definition
scripts/batch/push-image.sh   push a locally built image to ECR (:<sha> and :latest)
scripts/batch/submit.sh       submit an array job over an items list, under a run tag
scripts/batch/status.sh       children by state, rate, ETA, failed children with exit codes
```

## One-time setup

`setup.sh` needs an identity that can create IAM roles, ECR repositories and
Batch resources. The `mapsnap` profile (`mapsnap-mirror`) is scoped to
S3/SQS/EC2 and cannot; run it as the account's admin identity, or grant
`mapsnap-mirror` `ecr:*`, `batch:*`, `iam:CreateServiceLinkedRole` and
`iam:PassRole`. `--dry-run` prints the resource JSON without creating anything.

## The pilot: the test-200 items, fit-only, on Batch

`pilot-200.txt` is the test-200b list: the 192 items that finished under the
EC2 fleet plus the two it never took and the six it dead-lettered. Rerunning
them under a new run tag exercises the whole Docker + Batch path against
inputs whose outcomes are known; CRAFT boxes already exist for every one, so
no job needs a GPU.

```
# once, as the account admin (the `aws login` session):
scripts/batch/setup.sh
aws iam create-policy --policy-name mapsnap-batch-operator \
  --policy-document file://scripts/batch/mapsnap-mirror-batch-policy.json
aws iam attach-user-policy --user-name mapsnap-mirror \
  --policy-arn arn:aws:iam::213478311378:policy/mapsnap-batch-operator

# everything after that under the stable profile:
export AWS_PROFILE=mapsnap
docker buildx build --platform linux/amd64 --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t mapsnap:$(git rev-parse --short HEAD) --load .
scripts/batch/push-image.sh                      # ~40 min the first time
scripts/batch/submit.sh batch-test-200 scripts/batch/pilot-200.txt
scripts/batch/status.sh <array-job-id> --watch
```

The operator policy lets `mapsnap-mirror` push images, register job
definitions, submit and inspect jobs and read their CloudWatch logs. It cannot
create or change the roles, the queue or the compute environment, which stay
with the admin identity and a full `setup.sh`.

That split matters in practice because the admin session expires every ten to
twenty minutes. Pointing the job definitions at a newly pushed image is the
one setup step that comes up on every code change, so it is the one the scoped
identity can do:

```
scripts/batch/setup.sh --job-definitions-only \
  IMAGE=...  # or leave IMAGE unset to keep :latest
```

A full `setup.sh` under the scoped profile now says so plainly instead of
failing on a `CreateRole` for a role that already exists.

It is attached as a *managed* policy rather than an inline one: all of a
user's inline policies together may not exceed 2,048 bytes, and
`mapsnap-mirror` already spends most of that on its S3, SQS and EC2 grants,
so `put-user-policy` fails with `LimitExceeded`. A managed policy has its own
6,144-byte budget and can be edited later with `create-policy-version`.

## A run

```
docker buildx build --platform linux/amd64 --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t mapsnap:$(git rev-parse --short HEAD) --load .
scripts/batch/push-image.sh
scripts/batch/submit.sh batch-test-200 items.txt      # one item id per line
scripts/batch/status.sh <array-job-id> --watch
```

Outputs land under `<item>/runs/<run-tag>/` exactly as a fleet run's do, and
`mapsnap loc-fit` skips an item already finished under that tag, so
resubmitting the same list after failures re-runs only what is left.

## How a job maps to `loc-fit`

Child N of the array runs `mapsnap loc-fit --items <s3 list> --run-tag <tag> …`
and takes line N from `AWS_BATCH_JOB_ARRAY_INDEX`. Its exit code is the
item's outcome, which the job definition's retry policy reads:

| exit | meaning | Batch |
|---|---|---|
| 0 | fitted, or already done under this tag | SUCCEEDED |
| 1 | the chain raised | retried once, then FAILED |
| 3 | unprocessable: no pages in the mirror, no county extract | FAILED, never retried |
| 4 | inputs missing (CRAFT boxes) | FAILED, never retried |
| 137 | killed for memory | retried once — then raise the job definition's memory |

The run's `artifacts/mapsnap/manifest.json` records the Batch job id, attempt
and array index beside the git sha, and `loc-fit`'s summary line prints the
peak resident set of any stage, which is what to size the job definition's
memory from after a pilot (2 vCPU / 7 GB to start).

## If `setup.sh` stops

**"Compute Environment ... is not valid. It must be valid before attaching it
to the job queue"** -- the environment was still `CREATING`. The script now
waits for `VALID`; re-run it and it will skip everything that exists and
create the queue. If it reports `INVALID` instead, that is a configuration
fault and re-running cannot repair it: the environment has to be disabled,
deleted and recreated, which the error message spells out. The usual causes
are a missing `AWSServiceRoleForEC2Spot` (the script now creates it), an
`ecsInstanceRole` whose instance profile has not propagated, or a default VPC
with no subnets in the region.

Check by hand with:

```
aws batch describe-compute-environments --region us-west-2 \
  --compute-environments mapsnap-cpu-spot \
  --query 'computeEnvironments[0].[status,statusReason]' --output text
```

## What the 200-item pilot cost and found (2026-09-19)

194 of 200 items, 5,459 pages, in 2.6 hours of wall clock over 34 spot
instances: **$5.88**, or $0.0011 per page. Two instances were reclaimed by
spot mid-run and all eight of their children retried and succeeded, which is
the behaviour the EC2 fleet never had.

Fitting a fixed-plus-marginal model to the per-item times -- 189 s of fixed
cost per item, 39 s per page -- and projecting over the manifest's 35,159
items and 441,179 sheets gives **about $550 and 13,000 vCPU-hours** for the
whole corpus, or 52 hours of wall clock at 128 concurrent jobs and 26 at 256.
Do not project per *item* from this pilot: its items average 29.7 sheets
against the corpus's 12.5, so a per-item extrapolation overstates the bill by
2.4x.

Four items failed for reasons worth knowing: three were SIGKILLed against the
7,000 MB ceiling (hence the `-large` definition), and one hit a crash in the
clip-mask pass that is now fixed. The two exit-3 items are genuinely absent
from the mirror.

### Why that $550 is roughly twice what it should be

The fleet supplied 308 vCPU-hours and the jobs consumed 143: **46% vCPU
utilisation**. The cause is packing, not Batch, which charges nothing of its
own. A job asks for 2 vCPU and 7.5 GB, so four fit on an 8 vCPU box -- but
only if the box carries 30 GB. `c5.2xlarge` and `c6i.2xlarge` carry 16 GB, so
memory capped them at two jobs each and half of their vCPUs sat idle while we
paid for them. 23 of the 38.5 instance-hours were on those shapes. They are
out of the compute environment now, which should roughly halve the corpus
bill on its own.

The second overhead is per-job startup: a one-page item takes 57 s at best and
113 s typically, nearly all of it container start plus parsing the 3.8 MB
county manifest. The EC2+SQS worker paid that once and then drained a queue;
one job per item pays it 35,159 times, about 18% of the corpus bill. The fix
is to give each Batch child a slice of the list rather than a single line,
which needs a `loc-fit` change and is worth roughly another $30-40.

Projected corpus cost, fit-only, at the pilot's spot prices:

| | job-hours | cost |
|---|---|---|
| as the pilot ran | 6,168 | $507 |
| 32 GB+ shapes only | 6,168 | ~$254 |
| plus 8 items per child | 5,202 | ~$214 |

### Mopping up after a run

Nothing retries into the larger definition on its own: Batch cannot change a
job definition on retry, so it takes a second submission. `retry-list.sh`
works out which failures are worth one. `loc-fit` exits 1 whether a stage was
killed for memory or raised, and only its log tells them apart, so this reads
the logs and splits them:

```
scripts/batch/retry-list.sh <array-job-id> retry
  sanborn06645_034  child 28   memory   mapsnap ocr failed (exit -9)
  sanborn01711_002  child 38   memory   mapsnap fit failed (exit 247)
6 out of memory, 0 other, 0 not worth retrying (exit 3 or 4), 0 unattributed

PER_JOB=1 JOBDEF=mapsnap-loc-fit-large scripts/batch/submit.sh <run-tag> retry-oom.txt
```

The failing items come out of the logs, not off the items list: in a chunked
run a child holds eight items and its index says nothing about which one died.
A memory kill wears three different exit codes depending on how deep it
happened -- `-9` for a stage killed outright, `247` for a stage whose own
child was killed, `137` for the container itself -- and all three mean the
same thing.

Reuse the same run tag. Items that already finished are marked done by
`plan_fit` and cost one listing each, so a resubmission only does what is
left. Exit 3 and 4 are left out of both lists, since neither is fixed by
running it again. An array needs at least two children, so a lone survivor
goes through `mapsnap loc-fit --item` instead.

## Running a sample first, then the rest

`sample-items.py` splits the mirror into a seeded random sample and everything
else, and the two together are every item exactly once:

```
scripts/batch/sample-items.py --size 1000 --out-prefix corpus
  corpus-sample.txt: 1,000 items, 13,132 sheets, mean 13.1, median 4, p90 32
  corpus-rest.txt:  34,159 items, 428,047 sheets, mean 12.5, median 4, p90 28
  the sample averages 1.05x the typical item
```

Run the sample, check the cost and the failure rate, then run the rest **under
the same run tag** and the halves are a complete corpus run. A run tag does not
care that it was filled in several passes: an item already published under it
is skipped for the price of one listing, so the passes can even overlap.
`submit.sh` keys each uploaded list by its own contents (`items-<cksum>.txt`),
so they sit beside each other rather than overwriting.

`--exclude` takes a list to leave out of both halves, which is how the second
pass avoids the first:

```
aws s3 cp s3://mapsnap-sanborn/_runs/corpus-v1/items-<cksum>.txt done-so-far.txt
scripts/batch/sample-items.py --size 1000 --seed 20260921 \
  --exclude done-so-far.txt --out-prefix corpus-2
```

That 1.05x is the line worth reading. A random draw is representative and
projects honestly; the 200-item pilot was 2.37x the typical item and its
per-item cost overstated the corpus by the same factor.

## Measuring what a run cost

Two of the three numbers expire: Batch drops a child's attempt history 24 hours
after it finishes, and a terminated instance leaves `describe-instances` within
the hour, taking its spot-reclamation reason with it. So the fleet is sampled
*while* the job runs:

```
scripts/batch/snapshot-instances.sh "$JOB" instances.tsv &   # stops with the job
scripts/batch/status.sh "$JOB" --watch

scripts/batch/collect-run.py "$JOB" planned.txt \
  --instances instances.tsv --items-per-job 8 --out run-report.json
```

`collect-run.py` prices the instance-seconds the fleet actually ran at each
type and zone's spot rate, counts the children whose host was reclaimed, and
reads pages and peak RSS out of `loc-fit`'s own summary lines.

**vCPU utilisation is the number to watch**: the share of what we rented that
the jobs occupied. 46% is what two-jobs-to-a-box packing looks like; anything
near 90% means the shapes and the memory request agree.

Two cautions. Its fixed-plus-per-page **model line is meaningless on a chunked
run** -- balancing makes pages-per-child nearly constant, so the regression has
no spread to fit a slope through and returns a negative per-page cost. Use cost
per sheet. And `mapsnap fit` records its own sub-stage seconds in every item's
`artifacts/mapsnap/manifest.json`, which is where `snap` versus `street-solve`
versus `georef` can be read for a finished run.

## Items per child

A Batch array caps at 10,000 children and the mirror holds 35,159 items, so
the corpus **cannot** run one item to a child. `PER_JOB` sets how many
consecutive lines of the list each child takes, and `submit.sh` sizes the
array from it:

```
PER_JOB=8 scripts/batch/submit.sh corpus-v1 corpus-items.txt   # 4,395 children
```

| items per child | children | fits the cap |
|---|---|---|
| 1 | 35,159 | no |
| 2 | 17,580 | no |
| 4 | 8,790 | yes |
| 8 | 4,395 | yes |

**Plan the list before submitting.** Item ids say nothing about volume size,
so chunking the list in its natural order hands one child several 150-sheet
volumes while another gets eight single sheets. `plan-items.py` reorders the
list so consecutive runs of `PER_JOB` are similar work -- `submit.sh` slices
by position, so reordering is all it takes:

```
scripts/batch/plan-items.py corpus-items.txt --per-job 8 --out planned.txt
PER_JOB=8 scripts/batch/submit.sh corpus-v1 planned.txt
```

Simulated over the whole mirror at 8 items a child and 128 slots:

| | list order | balanced |
|---|---|---|
| longest child | 10.0 h | 1.7 h |
| makespan | 54 h | 49 h |
| idle slot-hours | 770 | 133 |
| work redone at a 4% interrupt rate | 4.7% | 2.0% |

The rework column is the part that is easy to miss: a spot reclamation costs
whatever the child had done so far, so a 10-hour child is a far worse thing to
lose than a 1.7-hour one. Sheet count is the weight, which balances as well
here as the pilot's measured timings and needs no measurements to stay true.

It also puts `loc-fit`'s prefetch back to work. The driver runs `prepare_next`
on a worker thread, downloading the next item while the current one fits; a
one-item process has nothing to overlap, so that thread has been idle since
the queue went away.

Do not expect it to move the bill much. The per-item fixed cost is 57 s at
best and 113 s typically, and the parts a chunk actually amortises are small:
container start is ~4 s and parsing the county manifest is 0.1 s. Most of the
rest is the chain itself spawning a subprocess per stage, each importing
torch and friends -- 9.1 s of import floor per item, about 89 job-hours across
the corpus -- and that is paid per item however the children are grouped.
Chunking is for the array cap and the prefetch; packing was the money.

## Limits worth knowing

An array job holds at most 10,000 children, which is why the corpus runs
several items to a child (see above) rather than as four arrays. Spot
reclamation shows as a `Host EC2…` status reason and is always retried; the
pilot lost two instances that way and all eight of their children recovered.

## Running the image locally on an Apple Silicon Mac

The amd64 image cannot run torch inference under Docker's x86 emulation on
Apple Silicon: EasyOCR's first forward pass dies with
`qemu: uncaught target signal 4 (Illegal instruction)`, and neither
`ATEN_CPU_CAPABILITY` nor `DNNL_MAX_CPU_ISA` reaches whichever kernel QEMU
lacks. That is a property of the emulator, not the image -- the same wheel is
what `bootstrap.sh` installs on EC2, where it fits whole volumes. To run an
item locally, build the arm64 variant of the same Dockerfile (the AWS CLI
archive is picked by `TARGETARCH`) and it runs natively:

```
docker buildx build --platform linux/arm64 --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t mapsnap:dev-arm64 --load .                                  # 44 s, 2.07GB
docker run --rm --env-file <creds> mapsnap:dev-arm64 \
  loc-fit --item sanborn05939_001 --run-tag docker-smoke --counties s3://... s3://...
```

Gardiner (one page) fits end to end that way in 156 s. The arm64 image is
also what a Graviton compute environment would run.
