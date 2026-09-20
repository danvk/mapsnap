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

The operator policy lets `mapsnap-mirror` push images, submit and inspect
jobs and read their CloudWatch logs; it cannot create or change the
infrastructure, which stays with the admin identity and `setup.sh`.

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
old 7,000 MB ceiling (hence the 8 GB default and the `-large` definition),
and one hit a crash in the clip-mask pass that is now fixed. The two exit-3
items are genuinely absent from the mirror.

To rerun the heavy tail against the larger definition:

```
JOBDEF=mapsnap-loc-fit-large scripts/batch/submit.sh <run-tag> heavy-items.txt
```

## Limits worth knowing

An array job holds at most 10,000 children: the full corpus is four arrays.
A job has no cross-item prefetch (the fleet downloaded the next item while
fitting the current one); parallelism across jobs covers it. Spot
reclamation shows as a `Host EC2…` status reason and is always retried.

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
