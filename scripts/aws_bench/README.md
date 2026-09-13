# GPU-vs-CPU sizing benchmark on AWS

Concrete per-page timings for the pipeline's accelerator-eligible stages (CRAFT,
ocr recognition, the road and region UNets, the key-map sheet) on NVIDIA spot
instances and on a CPU-only control, to size the #354 corpus run. Each instance
clones the repo at a pinned ref, installs the locked environment with `uv` (the
Linux lockfile brings CUDA-enabled torch, so no Docker image is needed), runs
`mapsnap bench` on one Hudson volume, uploads a JSON result, and terminates itself.

Everything the instance needs comes from the mirror bucket under the `_bench/`
prefix; nothing else in the bucket is written.

## One-time setup (admin session)

```sh
aws login                                   # short-lived admin credentials
scripts/aws_bench/iam-setup.sh              # role mapsnap-bench + launch rights for mapsnap-mirror
```

`iam-setup.sh` creates the `mapsnap-bench` instance role (read the bucket, write
`_bench/*`, SSM Session Manager), EC2's service-linked role for Spot, and adds an
inline launch policy to the existing `mapsnap-mirror` IAM user, so every later step
runs with `AWS_PROFILE=mapsnap`.

A fresh account's EC2 vCPU quotas for GPU instances are zero, and the Standard
family's are a handful, so `run-instances` fails with `MaxSpotInstanceCountExceeded`
(or `VcpuLimitExceeded` on-demand) until they are raised. Still in the admin session:

```sh
REGION=us-west-2
for code in L-3819A6DF L-DB2E81BA L-34B43A08 L-1216C47A; do   # G+VT spot, G+VT on-demand, Standard spot, Standard on-demand
  aws service-quotas get-service-quota --region $REGION --service-code ec2 --quota-code $code \
    --query '[Quota.QuotaName,Quota.Value]' --output text
done
aws service-quotas request-service-quota-increase --region $REGION --service-code ec2 --quota-code L-3819A6DF --desired-value 32
aws service-quotas request-service-quota-increase --region $REGION --service-code ec2 --quota-code L-DB2E81BA --desired-value 16
aws service-quotas request-service-quota-increase --region $REGION --service-code ec2 --quota-code L-34B43A08 --desired-value 32
```

Quotas are vCPUs: the three GPU launches below need 16 Spot vCPUs at once, the
c6i.2xlarge control needs 8 Standard. Modest increases are usually granted
automatically within the hour; larger ones (the corpus run will want hundreds) open
a support case, so request those early.

## Run

```sh
git push origin HEAD                        # the instance clones the ref you launch from
scripts/aws_bench/pack.sh                   # Hudson pages + centerlines + raw p0 + EasyOCR weights

scripts/aws_bench/launch.sh g4dn.xlarge     # T4, 4 vCPU
scripts/aws_bench/launch.sh g4dn.2xlarge    # T4, 8 vCPU: does ocr scale with workers?
scripts/aws_bench/launch.sh g6.xlarge       # L4, if spot capacity allows
scripts/aws_bench/launch.sh c6i.2xlarge --bench-args "--cpu-pages 16 --slow"   # CPU control
```

Each launch prints the instance id and the S3 key its result will land at. GPU
instances finish in 25 to 35 minutes; the CPU control with `--slow` (CRAFT on the
raw sheet at CPU speed) takes about an hour. Total spend for the four is well under
two dollars.

Watch and collect:

```sh
AWS_PROFILE=mapsnap aws ec2 describe-instances --region us-west-2 \
  --filters Name=tag:project,Values=mapsnap-bench \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output table

AWS_PROFILE=mapsnap aws s3 sync s3://mapsnap-sanborn/_bench/results/ results/
uv run mapsnap bench --report results/*.json
```

The report prints seconds per page for every stage and device, one column per
machine, next to this laptop's numbers if you ran `mapsnap bench` locally too:

```sh
uv run mapsnap bench --volume data/hudson_co_nj_1950_vol_9 --out results/laptop.json --pages 12
```

(Locally the accelerator is MPS; the bench copies nothing and only writes sidecars
into a scratch copy of the volume it makes under the output directory.)

## If something goes wrong

The bootstrap log is `/var/log/mapsnap-bench.log` on the instance and is uploaded to
`_bench/logs/` at the end. An instance that never finishes stays running; read its
console and terminate it:

```sh
AWS_PROFILE=mapsnap aws ec2 get-console-output --region us-west-2 --instance-id <id> --output text | tail -50
AWS_PROFILE=mapsnap aws ec2 terminate-instances --region us-west-2 --instance-ids <id>
```

For a live shell (no SSH keys involved) install the Session Manager plugin once with
`brew install --cask session-manager-plugin`, then:

```sh
AWS_PROFILE=mapsnap aws ssm start-session --region us-west-2 --target <id>
```

`launch.sh --on-demand` sidesteps a spot capacity shortage at roughly twice the price.

## What the numbers decide

- **CRAFT and the road UNet** are the stages expected to gain 5 to 10× from the GPU.
  Their T4 seconds per page, against the c6i control, set the GPU-hour budget for the
  corpus run.
- **ocr** is expected to be CPU-bound by the Python trie decoder. The `workers=N`
  rows show how many recognizer processes one T4 can feed before it saturates,
  which picks the instance size (xlarge versus 2xlarge).
- **The raw key-map sheet** (tiled CRAFT at native resolution) is the per-volume
  fixed cost. There are about 4,080 of them in the mirror.
