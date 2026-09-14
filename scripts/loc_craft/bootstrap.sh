#!/bin/bash
# EC2 user-data for one shard of the corpus CRAFT + P(road) pass (#354).
#
# launch.sh substitutes the __PLACEHOLDERS__ below and hands this file to
# `aws ec2 run-instances --user-data`. cloud-init runs it once, as root, on first
# boot: clone the repo at a pinned ref, install the locked environment with uv,
# run `mapsnap loc-craft` for this shard, then power off (the launch sets
# InstanceInitiatedShutdownBehavior=terminate, so that terminates the instance).
#
# The log is uploaded to <bucket>/_craft/logs/ every two minutes and again on
# exit, success or failure, and the instance powers off either way so a failed
# boot never sits idle. Work is idempotent: an interrupted shard is finished by
# launching the same shard number again, which skips every item already done.
set -euxo pipefail
exec > >(tee -a /var/log/mapsnap-craft.log) 2>&1

BUCKET="__BUCKET__"
GIT_REF="__GIT_REF__"
SHARD="__SHARD__"
SHARDS="__SHARDS__"
WORKERS="__WORKERS__"
JOB="__JOB__"
EXTRA_ARGS="__EXTRA_ARGS__"
TORCH_CUDA_FALLBACK="cu126"  # wheel variant for drivers too old for the locked CUDA 13 build
LOG=/var/log/mapsnap-craft.log

export DEBIAN_FRONTEND=noninteractive
export HOME=/root
export PATH="$HOME/.local/bin:$PATH"
WORK=/opt/craft
mkdir -p "$WORK"

TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
meta() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
INSTANCE_TYPE=$(meta instance-type)
INSTANCE_ID=$(meta instance-id)
LOG_KEY="_craft/logs/${JOB}-shard-${SHARD}-of-${SHARDS}-${INSTANCE_TYPE}-${INSTANCE_ID}.log"

# opencv-python (not the headless build) links libGL; git/curl/unzip for the rest;
# libopenjp2-tools is opj_decompress, which loc-raw decodes JP2s with.
apt-get update -q
apt-get install -y -q libgl1 libglib2.0-0 git curl unzip libopenjp2-tools
if ! command -v aws > /dev/null; then
  curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp
  /tmp/aws/install
fi

# From here on the log reaches S3 whatever happens.
upload_log() { aws s3 cp "$LOG" "$BUCKET/$LOG_KEY" > /dev/null 2>&1 || true; }
( while sleep 120; do upload_log; done ) &
UPLOADER=$!
finish() {
  local status=$?
  echo "bootstrap exit status $status"
  # Stop the periodic uploader FIRST. It reads the log when it wakes, so one
  # that woke before the job's closing summary was written can land its PUT
  # after this one and overwrite the finished log with a stale copy -- which is
  # how 41 of 96 shard summaries went missing from the key-map run.
  kill "$UPLOADER" 2> /dev/null || true
  wait "$UPLOADER" 2> /dev/null || true
  upload_log
  shutdown -h now
}
trap finish EXIT

curl -LsSf https://astral.sh/uv/install.sh | sh

cd "$WORK"
git clone --quiet https://github.com/danvk/mapsnap.git
cd mapsnap
git checkout --quiet "$GIT_REF"
git log -1 --oneline
uv sync --frozen --no-dev

# The lockfile's Linux torch is the CUDA 13 build, which needs driver >= 580.
# Older AMI drivers get the cu126 wheel so torch.cuda.is_available() stays true.
# The Deep Learning AMI ships nvidia-smi even on GPU-less instances, where it
# exits non-zero, so test that it works rather than that it exists.
GPU_FLAG=""
if nvidia-smi > /dev/null 2>&1; then
  nvidia-smi
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if [ "${DRIVER%%.*}" -lt 580 ]; then
    echo "driver $DRIVER < 580: swapping torch to the $TORCH_CUDA_FALLBACK wheel"
    uv pip install --reinstall --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_FALLBACK" \
      torch torchvision
  fi
  uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
  GPU_FLAG="--gpu"
fi

# EasyOCR's detector weights, so 35k items do not each race to jaided.ai.
mkdir -p "$HOME/.EasyOCR/model"
uv run python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" > /dev/null

# One process per worker, each taking a sub-shard of this instance's shard, so
# CRAFT's CPU post-processing on one item overlaps the GPU work of another. The
# nesting keeps every sub-shard static and resumable: worker w of this instance
# always owns shard (SHARD * WORKERS + w) of (SHARDS * WORKERS).
pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  # shellcheck disable=SC2086  # GPU_FLAG and EXTRA_ARGS are flag strings by design
  uv run mapsnap "$JOB" \
    --bucket "$BUCKET" \
    --shard "$((SHARD * WORKERS + worker))" \
    --shards "$((SHARDS * WORKERS))" \
    --work-dir "$WORK/scratch-$worker" \
    $GPU_FLAG $EXTRA_ARGS &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

echo "shard $SHARD/$SHARDS finished ($WORKERS worker(s), status $status)"
# The EXIT trap uploads the log and powers off.
