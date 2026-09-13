#!/bin/bash
# EC2 user-data for the GPU-vs-CPU sizing benchmark (#354).
#
# launch.sh substitutes the __PLACEHOLDERS__ below and hands this file to
# `aws ec2 run-instances --user-data`. cloud-init runs it once, as root, on first
# boot. It clones the repo at a pinned ref, installs the locked environment with uv,
# fetches the Hudson benchmark payload from S3, runs `mapsnap bench`, uploads the
# JSON result and the full log, and powers the instance off (the launch sets
# InstanceInitiatedShutdownBehavior=terminate, so that also terminates it).
#
# Anything that fails leaves the instance running with the log at
# /var/log/mapsnap-bench.log; read it with `aws ec2 get-console-output` or an SSM
# session (see README.md), then terminate the instance by hand.
set -euxo pipefail
exec > >(tee -a /var/log/mapsnap-bench.log) 2>&1

BUCKET="__BUCKET__"
PREFIX="__PREFIX__"
GIT_REF="__GIT_REF__"
BENCH_ARGS="__BENCH_ARGS__"
TORCH_CUDA_FALLBACK="cu126"  # wheel variant for drivers too old for the locked CUDA 13 build

export DEBIAN_FRONTEND=noninteractive
export HOME=/root
export PATH="$HOME/.local/bin:$PATH"
WORK=/opt/bench
mkdir -p "$WORK"

# Instance identity, for naming the result (IMDSv2 only; the launch requires tokens).
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 3600")
meta() { curl -s -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
INSTANCE_TYPE=$(meta instance-type)
INSTANCE_ID=$(meta instance-id)
RESULT_KEY="$PREFIX/results/${INSTANCE_TYPE}-${INSTANCE_ID}.json"
LOG_KEY="$PREFIX/logs/${INSTANCE_TYPE}-${INSTANCE_ID}.log"

# opencv-python (not the headless build) links libGL; git/curl/unzip for the rest.
apt-get update -q
apt-get install -y -q libgl1 libglib2.0-0 git curl unzip
if ! command -v aws > /dev/null; then
  curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp
  /tmp/aws/install
fi
curl -LsSf https://astral.sh/uv/install.sh | sh

cd "$WORK"
git clone --quiet https://github.com/danvk/mapsnap.git
cd mapsnap
git checkout --quiet "$GIT_REF"
git log -1 --oneline
uv sync --frozen --no-dev

# The lockfile's Linux torch is the CUDA 13 build, which needs driver >= 580. Older
# DLAMI drivers get the cu126 wheel instead so torch.cuda.is_available() stays true.
if command -v nvidia-smi > /dev/null; then
  nvidia-smi
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if [ "${DRIVER%%.*}" -lt 580 ]; then
    echo "driver $DRIVER < 580: swapping torch to the $TORCH_CUDA_FALLBACK wheel"
    uv pip install --reinstall --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_FALLBACK" \
      torch torchvision
  fi
  uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
fi

# Benchmark payload: one volume's 25% pages + centerlines + raw key-map sheet, and the
# EasyOCR detector/recognizer weights so nothing is downloaded from jaided.ai mid-run.
aws s3 cp "s3://$BUCKET/$PREFIX/hudson-bench.tar.gz" /tmp/hudson-bench.tar.gz
mkdir -p "$WORK/data" "$HOME/.EasyOCR"
tar xzf /tmp/hudson-bench.tar.gz -C "$WORK/data"
mv "$WORK/data/model" "$HOME/.EasyOCR/model"
VOLUME=$(find "$WORK/data" -mindepth 1 -maxdepth 1 -type d | head -1)

# shellcheck disable=SC2086  # BENCH_ARGS is a flag string by design
uv run mapsnap bench --volume "$VOLUME" --out "$WORK/results.json" --workers "$(nproc)" $BENCH_ARGS

aws s3 cp "$WORK/results.json" "s3://$BUCKET/$RESULT_KEY"
aws s3 cp /var/log/mapsnap-bench.log "s3://$BUCKET/$LOG_KEY"
echo "benchmark done: s3://$BUCKET/$RESULT_KEY"
shutdown -h now
