#!/bin/bash
# Launch one benchmark instance (#354 sizing experiment).
#
#   scripts/aws_bench/launch.sh g4dn.xlarge
#   scripts/aws_bench/launch.sh c6i.2xlarge --bench-args "--cpu-pages 16 --slow"
#   scripts/aws_bench/launch.sh g4dn.xlarge --on-demand        # if spot capacity is short
#
# Uses the mapsnap-mirror IAM user (AWS_PROFILE=mapsnap) once iam-setup.sh has granted
# it launch rights. Resolves the newest Deep Learning Base OSS NVIDIA driver AMI, fills
# in bootstrap.sh as user-data, and starts a one-time spot instance that terminates
# itself when the benchmark finishes. Prints the instance id; watch it with:
#
#   aws ec2 describe-instances --instance-ids <id> --query 'Reservations[].Instances[].State.Name'
set -euo pipefail

INSTANCE_TYPE=${1:?usage: launch.sh <instance-type> [--on-demand] [--bench-args "..."] [--git-ref REF] [--region R]}
shift
MARKET=spot
BENCH_ARGS=""
GIT_REF=$(git rev-parse HEAD)
REGION=${AWS_REGION:-us-west-2}
BUCKET=${BUCKET:-mapsnap-sanborn}
PREFIX=${PREFIX:-_bench}
ROLE=mapsnap-bench
while [ $# -gt 0 ]; do
  case "$1" in
    --on-demand) MARKET=on-demand; shift ;;
    --bench-args) BENCH_ARGS="$2"; shift 2 ;;
    --git-ref) GIT_REF="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
export AWS_REGION=$REGION
# `cd` echoes the directory when CDPATH is set, so silence it.
HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)

# The ref must be on GitHub for the instance to check it out.
if ! git branch -r --contains "$GIT_REF" 2>/dev/null | grep -q origin; then
  echo "git ref $GIT_REF is not on origin; push it first (the instance clones from GitHub)" >&2
  exit 1
fi

# Newest Deep Learning Base OSS NVIDIA driver AMI (Ubuntu). Try the SSM public
# parameters first, then fall back to searching Amazon's images by name.
resolve_ami() {
  for os in ubuntu-24.04 ubuntu-22.04; do
    ami=$(aws ssm get-parameter \
      --name "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-${os}/latest/ami-id" \
      --query Parameter.Value --output text 2>/dev/null || true)
    if [ -n "$ami" ] && [ "$ami" != "None" ]; then echo "$ami"; return; fi
  done
  aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu *" \
              "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text
}
AMI=$(resolve_ami)
if [ -z "$AMI" ] || [ "$AMI" = "None" ]; then
  echo "could not resolve a Deep Learning Base AMI in $REGION" >&2
  exit 1
fi
AMI_NAME=$(aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].Name' --output text)
echo "AMI $AMI ($AMI_NAME)"

USER_DATA=$(mktemp)
sed -e "s|__BUCKET__|$BUCKET|" -e "s|__PREFIX__|$PREFIX|" -e "s|__GIT_REF__|$GIT_REF|" \
    -e "s|__BENCH_ARGS__|$BENCH_ARGS|" "$HERE/bootstrap.sh" > "$USER_DATA"

MARKET_ARGS=()
if [ "$MARKET" = spot ]; then
  MARKET_ARGS=(--instance-market-options \
    "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
fi

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile "Name=$ROLE" \
  ${MARKET_ARGS[@]+"${MARKET_ARGS[@]}"} \
  --instance-initiated-shutdown-behavior terminate \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}" \
  --user-data "file://$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mapsnap-bench-$INSTANCE_TYPE},{Key=project,Value=mapsnap-bench}]" \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$USER_DATA"
echo "launched $INSTANCE_TYPE ($MARKET) as $INSTANCE_ID at ref ${GIT_REF:0:10}; bench args: '${BENCH_ARGS}'"
echo "result will land at s3://$BUCKET/$PREFIX/results/$INSTANCE_TYPE-$INSTANCE_ID.json"
