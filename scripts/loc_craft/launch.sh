#!/bin/bash
# Launch the corpus CRAFT + P(road) fleet: one instance per shard (#354).
#
#   scripts/loc_craft/launch.sh --shards 4                      # 4 g6.xlarge spot
#   scripts/loc_craft/launch.sh --shards 4 --only 2             # just shard 2 (a replacement)
#   scripts/loc_craft/launch.sh --shards 6 --only 4-5 --region us-east-2  # this region's slice
#   scripts/loc_craft/launch.sh --shards 4 --on-demand-from 2   # shards 2,3 on demand
#   scripts/loc_craft/launch.sh --shards 1 --extra-args "--limit 50"   # a pilot
#   scripts/loc_craft/launch.sh --shards 4 --workers 2                # 2 processes per instance
#   scripts/loc_craft/launch.sh --job loc-keymaps --shards 32       # CPU type chosen from the job
#
# Each instance runs one shard and terminates itself when the shard is done.
# Shards are static, so re-launching a shard after a spot interruption resumes
# it: finished items are skipped by their S3 sidecars.
#
# Quotas are counted in vCPUs, and G-family spot and on-demand have separate
# pools, so --on-demand-from puts the later shards in the other pool. Quotas are
# also per region, so a fleet can span regions: launch every region with the
# same --shards and give each a disjoint --only, since the partition is
# (shard, worker) over --shards * --workers and only lines up if both agree.
set -euo pipefail

SHARDS=4
# Chosen from the job below unless --instance-type says otherwise: only loc-craft
# uses a GPU, and defaulting the others to one sends them at the 8-vCPU G-family
# quota, where they fail with MaxSpotInstanceCountExceeded while 256 vCPUs of
# Standard spot sit idle.
INSTANCE_TYPE=""
DRY_RUN=0
ONLY=""
ON_DEMAND_FROM=""
EXTRA_ARGS=""
WORKERS=1
JOB=loc-craft
GIT_REF=$(git rev-parse HEAD)
REGION=${AWS_REGION:-us-west-2}
BUCKET=${BUCKET:-s3://mapsnap-sanborn}
ROLE=mapsnap-craft
while [ $# -gt 0 ]; do
  case "$1" in
    --shards) SHARDS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;   # one shard, a list, or a range: 2 / 0,3 / 4-5
    --on-demand-from) ON_DEMAND_FROM="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;   # one type, or a comma-separated list
    --dry-run) DRY_RUN=1; shift ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;   # a count, or "auto" for one per vCPU
    --job) JOB="$2"; shift 2 ;;
    --git-ref) GIT_REF="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
# A flag with no value costs a whole fleet: on 2026-09-15 an unset $QUEUE made
# --extra-args "--queue $QUEUE" arrive as a bare --queue, and both instances
# booted, installed, then died on argparse and powered off. The shell will not
# catch it, so catch it here, before anything is launched.
case "$EXTRA_ARGS" in
  *--*)
    for word in $EXTRA_ARGS; do
      case "$word" in
        # Flags that take no value are legitimately last. Without this, a
        # perfectly good --extra-args "--manifest s3://... --gpu" is rejected
        # as a dangling flag, which is the opposite of the mistake the guard
        # was written to catch (an unset "$QUEUE" leaving --queue bare).
        --gpu|--dry-run|--force|--check-args|--resume|--no-snap|--omit-missing)
          dangling="" ;;
        --*) dangling=$word ;;
        *) dangling="" ;;
      esac
    done
    if [ -n "${dangling:-}" ]; then
      echo "--extra-args ends with $dangling and no value for it." >&2
      echo "  Usually an unset shell variable: --extra-args \"--queue \$QUEUE\"" >&2
      exit 2
    fi
    ;;
esac

if [ -z "$INSTANCE_TYPE" ]; then
  case "$JOB" in
    loc-craft) INSTANCE_TYPE=g6.xlarge ;;   # CRAFT and the road UNet want the GPU
    *) INSTANCE_TYPE=c6i.2xlarge ;;         # everything else is CPU work
  esac
fi
export AWS_PROFILE=${AWS_PROFILE:-mapsnap}
export AWS_REGION=$REGION
# `cd` echoes the directory when CDPATH is set, so silence it.
HERE=$(cd -- "$(dirname -- "$0")" > /dev/null && pwd -P)
source "$HERE/shards.sh"

if ! git branch -r --contains "$GIT_REF" 2>/dev/null | grep -q origin; then
  echo "git ref $GIT_REF is not on origin; push it first (instances clone from GitHub)" >&2
  exit 1
fi

# Newest Deep Learning Base OSS NVIDIA driver AMI (Ubuntu).
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
echo "AMI $AMI, $JOB on $INSTANCE_TYPE x${WORKERS} worker(s), ref ${GIT_REF:0:10}, bucket $BUCKET"

# Capacity is per availability zone, so try each zone's default subnet.
SUBNETS=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
  --query 'sort_by(Subnets,&AvailabilityZone)[].[AvailabilityZone,SubnetId]' --output text)

# Zones are ordered by current spot price, but each shard STARTS at a different
# one, rotating through the list. Ordering by price alone put all 30 key-map
# instances in us-west-2d on 2026-09-14, and a single pool reclamation at
# 19:04:44 took 29 of them five minutes after launch, before any had done work.
# Concentrating a fleet in the cheapest pool trades a few dollars for a
# correlated total loss; rotating keeps the price preference per shard while
# spreading the fleet across pools, and every shard still falls through the
# other zones when one has no capacity.
spot_prices() {
  aws ec2 describe-spot-price-history \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text 2>/dev/null | sort -u
}

zones_by_price() {
  local prices
  prices=$(spot_prices)
  if [ -z "$prices" ]; then
    echo "$SUBNETS"   # no price data: keep the alphabetical order
    return
  fi
  while read -r zone subnet; do
    [ -z "$zone" ] && continue
    # A zone the history does not mention sorts last rather than being dropped.
    price=$(echo "$prices" | awk -v z="$zone" '$1 == z {print $2; exit}')
    echo "${price:-9.999999} $zone $subnet"
  done <<< "$SUBNETS" | sort -n | awk '{print $2, $3}'
}

SPOT_SUBNETS=$(zones_by_price)
echo "spot zones, cheapest first (each shard starts at a different one): $(echo "$SPOT_SUBNETS" | awk '{printf "%s ", $1}')"

launch_shard() {
  local shard=$1 market=$2 user_data=$3 instance_id="" output="" last_error=""
  local market_args=()
  # Spot tries the cheapest zone with capacity; on-demand costs the same
  # everywhere, so it asks EC2 to place the instance before naming any zone.
  # Pinning a subnet is what made 3 of 6 shards fail on 2026-09-15: EC2 said so
  # itself -- "You can currently get g6.xlarge capacity by not specifying an
  # Availability Zone" -- while we were asking zone by zone and being refused.
  # Spot still pins, because pool diversity is what keeps one reclamation from
  # taking the whole fleet.
  local subnets=$SUBNETS
  if [ "$market" != spot ]; then
    subnets=$(printf 'any -\n'; echo "$SUBNETS")
  fi
  # Asking for ONE instance type is what made 6 of 11 spot instances fail or be
  # reclaimed on 2026-09-16. AWS scores a single-type request 1-3 out of 10 for
  # capacity and a diversified list 9, so every type is tried in every zone
  # before a shard is called a failure (#448).
  local types=${INSTANCE_TYPE//,/ }
  if [ "$market" = spot ]; then
    # Rotate the price-ordered list by the shard number so consecutive shards
    # start in different pools.
    local count offset
    count=$(echo "$SPOT_SUBNETS" | grep -c .)
    offset=$((shard % count))
    if [ "$offset" -eq 0 ]; then
      subnets=$SPOT_SUBNETS   # BSD head rejects -n 0
    else
      subnets=$(echo "$SPOT_SUBNETS" | tail -n +$((offset + 1)); echo "$SPOT_SUBNETS" | head -n "$offset")
    fi
  fi
  if [ "$market" = spot ]; then
    market_args=(--instance-market-options \
      "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
  fi
  local itype
  for itype in $types; do
  while read -r zone subnet; do
    # "-" is the unpinned attempt: no subnet, so EC2 picks a zone with capacity.
    local placement=(--subnet-id "$subnet")
    if [ "$subnet" = "-" ]; then placement=(); fi
    if [ "$DRY_RUN" = 1 ]; then
      echo "shard $shard/$SHARDS would try $itype ($market, $zone)"
      return 0
    fi
    if output=$(aws ec2 run-instances \
        --image-id "$AMI" \
        --instance-type "$itype" \
        ${placement[@]+"${placement[@]}"} \
        --iam-instance-profile "Name=$ROLE" \
        ${market_args[@]+"${market_args[@]}"} \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data "file://$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$JOB-$shard},{Key=project,Value=mapsnap-craft},{Key=job,Value=$JOB},{Key=shard,Value=$shard}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1); then
      instance_id=$output
      echo "shard $shard/$SHARDS ($itype, $market, $zone): $instance_id"
      return 0
    fi
    case "$output" in
      *InsufficientInstanceCapacity*|*Unsupported*) last_error=$output ;;
      *) echo "shard $shard: $output" >&2; return 1 ;;
    esac
  done <<< "$subnets"
  done
  # Report why the last zone refused. Swallowing it made two separate failures
  # on 2026-09-15 -- g6.xlarge spot in both regions, and three of four zones for
  # c6i.2xlarge -- indistinguishable from a quota problem, which is a different
  # fix entirely: capacity means wait or change zone, quota means ask AWS.
  echo "shard $shard: no $market capacity for any of [$types] in any zone" >&2
  echo "shard $shard: last zone said: ${last_error:-(no error recorded)}" >&2
  return 1
}

WANTED=$(expand_shards "${ONLY:-0-$((SHARDS - 1))}" "$SHARDS")

failed=0
for shard in $WANTED; do
  market=spot
  if [ -n "$ON_DEMAND_FROM" ] && [ "$shard" -ge "$ON_DEMAND_FROM" ]; then market=on-demand; fi
  user_data=$(mktemp)
  sed -e "s|__BUCKET__|$BUCKET|" -e "s|__GIT_REF__|$GIT_REF|" \
      -e "s|__SHARD__|$shard|" -e "s|__SHARDS__|$SHARDS|" \
      -e "s|__EXTRA_ARGS__|$EXTRA_ARGS|" -e "s|__WORKERS__|$WORKERS|" \
      -e "s|__JOB__|$JOB|" \
      "$HERE/bootstrap.sh" > "$user_data"
  launch_shard "$shard" "$market" "$user_data" || failed=$((failed + 1))
  rm -f "$user_data"
done
if [ "$failed" -gt 0 ]; then
  echo "$failed shard(s) did not launch; re-run with --only <shards> once capacity or quota allows" >&2
  exit 1
fi
