#!/bin/bash
# One-time AWS Batch setup for the corpus fit (#448): an ECR repository for
# the image, the two roles Batch needs, a spot compute environment, a job
# queue, and the loc-fit job definition. Idempotent where AWS allows it:
# re-running updates the job definition (a new revision) and leaves the rest.
#
#   scripts/batch/setup.sh                 # everything, us-west-2
#   scripts/batch/setup.sh --dry-run       # print the resource JSON, create nothing
#
# Needs an identity that can create IAM roles, ECR repositories and Batch
# resources: the `mapsnap` profile (mapsnap-mirror) is scoped to S3/SQS/EC2
# and cannot. Run it as the account's admin identity, or grant mapsnap-mirror
# ecr:*, batch:*, iam:CreateServiceLinkedRole and iam:PassRole on the roles.
#
# What it creates:
#   ECR    mapsnap                              (lifecycle: keep the last 10 tags)
#   IAM    mapsnap-batch-job (trusted by ecs-tasks; the S3/SQS policies of
#          mapsnap-craft copied over) and ecsInstanceRole + instance profile
#   Batch  compute environment mapsnap-cpu-spot  (SPOT_CAPACITY_OPTIMIZED, 0-256 vCPU)
#          job queue           mapsnap-fit
#          job definition      mapsnap-loc-fit  (2 vCPU / 7 GB, retry policy below)
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
DRY_RUN=0; JOBDEFS_ONLY=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    # A new image, or a changed command line, needs only the job definitions
    # re-registered. The roles, the queue and the compute environment are
    # already there, and it is reading those that a scoped identity trips on.
    --job-definitions-only) JOBDEFS_ONLY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
IMAGE=${IMAGE:-$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/mapsnap:latest}
# Dry-run echoes go to stderr so the callers' "> /dev/null" redirects cannot swallow them.
run() { if [ "$DRY_RUN" = 1 ]; then echo "+ $*" >&2; else "$@"; fi; }

# Does this thing exist? "I am not allowed to look" is not the same answer as
# "it is not there", and treating them alike is how this script came to report
# an AccessDenied on CreateRole for a role that was already sitting there.
#   0 it exists   1 it is genuinely absent   2 we cannot see it
exists() {
  local err
  if err=$("$@" 2>&1 > /dev/null); then return 0; fi
  case "$err" in
    *AccessDenied*|*"not authorized"*|*UnauthorizedOperation*) return 2 ;;
    *) return 1 ;;
  esac
}

needs_admin() {
  cat >&2 <<MESSAGE

$1 cannot be read or created by $(aws sts get-caller-identity --query Arn --output text 2>/dev/null || echo "this identity").

Creating and updating the IAM, ECR and Batch resources wants the account's
admin identity, not the scoped one. Re-run without the profile:

  env -u AWS_PROFILE ${IMAGE:+IMAGE=$IMAGE }$0

Everything after setup -- pushing an image, submitting, watching, collecting --
works as mapsnap-mirror. If all you are doing is pointing the job definitions
at a new image, --job-definitions-only skips every step that needs more.
MESSAGE
  exit 3
}
did() { if [ "$DRY_RUN" = 1 ]; then echo "would create $*"; else echo "created $*"; fi; }

echo "account $ACCOUNT, region $REGION, image $IMAGE"

if [ -n "$JOBDEFS_ONLY" ]; then
  echo "re-registering the job definitions only; leaving the roles, queue and compute environment alone"
else
# --- ECR ----------------------------------------------------------------------
exists aws ecr describe-repositories --repository-names mapsnap --region "$REGION"; state=$?
[ "$state" = 2 ] && needs_admin "The ECR repository mapsnap"
if [ "$state" = 1 ]; then
  run aws ecr create-repository --repository-name mapsnap --region "$REGION" \
    --image-scanning-configuration scanOnPush=false > /dev/null
  run aws ecr put-lifecycle-policy --repository-name mapsnap --region "$REGION" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep the last 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' > /dev/null
  echo "ECR: $(did mapsnap)"
else echo "ECR: mapsnap exists"; fi

# --- IAM: the job role (what the container is) --------------------------------
# Trusted by ECS tasks, carrying whatever mapsnap-craft carries today, so a job
# can read the mirror and write a run's outputs exactly as an instance did.
JOB_ROLE=mapsnap-batch-job
exists aws iam get-role --role-name $JOB_ROLE; state=$?
[ "$state" = 2 ] && needs_admin "The job role $JOB_ROLE"
if [ "$state" = 1 ]; then
  run aws iam create-role --role-name $JOB_ROLE --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > /dev/null
  for arn in $(aws iam list-attached-role-policies --role-name mapsnap-craft --query 'AttachedPolicies[].PolicyArn' --output text); do
    run aws iam attach-role-policy --role-name $JOB_ROLE --policy-arn "$arn"
  done
  for name in $(aws iam list-role-policies --role-name mapsnap-craft --query 'PolicyNames[]' --output text); do
    doc=$(aws iam get-role-policy --role-name mapsnap-craft --policy-name "$name" --query PolicyDocument --output json)
    run aws iam put-role-policy --role-name $JOB_ROLE --policy-name "$name" --policy-document "$doc"
  done
  echo "IAM: $(did "$JOB_ROLE with mapsnap-craft's policies")"
else echo "IAM: $JOB_ROLE exists"; fi

# --- IAM: the instance role (what the EC2 host under ECS is) ------------------
exists aws iam get-instance-profile --instance-profile-name ecsInstanceRole; state=$?
[ "$state" = 2 ] && needs_admin "The instance profile ecsInstanceRole"
if [ "$state" = 1 ]; then
  run aws iam create-role --role-name ecsInstanceRole --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > /dev/null
  run aws iam attach-role-policy --role-name ecsInstanceRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
  run aws iam create-instance-profile --instance-profile-name ecsInstanceRole > /dev/null
  run aws iam add-role-to-instance-profile --instance-profile-name ecsInstanceRole --role-name ecsInstanceRole
  echo "IAM: $(did ecsInstanceRole)"; sleep 10   # IAM propagation before the CE references it
else echo "IAM: ecsInstanceRole exists"; fi

# --- IAM: the service-linked roles a SPOT compute environment needs -----------
# Batch asks EC2 for spot capacity through AWSServiceRoleForEC2Spot. Without it
# the compute environment is created and then goes INVALID ("not authorized to
# perform: ec2:RequestSpotFleet" or a role-not-found), which only shows up as
# the job queue refusing to attach. Creating an existing one is an error we can
# ignore -- most accounts already have them from any past spot use.
for service in spot.amazonaws.com batch.amazonaws.com; do
  if [ "$DRY_RUN" = 1 ]; then echo "+ aws iam create-service-linked-role --aws-service-name $service"; continue; fi
  if aws iam create-service-linked-role --aws-service-name "$service" > /dev/null 2>&1; then
    echo "IAM: created the service-linked role for $service"
  fi
done

# --- Batch: compute environment -----------------------------------------------
# The default VPC's subnets, one per zone, so spot can diversify across pools.
#
# Only 32 GiB and 64 GiB shapes, which is what the pilot's bill turned on. A
# job asks for 2 vCPU and 7.5 GB, so an 8 vCPU box can run four of them -- but
# only if it carries 30 GB. The c5/c6i.2xlarge the fleet used have 16 GB, so
# memory capped them at two jobs and half of every compute-optimised vCPU we
# rented sat idle. 23 of the pilot's 38.5 instance-hours went to those shapes
# and overall vCPU utilisation came out at 46%. Dropping them is the single
# biggest cost lever in this file.
SUBNETS=$(aws ec2 describe-subnets --region "$REGION" --filters Name=default-for-az,Values=true \
  --query 'Subnets[].SubnetId' --output json)
SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=default \
  --query 'SecurityGroups[0].GroupId' --output text)
if ! aws batch describe-compute-environments --region "$REGION" --compute-environments mapsnap-cpu-spot \
     --query 'computeEnvironments[0].computeEnvironmentName' --output text 2>/dev/null | grep -q mapsnap-cpu-spot; then
  run aws batch create-compute-environment --region "$REGION" --cli-input-json "$(cat <<JSON
{
  "computeEnvironmentName": "mapsnap-cpu-spot",
  "type": "MANAGED",
  "state": "ENABLED",
  "computeResources": {
    "type": "SPOT",
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 256,
    "instanceTypes": ["m5.2xlarge", "m5a.2xlarge", "m6a.2xlarge", "m6i.2xlarge",
                      "r5.2xlarge", "r6i.2xlarge"],
    "subnets": $SUBNETS,
    "securityGroupIds": ["$SG"],
    "instanceRole": "ecsInstanceRole",
    "tags": {"project": "mapsnap-craft"}
  }
}
JSON
)" > /dev/null
  echo "Batch: $(did "compute environment mapsnap-cpu-spot")"
else
  # Already there: bring its instance types up to date, so a rerun of this
  # script is how the shape list changes rather than a hand-edited console.
  run aws batch update-compute-environment --region "$REGION" \
    --compute-environment mapsnap-cpu-spot \
    --compute-resources "$(printf '{"instanceTypes":["m5.2xlarge","m5a.2xlarge","m6a.2xlarge","m6i.2xlarge","r5.2xlarge","r6i.2xlarge"],"maxvCpus":256,"minvCpus":0}')" > /dev/null
  echo "Batch: compute environment exists (instance types refreshed)"
fi

# --- Batch: wait for the compute environment -----------------------------------
# create-compute-environment returns while the environment is still CREATING;
# create-job-queue refuses anything but VALID ("It must be valid before
# attaching it to the job queue"), so poll until it settles. An INVALID
# environment is a configuration fault, not a delay: report the reason Batch
# gives and stop, because re-running cannot repair one -- it has to be deleted
# and recreated.
wait_for_compute_environment() {
  local name=$1 status reason
  for _ in $(seq 60); do
    status=$(aws batch describe-compute-environments --region "$REGION" --compute-environments "$name" \
      --query 'computeEnvironments[0].status' --output text 2>/dev/null)
    case "$status" in
      VALID) echo "Batch: compute environment $name is valid"; return 0 ;;
      INVALID)
        reason=$(aws batch describe-compute-environments --region "$REGION" --compute-environments "$name" \
          --query 'computeEnvironments[0].statusReason' --output text 2>/dev/null)
        echo "compute environment $name is INVALID: $reason" >&2
        echo "fix the cause, then delete and recreate it:" >&2
        echo "  aws batch update-compute-environment --region $REGION --compute-environment $name --state DISABLED" >&2
        echo "  aws batch delete-compute-environment --region $REGION --compute-environment $name" >&2
        echo "  $0" >&2
        return 1 ;;
    esac
    sleep 5
  done
  echo "compute environment $name still $status after 5 minutes" >&2
  return 1
}
if [ "$DRY_RUN" = 1 ]; then echo "+ wait for compute environment mapsnap-cpu-spot to be VALID"
else wait_for_compute_environment mapsnap-cpu-spot; fi

# --- Batch: job queue ---------------------------------------------------------
if ! aws batch describe-job-queues --region "$REGION" --job-queues mapsnap-fit \
     --query 'jobQueues[0].jobQueueName' --output text 2>/dev/null | grep -q mapsnap-fit; then
  run aws batch create-job-queue --region "$REGION" --job-queue-name mapsnap-fit --priority 1 --state ENABLED \
    --compute-environment-order order=1,computeEnvironment=mapsnap-cpu-spot > /dev/null
  echo "Batch: $(did "job queue mapsnap-fit")"
else echo "Batch: job queue exists"; fi
fi

# --- Batch: the loc-fit job definitions ----------------------------------------
# One job = one item: child N of an array runs line N of Ref::items (loc-fit
# reads AWS_BATCH_JOB_ARRAY_INDEX).
#
# Memory, measured over the 200-item pilot (2026-09-19): peak stage RSS was
# 1.8 GB median, 3.3 GB at p90, 5.7 GB at p99 and 6.6 GB at the worst success,
# and three items -- big-city volumes whose OCR vocabulary runs to 40,000 name
# forms -- were SIGKILLed at the 7,000 MB ceiling.
#
# The ceiling is set by packing, not by the p99: four jobs must fit on a 32 GB
# instance or the fourth vCPU pair goes to waste, which costs far more than
# the handful of items a bigger ceiling would rescue. 7,500 leaves 900 MB over
# the worst success and still packs four; 8,192 would drop an m5.2xlarge to
# three jobs and a 16 GB instance to one. The three that need more go to the
# -large definition below, which packs three on an r5.2xlarge.
#
# The retry policy reads loc-fit's exit codes: 3 (unprocessable) and 4 (inputs
# missing) are never retried, a spot reclamation always is, anything else
# once. The pilot lost two instances to spot mid-run; all eight of their
# children retried and succeeded.
register_fit_definition() {
  local name=$1 memory=$2
  run aws batch register-job-definition --region "$REGION" --cli-input-json "$(cat <<JSON
{
  "jobDefinitionName": "$name",
  "type": "container",
  "platformCapabilities": ["EC2"],
  "parameters": {
    "items": "s3://mapsnap-sanborn/_runs/UNSET/items.txt",
    "runTag": "UNSET",
    "bucket": "s3://mapsnap-sanborn",
    "counties": "s3://mapsnap-sanborn/_craft/items.tsv",
    "cityCounties": "s3://mapsnap-sanborn/_craft/city-items.tsv",
    "itemsPerJob": "1"
  },
  "containerProperties": {
    "image": "$IMAGE",
    "command": ["loc-fit", "--items", "Ref::items", "--items-per-job", "Ref::itemsPerJob",
                "--run-tag", "Ref::runTag", "--bucket", "Ref::bucket",
                "--counties", "Ref::counties", "Ref::cityCounties"],
    "jobRoleArn": "arn:aws:iam::$ACCOUNT:role/$JOB_ROLE",
    "resourceRequirements": [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "$memory"}],
    "environment": [{"name": "OMP_NUM_THREADS", "value": "2"}, {"name": "AWS_REGION", "value": "$REGION"},
                    {"name": "PYTHONHASHSEED", "value": "0"}]
  },
  "retryStrategy": {
    "attempts": 2,
    "evaluateOnExit": [
      {"onStatusReason": "Host EC2*", "action": "RETRY"},
      {"onExitCode": "3", "action": "EXIT"},
      {"onExitCode": "4", "action": "EXIT"},
      {"onReason": "*", "action": "RETRY"}
    ]
  },
  "timeout": {"attemptDurationSeconds": 10800}
}
JSON
)" --query 'jobDefinitionArn' --output text
}

register_fit_definition mapsnap-loc-fit 7500
register_fit_definition mapsnap-loc-fit-large 16384

echo "done. Next: scripts/batch/push-image.sh, then scripts/batch/submit.sh <run-tag> <items.txt>"
