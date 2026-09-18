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
DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1
IMAGE=${IMAGE:-$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/mapsnap:latest}
run() { if [ "$DRY_RUN" = 1 ]; then echo "+ $*"; else "$@"; fi; }

echo "account $ACCOUNT, region $REGION, image $IMAGE"

# --- ECR ----------------------------------------------------------------------
if ! aws ecr describe-repositories --repository-names mapsnap --region "$REGION" > /dev/null 2>&1; then
  run aws ecr create-repository --repository-name mapsnap --region "$REGION" \
    --image-scanning-configuration scanOnPush=false > /dev/null
  run aws ecr put-lifecycle-policy --repository-name mapsnap --region "$REGION" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep the last 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' > /dev/null
  echo "ECR: created mapsnap"
else echo "ECR: mapsnap exists"; fi

# --- IAM: the job role (what the container is) --------------------------------
# Trusted by ECS tasks, carrying whatever mapsnap-craft carries today, so a job
# can read the mirror and write a run's outputs exactly as an instance did.
JOB_ROLE=mapsnap-batch-job
if ! aws iam get-role --role-name $JOB_ROLE > /dev/null 2>&1; then
  run aws iam create-role --role-name $JOB_ROLE --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > /dev/null
  for arn in $(aws iam list-attached-role-policies --role-name mapsnap-craft --query 'AttachedPolicies[].PolicyArn' --output text); do
    run aws iam attach-role-policy --role-name $JOB_ROLE --policy-arn "$arn"
  done
  for name in $(aws iam list-role-policies --role-name mapsnap-craft --query 'PolicyNames[]' --output text); do
    doc=$(aws iam get-role-policy --role-name mapsnap-craft --policy-name "$name" --query PolicyDocument --output json)
    run aws iam put-role-policy --role-name $JOB_ROLE --policy-name "$name" --policy-document "$doc"
  done
  echo "IAM: created $JOB_ROLE with mapsnap-craft's policies"
else echo "IAM: $JOB_ROLE exists"; fi

# --- IAM: the instance role (what the EC2 host under ECS is) ------------------
if ! aws iam get-instance-profile --instance-profile-name ecsInstanceRole > /dev/null 2>&1; then
  run aws iam create-role --role-name ecsInstanceRole --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > /dev/null
  run aws iam attach-role-policy --role-name ecsInstanceRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
  run aws iam create-instance-profile --instance-profile-name ecsInstanceRole > /dev/null
  run aws iam add-role-to-instance-profile --instance-profile-name ecsInstanceRole --role-name ecsInstanceRole
  echo "IAM: created ecsInstanceRole"; sleep 10   # IAM propagation before the CE references it
else echo "IAM: ecsInstanceRole exists"; fi

# --- Batch: compute environment -----------------------------------------------
# The default VPC's subnets, one per zone, so spot can diversify across pools;
# the type list is what launch.sh rotated through, plus the memory-heavy r5s.
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
                      "c5.2xlarge", "c6a.2xlarge", "c6i.2xlarge", "r5.2xlarge", "r6i.2xlarge"],
    "subnets": $SUBNETS,
    "securityGroupIds": ["$SG"],
    "instanceRole": "ecsInstanceRole",
    "tags": {"project": "mapsnap-craft"}
  }
}
JSON
)" > /dev/null
  echo "Batch: created compute environment mapsnap-cpu-spot"
else echo "Batch: compute environment exists"; fi

# --- Batch: job queue ---------------------------------------------------------
if ! aws batch describe-job-queues --region "$REGION" --job-queues mapsnap-fit \
     --query 'jobQueues[0].jobQueueName' --output text 2>/dev/null | grep -q mapsnap-fit; then
  run aws batch create-job-queue --region "$REGION" --job-queue-name mapsnap-fit --priority 1 --state ENABLED \
    --compute-environment-order order=1,computeEnvironment=mapsnap-cpu-spot > /dev/null
  echo "Batch: created job queue mapsnap-fit"
else echo "Batch: job queue exists"; fi

# --- Batch: the loc-fit job definition ----------------------------------------
# One job = one item: child N of an array runs line N of Ref::items (loc-fit
# reads AWS_BATCH_JOB_ARRAY_INDEX). 2 vCPU / 7 GB: the key-map georef peaked
# at 2.5 GB on Miami and OCR runs beside it; loc-fit prints the peak stage RSS
# so this can be tightened after the pilot. The retry policy reads loc-fit's
# exit codes: 3 (unprocessable) and 4 (inputs missing) are never retried, a
# spot reclamation always is, anything else once.
run aws batch register-job-definition --region "$REGION" --cli-input-json "$(cat <<JSON
{
  "jobDefinitionName": "mapsnap-loc-fit",
  "type": "container",
  "platformCapabilities": ["EC2"],
  "parameters": {
    "items": "s3://mapsnap-sanborn/_runs/UNSET/items.txt",
    "runTag": "UNSET",
    "bucket": "s3://mapsnap-sanborn",
    "counties": "s3://mapsnap-sanborn/_craft/items.tsv",
    "cityCounties": "s3://mapsnap-sanborn/_craft/city-items.tsv"
  },
  "containerProperties": {
    "image": "$IMAGE",
    "command": ["loc-fit", "--items", "Ref::items", "--run-tag", "Ref::runTag",
                "--bucket", "Ref::bucket", "--counties", "Ref::counties", "Ref::cityCounties"],
    "jobRoleArn": "arn:aws:iam::$ACCOUNT:role/$JOB_ROLE",
    "resourceRequirements": [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "7000"}],
    "environment": [{"name": "OMP_NUM_THREADS", "value": "2"}, {"name": "AWS_REGION", "value": "$REGION"}]
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
echo "done. Next: scripts/batch/push-image.sh, then scripts/batch/submit.sh <run-tag> <items.txt>"
