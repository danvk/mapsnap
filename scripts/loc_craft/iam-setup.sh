#!/bin/bash
# One-time IAM setup for the corpus CRAFT + P(road) fleet (#354). Run with ADMIN
# credentials:
#
#   aws login
#   scripts/loc_craft/iam-setup.sh
#
# Creates (idempotently):
#   * role + instance profile `mapsnap-craft`: read the mirror bucket and write
#     the sidecars back into it, plus SSM Session Manager for a live shell.
#   * EC2's service-linked role for Spot, if the account has never used spot.
#   * inline policy `mapsnap-craft-launch` on the long-lived `mapsnap-mirror`
#     user: launch/describe/terminate instances, pass the role above, read the
#     Deep Learning AMI parameters, and read *and raise* EC2 quotas (a G-family
#     vCPU increase is the recurring blocker, and needing the admin session for
#     each attempt is the only reason it waits).
#
# The instance role can write only under by-state/ and _craft/, so a runaway
# worker cannot touch the manifest or the README at the bucket root.
set -euo pipefail

BUCKET_NAME=${BUCKET_NAME:-mapsnap-sanborn}
USER_NAME=${USER_NAME:-mapsnap-mirror}
ROLE=mapsnap-craft
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "account $ACCOUNT, bucket $BUCKET_NAME, user $USER_NAME"

if ! aws iam get-role --role-name AWSServiceRoleForEC2Spot > /dev/null 2>&1; then
  aws iam create-service-linked-role --aws-service-name spot.amazonaws.com > /dev/null
  echo "created service-linked role AWSServiceRoleForEC2Spot"
fi

TRUST=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
  "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
EOF
)
INSTANCE_POLICY=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
   "Resource": "arn:aws:s3:::$BUCKET_NAME"},
  {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::$BUCKET_NAME/*"},
  {"Effect": "Allow", "Action": ["s3:PutObject", "s3:DeleteObject"],
   "Resource": ["arn:aws:s3:::$BUCKET_NAME/by-state/*", "arn:aws:s3:::$BUCKET_NAME/_craft/*"]},
  {"Effect": "Allow", "Action": [
     "sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility",
     "sqs:GetQueueAttributes", "sqs:GetQueueUrl"],
   "Resource": "arn:aws:sqs:*:$ACCOUNT:mapsnap-*"}
]}
EOF
)
LAUNCH_POLICY=$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": [
     "ec2:RunInstances", "ec2:TerminateInstances", "ec2:CreateTags",
     "ec2:DescribeInstances", "ec2:DescribeInstanceStatus", "ec2:DescribeImages",
     "ec2:DescribeInstanceTypes", "ec2:DescribeInstanceTypeOfferings",
     "ec2:DescribeSpotPriceHistory", "ec2:DescribeSpotInstanceRequests",
     "ec2:DescribeSubnets", "ec2:DescribeVpcs", "ec2:DescribeSecurityGroups",
     "ec2:GetConsoleOutput"],
   "Resource": "*"},
  {"Effect": "Allow", "Action": "iam:PassRole",
   "Resource": "arn:aws:iam::$ACCOUNT:role/$ROLE"},
  {"Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
   "Resource": "arn:aws:ssm:*::parameter/aws/service/deeplearning/*"},
  {"Effect": "Allow", "Action": ["ssm:StartSession", "ssm:TerminateSession", "ssm:DescribeInstanceInformation"],
   "Resource": "*"},
  {"Effect": "Allow", "Action": ["servicequotas:GetServiceQuota", "servicequotas:ListServiceQuotas",
                                 "servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota",
                                 "servicequotas:RequestServiceQuotaIncrease"],
   "Resource": "*"},
  {"Effect": "Allow", "Action": [
     "sqs:CreateQueue", "sqs:SendMessage", "sqs:GetQueueAttributes", "sqs:SetQueueAttributes",
     "sqs:GetQueueUrl", "sqs:ListQueues", "sqs:ReceiveMessage", "sqs:DeleteMessage",
     "sqs:PurgeQueue", "sqs:DeleteQueue"],
   "Resource": "*"}
]}
EOF
)

if ! aws iam get-role --role-name "$ROLE" > /dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" > /dev/null
  echo "created role $ROLE"
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name "$ROLE-s3" --policy-document "$INSTANCE_POLICY"
aws iam attach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" > /dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" > /dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  echo "created instance profile $ROLE (allow ~10 s before the first launch)"
fi
# A customer-managed policy, not an inline one: IAM caps the *aggregate* size of
# a user's inline policies at 2048 bytes, and this one plus the sizing
# benchmark's `mapsnap-bench-launch` exceeds that. Managed policies have their
# own 6144-byte budget and can be detached in one call.
POLICY_ARN="arn:aws:iam::$ACCOUNT:policy/$ROLE-launch"
if aws iam get-policy --policy-arn "$POLICY_ARN" > /dev/null 2>&1; then
  # Five versions per policy is the hard limit, so clear the old ones first.
  for version in $(aws iam list-policy-versions --policy-arn "$POLICY_ARN" \
      --query 'Versions[?!IsDefaultVersion].VersionId' --output text); do
    aws iam delete-policy-version --policy-arn "$POLICY_ARN" --version-id "$version"
  done
  aws iam create-policy-version --policy-arn "$POLICY_ARN" \
    --policy-document "$LAUNCH_POLICY" --set-as-default > /dev/null
  echo "updated managed policy $ROLE-launch"
else
  aws iam create-policy --policy-name "$ROLE-launch" \
    --policy-document "$LAUNCH_POLICY" > /dev/null
  echo "created managed policy $ROLE-launch"
fi
aws iam attach-user-policy --user-name "$USER_NAME" --policy-arn "$POLICY_ARN"
# An inline copy from an earlier version of this script would eat the user's
# 2048-byte inline budget for nothing.
aws iam delete-user-policy --user-name "$USER_NAME" --policy-name "$ROLE-launch" 2> /dev/null || true
echo "granted $USER_NAME launch rights (managed policy $ROLE-launch)"
echo
echo "If PutUserPolicy ever fails with LimitExceeded, an inline policy is using"
echo "the user's 2048-byte budget; the sizing benchmark left one:"
echo "  aws iam delete-user-policy --user-name $USER_NAME --policy-name mapsnap-bench-launch"
