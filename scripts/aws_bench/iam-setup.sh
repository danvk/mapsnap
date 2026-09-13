#!/bin/bash
# One-time IAM setup for the benchmark (#354). Run with ADMIN credentials:
#
#   aws login                      # the short-lived admin session
#   scripts/aws_bench/iam-setup.sh
#
# Creates (idempotently):
#   * role + instance profile `mapsnap-bench`: what the instance itself may do. Read the
#     whole mirror bucket, write only under the _bench/ prefix, and register with SSM
#     Session Manager so a stuck instance can be inspected without SSH keys.
#   * inline policy `mapsnap-bench-launch` on the long-lived `mapsnap-mirror` IAM user:
#     launch/describe/terminate EC2 instances, pass the role above to them, look up the
#     Deep Learning AMI parameters, open SSM sessions, and read/write _bench/ in S3.
#
# Nothing here touches the user's existing mirror policy.
set -euo pipefail

BUCKET=${BUCKET:-mapsnap-sanborn}
PREFIX=${PREFIX:-_bench}
USER_NAME=${USER_NAME:-mapsnap-mirror}
ROLE=mapsnap-bench
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "account $ACCOUNT, bucket $BUCKET, user $USER_NAME"

# A spot request needs EC2's service-linked role for Spot, which an account gets
# the first time an admin creates it. Without it the launcher's first spot request
# fails with AuthFailure.ServiceLinkedRoleCreationNotPermitted.
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
   "Resource": "arn:aws:s3:::$BUCKET"},
  {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::$BUCKET/*"},
  {"Effect": "Allow", "Action": ["s3:PutObject", "s3:DeleteObject"],
   "Resource": "arn:aws:s3:::$BUCKET/$PREFIX/*"}
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
     "ec2:DescribeSubnets", "ec2:DescribeVpcs",
     "ec2:DescribeSecurityGroups", "ec2:GetConsoleOutput"],
   "Resource": "*"},
  {"Effect": "Allow", "Action": "iam:PassRole",
   "Resource": "arn:aws:iam::$ACCOUNT:role/$ROLE"},
  {"Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
   "Resource": "arn:aws:ssm:*::parameter/aws/service/deeplearning/*"},
  {"Effect": "Allow", "Action": ["ssm:StartSession", "ssm:TerminateSession", "ssm:DescribeInstanceInformation"],
   "Resource": "*"},
  {"Effect": "Allow", "Action": ["servicequotas:GetServiceQuota", "servicequotas:ListServiceQuotas",
                                 "servicequotas:ListRequestedServiceQuotaChangeHistoryByQuota"],
   "Resource": "*"},
  {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::$BUCKET"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
   "Resource": "arn:aws:s3:::$BUCKET/$PREFIX/*"}
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
aws iam put-user-policy --user-name "$USER_NAME" --policy-name "$ROLE-launch" \
  --policy-document "$LAUNCH_POLICY"
echo "granted $USER_NAME launch rights (policy $ROLE-launch)"
