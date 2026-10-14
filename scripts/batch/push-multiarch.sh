#!/bin/bash
# Build the image for amd64 AND arm64 and push both to ECR under one tag (#563).
#
#   scripts/batch/push-multiarch.sh            # mapsnap:<short sha of HEAD> and :latest
#
# One tag that names a manifest list: each Batch host pulls the variant for its
# own architecture, so the job definitions stay shared between the x86 queue
# (mapsnap-fit) and the Graviton one (mapsnap-fit-arm). push-image.sh pushes a
# single-architecture image instead; use it when only x86 matters.
#
# Multi-platform builds need a docker-container builder (the default Docker
# Desktop builder can only --load one platform); one named mapsnap-multiarch is
# created on first use. On an Apple Silicon Mac the arm64 half builds natively
# and the amd64 half under emulation, so the amd64 half is the slow one. The
# first push of the arm64 layers moves ~2.3 GB.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/mapsnap
SHA=$(git rev-parse --short HEAD)

if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is dirty; the image tagged $SHA would not be that commit" >&2
  exit 1
fi
docker buildx inspect mapsnap-multiarch > /dev/null 2>&1 \
  || docker buildx create --name mapsnap-multiarch --driver docker-container > /dev/null
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com" > /dev/null
docker buildx build --builder mapsnap-multiarch \
  --platform linux/amd64,linux/arm64 \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t "$REPO:$SHA" -t "$REPO:latest" --push .
echo "pushed $REPO:$SHA (and :latest) for linux/amd64 and linux/arm64"
