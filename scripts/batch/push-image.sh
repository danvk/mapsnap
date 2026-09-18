#!/bin/bash
# Push a locally built image to ECR under its commit, and as :latest.
#
#   scripts/batch/push-image.sh                # pushes mapsnap:<short sha of HEAD>
#   scripts/batch/push-image.sh mapsnap:dev    # pushes that local tag instead
#
# Build first (see the Dockerfile header). The first push moves the whole
# 2.3 GB image -- ~40 minutes on the ~1 MB/s uplink measured 2026-09-18 --
# after that only changed layers move: a code change is the ~90 MB source
# layer, a lockfile change the dependency layer. Pushing from CI would need
# an OIDC role for GitHub Actions; not worth it for the pilot.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/mapsnap
SHA=$(git rev-parse --short HEAD)
LOCAL=${1:-mapsnap:$SHA}

docker image inspect "$LOCAL" > /dev/null 2>&1 || { echo "no local image $LOCAL; build it first" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ] && [ "$LOCAL" = "mapsnap:$SHA" ]; then
  echo "working tree is dirty; the image tagged $SHA may not be that commit" >&2
fi

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com" > /dev/null
docker tag "$LOCAL" "$REPO:$SHA"
docker tag "$LOCAL" "$REPO:latest"
docker push "$REPO:$SHA"
docker push "$REPO:latest"
echo "pushed $REPO:$SHA (and :latest)"
