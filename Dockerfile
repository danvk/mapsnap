# The corpus worker as an image: everything scripts/loc_craft/bootstrap.sh
# builds at boot, built once instead. One image serves both jobs -- loc-craft
# on GPU instances, loc-fit on CPU -- because the lockfile's Linux torch is a
# CUDA build that falls back to CPU when no device is present.
#
# Build for EC2 from an Apple Silicon Mac (the builder emulates amd64):
#
#   docker buildx build --platform linux/amd64 \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     -t mapsnap:$(git rev-parse --short HEAD) --load .
#
# Run a job exactly as bootstrap.sh does, the job and its flags as arguments:
#
#   docker run --rm -e AWS_REGION=us-west-2 mapsnap:abc1234 \
#     loc-fit --queue https://sqs... --run-tag v1.3 --counties s3://... s3://...
#
# Layers are ordered so the expensive ones survive a code change: base + apt,
# then the dependency set from pyproject.toml/uv.lock alone (torch, easyocr,
# ~2 GB), then the EasyOCR weights, and only then the source. Editing a .py
# file rebuilds the last layer only.

FROM python:3.12-slim-bookworm

# What bootstrap.sh apt-installs, minus git: the source is COPYed in, not
# cloned, and .git is 963 MB. libgl1/libglib2.0-0 are OpenCV's runtime;
# libopenjp2-tools decodes the mirror's JP2s.
RUN apt-get update -q \
    && apt-get install -y -q --no-install-recommends \
        ca-certificates curl unzip libgl1 libglib2.0-0 libopenjp2-tools \
    && rm -rf /var/lib/apt/lists/*

# The workers shell out to `aws s3 cp/sync` (loc_craft.run_aws), so the CLI
# is part of the runtime, not a build tool. v2 from the same archive
# bootstrap.sh fetches; the platform is amd64 by construction (see above).
RUN curl -sSf https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip

COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /uvx /bin/

# Bytecode compiled at build so the first import in a job is not the slow
# one; copy rather than hardlink because the cache mount is a different FS.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# /app is the repo root and stays the working directory: keymap wants
# models/number_detector.pt by a path relative to it (see loc_fit.stage).
WORKDIR /app

# Dependencies from the lockfile alone, before any source is present, so this
# layer is keyed on pyproject.toml + uv.lock and nothing else.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# bootstrap.sh's driver check, at build time: the locked Linux torch is the
# CUDA 13 build, which needs driver >= 580 on the host. An AMI with an older
# driver gets an image built with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
# instead, which is the wheel bootstrap.sh swaps to. Empty (the default) keeps
# the lockfile's build.
ARG TORCH_INDEX=""
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_INDEX" ]; then \
        uv pip install --reinstall --index-url "$TORCH_INDEX" torch torchvision; \
    fi

# EasyOCR downloads its recognizer weights on first use; fetch them here so a
# job never does, and so a fleet of jobs does not hit the model host at once.
RUN /app/.venv/bin/python -c \
    "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" > /dev/null

# The source, and the project install that points at it. models/ is tracked in
# git (~92 MB of weights) and comes in with the source; .dockerignore keeps
# data/ (19 GB), .venv, app/ and the rest out.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# The commit this image was built from, for provenance: git_head_info returns
# empty fields outside a checkout, so the worker reads this instead.
ARG GIT_SHA=unknown
ENV MAPSNAP_GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.source=https://github.com/danvk/mapsnap

# One torch thread per job by default. A container sees the HOST's core count,
# so torch's own default oversubscribes a box running several jobs -- the bug
# #444 fixed in bootstrap.sh by splitting cores between workers. A Batch job
# definition that grants N vCPUs sets OMP_NUM_THREADS=N in its environment.
ENV OMP_NUM_THREADS=1 \
    PATH=/app/.venv/bin:$PATH

# The job name is the first argument -- loc-fit, loc-craft -- then its flags,
# exactly the `uv run mapsnap "$JOB" ...` line in bootstrap.sh. No default
# command: a Batch job definition supplies it.
ENTRYPOINT ["mapsnap"]
