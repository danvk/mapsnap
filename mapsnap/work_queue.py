"""An SQS work queue, so instances need not agree on a partition (#354).

Static sharding makes every instance's share of the corpus a launch-time
decision: resizing the fleet means tearing it down and re-partitioning, and a
shard whose instance never launched is a hole nobody fills. Capacity, meanwhile,
arrives and disappears on AWS's schedule rather than ours.

A queue removes the decision. Workers become identical and interchangeable:
start one or twenty, in any region, at any moment, and each takes the next item.

The property that makes this cheap here is that the jobs are already idempotent
-- completion lives in the S3 sidecars -- so at-least-once delivery, which is
all SQS promises, is all they need. A redelivered item costs one wasted listing.

How a message moves:

  * ``receive`` hands it over and hides it from other consumers for
    ``visibility`` seconds. This is a lease, not a delivery interval: throughput
    is unrelated to it.
  * ``delete`` within the lease retires it for good.
  * A worker that dies without deleting loses nothing -- the lease lapses and
    the next worker picks the item up.

So ``visibility`` must exceed the longest an item can take, or a still-working
instance has its item handed to someone else. The corpus's longest are 90-page
volumes carrying a key-map sheet tiled at native resolution, ~57 s per page on
an L4, hence the 30-minute default.

An item that kills its worker would otherwise cycle forever, which is how one
bad TIFF stalled four loc-raw shards. ``max_receives`` sends it to a dead-letter
queue instead, where it can be looked at rather than retried blindly.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from mapsnap.aws_cli import run_aws

DEFAULT_VISIBILITY_SECONDS = 1800
DEFAULT_MAX_RECEIVES = 3
# SQS caps a send batch at 10 messages.
SEND_BATCH = 10


@dataclass(frozen=True)
class Message:
    """One item handed to this worker, and the handle that retires it."""

    body: str
    handle: str


@dataclass(frozen=True)
class Depth:
    """What is left: waiting to be taken, and currently leased out."""

    visible: int
    in_flight: int

    @property
    def total(self) -> int:
        return self.visible + self.in_flight


def queue_region(url: str) -> str | None:
    """The region baked into an SQS queue URL, or None if it is not one.

    Every call has to name this region explicitly. The CLI otherwise signs
    against whatever region the caller defaults to -- on an instance, the one
    IMDS reports -- and SQS then looks for the queue in the wrong region and
    answers ``NonExistentQueue``. That is exactly what killed both us-east-2
    workers on 2026-09-15 while the us-west-2 four, whose default happened to
    match, ran fine.
    """
    match = re.search(r"sqs\.([a-z0-9-]+)\.amazonaws\.com", url)
    return match.group(1) if match else None


def sqs_command(url: str, operation: str, *args: str) -> list[str]:
    """An ``aws sqs`` command pinned to the queue's own region."""
    command = ["aws", "sqs", operation]
    region = queue_region(url)
    if region:
        command += ["--region", region]
    return [*command, *args]


def create_queue(
    name: str,
    *,
    visibility: int = DEFAULT_VISIBILITY_SECONDS,
    max_receives: int = DEFAULT_MAX_RECEIVES,
) -> tuple[str, str]:
    """Create the queue and its dead-letter queue; return both URLs.

    Idempotent: SQS returns the existing URL when the attributes match, so this
    is safe to re-run.
    """
    dead = run_aws(
        [
            "aws",
            "sqs",
            "create-queue",
            "--queue-name",
            f"{name}-dead",
            "--output",
            "text",
            "--query",
            "QueueUrl",
        ],
        capture=True,
    ).stdout.strip()
    arn = run_aws(
        sqs_command(
            dead,
            "get-queue-attributes",
            "--queue-url",
            dead,
            "--attribute-names",
            "QueueArn",
            "--output",
            "text",
            "--query",
            "Attributes.QueueArn",
        ),
        capture=True,
    ).stdout.strip()
    policy = json.dumps({"deadLetterTargetArn": arn, "maxReceiveCount": max_receives})
    attributes = json.dumps(
        {"VisibilityTimeout": str(visibility), "RedrivePolicy": policy}
    )
    url = run_aws(
        [
            "aws",
            "sqs",
            "create-queue",
            "--queue-name",
            name,
            "--attributes",
            attributes,
            "--output",
            "text",
            "--query",
            "QueueUrl",
        ],
        capture=True,
    ).stdout.strip()
    return url, dead


def batches(names: list[str], size: int = SEND_BATCH) -> list[list[str]]:
    """Split the fill into batches SQS will accept."""
    return [names[i : i + size] for i in range(0, len(names), size)]


def fill_queue(url: str, names: list[str]) -> int:
    """Send one message per item; return how many were sent.

    Ids are the item's position in the batch, which only has to be unique
    within the request.
    """
    sent = 0
    for batch in batches(names):
        entries = json.dumps(
            [
                {"Id": str(index), "MessageBody": name}
                for index, name in enumerate(batch)
            ]
        )
        run_aws(
            sqs_command(
                url, "send-message-batch", "--queue-url", url, "--entries", entries
            ),
            capture=True,
        )
        sent += len(batch)
    return sent


def receive(url: str, *, count: int = 1, wait: int = 20) -> list[Message]:
    """Take up to ``count`` items, long-polling so an empty queue is not a spin.

    An empty list means the queue looks drained; with long polling that is a
    reasonably strong signal, though items leased to other workers may still
    come back if those workers die.
    """
    result = run_aws(
        sqs_command(
            url,
            "receive-message",
            "--queue-url",
            url,
            "--max-number-of-messages",
            str(count),
            "--wait-time-seconds",
            str(wait),
            "--output",
            "json",
        ),
        capture=True,
    ).stdout.strip()
    if not result:
        return []
    payload = json.loads(result)
    return [
        Message(body=m["Body"], handle=m["ReceiptHandle"])
        for m in payload.get("Messages", [])
    ]


def delete(url: str, handle: str) -> None:
    """Retire an item so no other worker sees it."""
    run_aws(
        sqs_command(
            url, "delete-message", "--queue-url", url, "--receipt-handle", handle
        ),
        capture=True,
    )


def extend(url: str, handle: str, seconds: int) -> None:
    """Push this item's lease out, for work that outruns the visibility timeout."""
    run_aws(
        sqs_command(
            url,
            "change-message-visibility",
            "--queue-url",
            url,
            "--receipt-handle",
            handle,
            "--visibility-timeout",
            str(seconds),
        ),
        capture=True,
    )


def depth(url: str) -> Depth:
    """How many items are waiting, and how many are leased out right now."""
    result = run_aws(
        sqs_command(
            url,
            "get-queue-attributes",
            "--queue-url",
            url,
            "--attribute-names",
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
            "--output",
            "json",
        ),
        capture=True,
    ).stdout.strip()
    attributes = json.loads(result).get("Attributes", {}) if result else {}
    return Depth(
        visible=int(attributes.get("ApproximateNumberOfMessages", 0)),
        in_flight=int(attributes.get("ApproximateNumberOfMessagesNotVisible", 0)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage a corpus work queue.")
    sub = parser.add_subparsers(dest="command", required=True)

    made = sub.add_parser("create", help="Create the queue and its dead-letter queue.")
    made.add_argument("--name", required=True, help="Queue name, e.g. mapsnap-craft.")
    made.add_argument(
        "--visibility",
        type=int,
        default=DEFAULT_VISIBILITY_SECONDS,
        help="Lease seconds; must exceed the longest item (default: %(default)s).",
    )
    made.add_argument(
        "--max-receives",
        type=int,
        default=DEFAULT_MAX_RECEIVES,
        help="Attempts before an item goes to the dead-letter queue (default: %(default)s).",
    )

    filled = sub.add_parser("fill", help="Send one message per manifest item.")
    filled.add_argument("--url", required=True)
    filled.add_argument(
        "--bucket",
        default="s3://mapsnap-sanborn",
        help="Mirror bucket (default: %(default)s).",
    )
    filled.add_argument("--manifest", help="Sheet manifest (default: the bucket's).")
    filled.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/loc-craft"),
        help="Scratch directory.",
    )
    filled.add_argument("--limit", type=int, help="Send only this many (a pilot).")

    status = sub.add_parser("status", help="Report what is left.")
    status.add_argument("--url", required=True)

    args = parser.parse_args(argv)

    if args.command == "create":
        url, dead = create_queue(
            args.name, visibility=args.visibility, max_receives=args.max_receives
        )
        print(f"queue      {url}")
        print(f"dead-letter {dead}")
        return 0

    if args.command == "fill":
        # Imported here, not at module scope: loc_craft imports this module, and
        # the queue itself must not depend on the job that consumes it.
        from mapsnap.loc_craft import read_manifest, resolve_manifest

        manifest = resolve_manifest(args.manifest, args.bucket, args.work_dir)
        names = [item.item for item in read_manifest(manifest)]
        if args.limit:
            names = names[: args.limit]
        sent = fill_queue(args.url, names)
        print(f"sent {sent:,} items")
        return 0

    left = depth(args.url)
    print(
        f"{left.visible:,} waiting, {left.in_flight:,} in flight, {left.total:,} left"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
