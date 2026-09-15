"""Tests for the SQS work queue (mapsnap.work_queue)."""

import json
import subprocess

import pytest

from mapsnap import work_queue
from mapsnap.work_queue import (
    Depth,
    Message,
    batches,
    create_queue,
    delete,
    depth,
    extend,
    fill_queue,
    receive,
)


class FakeAws:
    """Records the aws commands issued and replays canned stdout."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.replies = replies or []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(command)
        out = self.replies.pop(0) if self.replies else ""
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    def flag(self, call: int, name: str) -> str:
        """The value following a flag in one recorded call."""
        command = self.calls[call]
        return command[command.index(name) + 1]


def test_batches_splits_at_the_sqs_limit() -> None:
    assert batches(list("abc"), size=2) == [["a", "b"], ["c"]]
    assert len(batches([str(i) for i in range(25)])) == 3
    assert batches([]) == []


def test_fill_queue_sends_every_item_in_batches(monkeypatch) -> None:
    fake = FakeAws()
    monkeypatch.setattr(work_queue, "run_aws", fake)
    names = [f"sanborn{i:05d}" for i in range(23)]
    assert fill_queue("https://q", names) == 23
    assert len(fake.calls) == 3  # 10 + 10 + 3
    entries = json.loads(fake.flag(0, "--entries"))
    assert len(entries) == 10
    assert entries[0]["MessageBody"] == "sanborn00000"
    assert {e["Id"] for e in entries} == {str(i) for i in range(10)}


def test_receive_parses_messages_and_survives_an_empty_queue(monkeypatch) -> None:
    payload = json.dumps(
        {
            "Messages": [
                {"Body": "sanborn1", "ReceiptHandle": "h1"},
                {"Body": "sanborn2", "ReceiptHandle": "h2"},
            ]
        }
    )
    fake = FakeAws([payload, "", "{}"])
    monkeypatch.setattr(work_queue, "run_aws", fake)
    assert receive("https://q", count=2) == [
        Message("sanborn1", "h1"),
        Message("sanborn2", "h2"),
    ]
    assert receive("https://q") == []  # the CLI prints nothing on an empty queue
    assert receive("https://q") == []  # ... and sometimes an object with no key


def test_receive_long_polls_by_default(monkeypatch) -> None:
    """Without the wait, an idle worker spins on empty receives."""
    fake = FakeAws(["{}"])
    monkeypatch.setattr(work_queue, "run_aws", fake)
    receive("https://q")
    assert fake.flag(0, "--wait-time-seconds") == "20"


def test_delete_and_extend_pass_the_handle(monkeypatch) -> None:
    fake = FakeAws(["", ""])
    monkeypatch.setattr(work_queue, "run_aws", fake)
    delete("https://q", "handle-1")
    extend("https://q", "handle-1", 900)
    assert fake.calls[0][2] == "delete-message"
    assert fake.flag(0, "--receipt-handle") == "handle-1"
    assert fake.calls[1][2] == "change-message-visibility"
    assert fake.flag(1, "--visibility-timeout") == "900"


def test_depth_counts_waiting_and_leased(monkeypatch) -> None:
    payload = json.dumps(
        {
            "Attributes": {
                "ApproximateNumberOfMessages": "1200",
                "ApproximateNumberOfMessagesNotVisible": "6",
            }
        }
    )
    monkeypatch.setattr(work_queue, "run_aws", FakeAws([payload]))
    got = depth("https://q")
    assert got == Depth(visible=1200, in_flight=6)
    assert got.total == 1206


def test_depth_of_a_queue_with_no_attributes_is_zero(monkeypatch) -> None:
    monkeypatch.setattr(work_queue, "run_aws", FakeAws([""]))
    assert depth("https://q") == Depth(0, 0)


def test_create_queue_attaches_a_dead_letter_queue(monkeypatch) -> None:
    """A poison item must land somewhere, not cycle: one bad TIFF stalled four shards."""
    fake = FakeAws(
        [
            "https://q/craft-dead",
            "arn:aws:sqs:us-west-2:1:craft-dead",
            "https://q/craft",
        ]
    )
    monkeypatch.setattr(work_queue, "run_aws", fake)
    url, dead = create_queue("craft", visibility=900, max_receives=2)
    assert (url, dead) == ("https://q/craft", "https://q/craft-dead")
    assert fake.flag(0, "--queue-name") == "craft-dead"
    attributes = json.loads(fake.flag(2, "--attributes"))
    assert attributes["VisibilityTimeout"] == "900"
    redrive = json.loads(attributes["RedrivePolicy"])
    assert redrive["maxReceiveCount"] == 2
    assert redrive["deadLetterTargetArn"].endswith("craft-dead")


@pytest.mark.parametrize("seconds", [60, 1800])
def test_visibility_is_a_lease_not_an_interval(monkeypatch, seconds: int) -> None:
    """Documents the contract: the timeout only governs redelivery after a crash."""
    fake = FakeAws(["https://d", "arn:d", "https://q"])
    monkeypatch.setattr(work_queue, "run_aws", fake)
    create_queue("craft", visibility=seconds)
    attributes = json.loads(fake.flag(2, "--attributes"))
    assert attributes["VisibilityTimeout"] == str(seconds)
