"""The georef sidecar contract: one file per channel per page, verdict inside.

A fit pipeline stage records its verdict on a pose *in* the sidecar it wrote,
not by renaming the file aside. The rename convention it replaces
(``p12.georef.json`` -> ``p12.georef-misscale.json``) had three costs:

- it made the file NAME the carrier of a judgement, so every reader had to
  know the full variant vocabulary, and one that didn't silently skipped
  pages (``GEOREF_VARIANTS`` omitted ``-keymap-outlier``, which is exactly the
  pose the p55 class needed weighed);
- it made a demotion indistinguishable from an absence, so a stage could
  refuse to publish but never explain itself to a later stage; and
- it grew a species per judgement -- a volume carried up to nine kinds of
  ``p*.georef*.json`` -- with publication decided by glob precedence over
  them, which is the "ordering is load-bearing" problem #270 exists to remove.

Under this contract a page has at most one sidecar per channel:
``p<stem>.georef.json`` (RANSAC), ``p<stem>.georef-snap.json`` (OSM snap),
``p<stem>.georef-street.json`` (street solver), and ``p<stem>.georef-final.json``
(the arbiter's answer, which is what gets published). ``status`` says what the
writing stage concluded; ``VALID`` means "this channel stands behind this
pose". Anything else is a pose the channel produced and declined -- still on
disk, still weighable, no longer publishable by that channel.
"""

import json
from pathlib import Path

# The channel stands behind this pose.
VALID = "fitted"

# Verdicts a channel can record against its own pose. Each was once a filename.
MISSCALE = "misscale"  # scale disagrees with the volume family / printed note
OUTLIER = "outlier"  # placed far from every other page in the volume
KEYMAP_OUTLIER = "keymap-outlier"  # placed far from the key map's expectation
ONE_GCP = "1gcp"  # single-GCP pose the confirmation pass could not confirm
NOFIT = "nofit"  # no pose at all (the sidecar records only the neighborhood)
CONTRADICTED = "contradicted"  # the page's printed adjacency claims say otherwise


def status(doc: dict) -> str:
    """The verdict recorded in a sidecar doc; VALID when it records none.

    Absent means accepted: a stage that has nothing to complain about writes no
    status, so pre-existing sidecars and the common case both read as VALID.
    """
    return doc.get("status") or VALID


def internally_valid(doc: dict) -> bool:
    """Whether this doc is a pose its own channel stands behind.

    INTERNALLY valid: valid by the writing channel's own lights, which is a
    much weaker claim than correct. Nobody external accepts anything here --
    the stage that wrote the sidecar recorded that it has no objection to what
    it produced, and the arbiter treats that as one input among many.

    Requires both a pose (corners) and an unblemished verdict, so a
    neighborhood-only sidecar is never mistaken for a fit.
    """
    return bool(doc.get("corners")) and status(doc) == VALID


def rejected_poses(doc: dict) -> list[dict]:
    """Poses this channel produced for the page and then set aside.

    A channel can reach more than one pose for a page — georef's key-map
    retry fits a second time with a different vocabulary and keeps whichever
    it prefers. Both belong to the same channel, so both live in the same
    sidecar: the kept one at the top level, the others here, each a complete
    pose doc carrying its own ``status``. They are still real hypotheses (the
    p55 class is exactly a rejected pose that was right), so the arbiter reads
    them; they are simply not what the channel would publish.
    """
    return doc.get("rejected") or []


def attach_rejected(path: str | Path, poses: list[dict]) -> None:
    """Append already-demoted pose docs to the sidecar's rejected list."""
    path = Path(path)
    doc = json.loads(path.read_text())
    doc["rejected"] = [*rejected_poses(doc), *poses]
    path.write_text(json.dumps(doc, indent=2))


def demote(path: str | Path, verdict: str, detail: dict | None = None) -> None:
    """Record ``verdict`` against an existing sidecar, in place.

    Replaces ``os.rename(georef.json, georef-<verdict>.json)``. The pose stays
    exactly where a reader expects to find it; what changes is that the channel
    no longer claims it. ``detail`` is merged under ``status_detail`` so the
    reason survives for the arbiter's report and for a human reading the file.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    doc["status"] = verdict
    if detail:
        doc["status_detail"] = {**(doc.get("status_detail") or {}), **detail}
    path.write_text(json.dumps(doc, indent=2))
