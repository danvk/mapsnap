"""Learned accept probabilities for snap's rescue and challenge decisions (#650).

Snap ranks a page's candidates by their select score and accepted the top one
when that score cleared a bar (rescue 1.35 with a 0.25 margin, panel-solo 2.2,
challenge 1.6). Graded against OIM truth on 690 corpus-v2 items, the bars sit
near their best for that score, but the score itself separates good poses from
disasters poorly: it leaves out how far the search center was, how much
refinement moved the scale, and how much ground the page covers, and it gives
region containment a weight the evidence does not support.

This module scores the same top candidate with two logistic models trained on
those graded candidates (scripts/train_snap_accept.py):

  - ``rescue``: P(the top candidate is within 25 ft of truth), for a page with
    no incumbent;
  - ``replace``: P(publishing the top candidate instead of the incumbent fit
    improves the page), for a fitted page whose top candidate disagrees with it.

Features are computed from a candidates.jsonl record alone, so training and
production share this one definition. Ranking is unchanged: the model only
decides whether the select score's choice is published.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapsnap.page_units import grid_rmse_ft_between

ACCEPT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "snap_accept.json"
)

# Candidates within this of the top one (grid RMSE) are the same lock found
# again, and count as agreement.
AGREE_FT = 50.0
# Caps keep a few extreme values from dominating a linear model.
CHAMFER_CAP_M = 30.0
MARGIN_CAP = 3.0
AGREE_CAP = 5


def log1p_nonnegative(value: float | None) -> float:
    """log(1 + value) for a value clipped at zero; 0 when missing."""
    return math.log1p(max(0.0, value)) if value is not None else 0.0


def capped_margin(margin: float | None) -> float:
    """A distinct-rival margin, infinite (no rival) capped at MARGIN_CAP; 0 when missing."""
    if margin is None:
        return 0.0
    return min(margin, MARGIN_CAP)


def agreeing_candidates(record: dict) -> int:
    """How many other scored candidates land within AGREE_FT of the top one."""
    candidates = record.get("candidates") or []
    top = np.array(candidates[0]["world_affine"])
    count = 0
    for candidate in candidates[1:]:
        if candidate.get("select_score") is None or not candidate.get("world_affine"):
            continue
        distance = grid_rmse_ft_between(
            top, np.array(candidate["world_affine"]), record["width"], record["height"]
        )
        count += distance <= AGREE_FT
    return min(count, AGREE_CAP)


def candidate_terms(record: dict, margin: float | None) -> dict[str, float]:
    """The top candidate's evidence, transformed to be roughly linear in log-odds."""
    top = record["candidates"][0]
    prior_sigma = top.get("prior_theta_residual_sigma")
    scale = top.get("scale_m_per_px") or 0.0
    footprint_km2 = record["width"] * record["height"] * scale * scale / 1e6
    scale_adjust = top.get("scale_adjust")
    return {
        "inlier_frac": top["inlier_frac"],
        "ncc_fine": top["ncc_fine"],
        "chamfer_m": min(top["chamfer_mean_m"], CHAMFER_CAP_M),
        "name_evidence": (top.get("name") or {}).get("evidence") or 0.0,
        "region_containment": top.get("region_containment") or 0.0,
        "prior_term": max(-1.0, 1.0 - prior_sigma) if prior_sigma is not None else 0.0,
        "log_center_dist": log1p_nonnegative(top.get("center_dist_m")),
        "log_distinct_margin": log1p_nonnegative(capped_margin(margin)),
        "log_footprint": math.log(max(footprint_km2, 1e-4)),
        "abs_log_scale_adjust": abs(math.log(scale_adjust)) if scale_adjust else 0.0,
        "log_points": log1p_nonnegative(top.get("n_points")),
        "agree": float(agreeing_candidates(record)),
        "panel": float("__" in record["target"]),
    }


def rescue_terms(record: dict) -> dict[str, float]:
    """Model inputs for rescuing a page with no incumbent."""
    # Imported here: snap_volume imports this module.
    from mapsnap.snap_volume import distinct_margin

    return candidate_terms(record, distinct_margin(record, pose_aware=True))


def replace_terms(record: dict) -> dict[str, float]:
    """Model inputs for replacing a fitted page's incumbent with its top candidate."""
    from mapsnap.snap_volume import distinct_margin, incumbent_disagreement_ft

    terms = candidate_terms(record, distinct_margin(record, pose_aware=False))
    incumbent = record["incumbent"]
    top = record["candidates"][0]
    incumbent_verification = incumbent.get("verification") or 0.0
    incumbent_name = (incumbent.get("name") or {}).get("score") or 0.0
    terms.update(
        incumbent_verification=incumbent_verification,
        incumbent_ncc_fine=incumbent.get("ncc_fine") or 0.0,
        incumbent_inlier_frac=incumbent.get("inlier_frac") or 0.0,
        incumbent_chamfer_m=min(incumbent.get("chamfer_mean_m") or 0.0, CHAMFER_CAP_M),
        log_incumbent_gcps=log1p_nonnegative(incumbent.get("effective_gcps")),
        log_disagreement=log1p_nonnegative(incumbent_disagreement_ft(record)),
        verification_gain=(top.get("verification") or 0.0) - incumbent_verification,
        name_gain=((top.get("name") or {}).get("score") or 0.0) - incumbent_name,
    )
    return terms


def scorable(record: dict) -> bool:
    """Whether a record's top candidate can be scored: present, plausible and posed."""
    candidates = record.get("candidates") or []
    return bool(
        candidates
        and candidates[0].get("select_score") is not None
        and candidates[0].get("world_affine")
        and record.get("width")
        and record.get("height")
    )


@dataclass
class LogisticModel:
    """A standardized logistic model: P = sigmoid(intercept + sum coef * (x - mean) / scale)."""

    intercept: float
    columns: dict[str, dict[str, float]]
    threshold: float

    def probability(self, terms: dict[str, float]) -> float:
        """The model's probability for one set of terms."""
        z = self.intercept
        for name, column in self.columns.items():
            z += column["coef"] * (terms[name] - column["mean"]) / column["scale"]
        return 1.0 / (1.0 + math.exp(-z))

    def to_json(self) -> dict:
        """The model as stored in models/snap_accept.json."""
        return {
            "intercept": self.intercept,
            "threshold": self.threshold,
            "columns": self.columns,
        }

    @classmethod
    def from_json(cls, doc: dict) -> "LogisticModel":
        """A model from its stored form."""
        return cls(float(doc["intercept"]), doc["columns"], float(doc["threshold"]))


@dataclass
class AcceptModel:
    """The rescue and replace models, with the version they were trained as."""

    version: str
    rescue: LogisticModel
    replace: LogisticModel

    def rescue_probability(self, record: dict) -> float | None:
        """P(good) for a rescue's top candidate, or None when it cannot be scored."""
        if not scorable(record):
            return None
        return self.rescue.probability(rescue_terms(record))

    def replace_probability(self, record: dict) -> float | None:
        """P(better than the incumbent) for a fitted page's top candidate, or None."""
        if not scorable(record) or not (record.get("incumbent") or {}).get(
            "world_affine"
        ):
            return None
        return self.replace.probability(replace_terms(record))


def load_accept_model(path: Path = ACCEPT_MODEL_PATH) -> AcceptModel:
    """Read models/snap_accept.json (or another trained file)."""
    doc = json.loads(path.read_text())
    return AcceptModel(
        version=doc["version"],
        rescue=LogisticModel.from_json(doc["rescue"]),
        replace=LogisticModel.from_json(doc["replace"]),
    )
