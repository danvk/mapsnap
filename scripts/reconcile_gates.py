#!/usr/bin/env python3
"""Pre-registered G1-G4 gate evaluation for `mapsnap reconcile` (#270 v1).

Runs reconcile over the four gate volumes' `reconcile-base` state, grades the
baseline and reconcile IIIFs through the SAME real compare, and emits
`artifacts/reconcile/gates.md` at the repo root's data dir. The gate table and
its predicates were pre-registered in the implementation plan before any
volume was run; expected-misses are marked, not silently dropped.

  uv run python scripts/reconcile_gates.py            # all four volumes
  uv run python scripts/reconcile_gates.py fargo_nd_1958
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapsnap.compare_iiif_georef import compare_pages
from mapsnap.score import summarize, volume_page_scores

DATA = Path(__file__).resolve().parent.parent / "data"
BASE_TAG = "reconcile-base"

VOLUMES = [
    "fargo_nd_1958",
    "washington_dc_1916_vol_2",
    "philadelphia_pa_1950_vol_3",
    "los_angeles_ca_1949_vol_14",
]

# G2 recovery targets: (volume, page, predicate, note). Predicates are judged
# on region-graded compare rows. Pre-registered before any gate run.
RECOVERY = [
    ("washington_dc_1916_vol_2", "p174", "le50", ""),
    ("washington_dc_1916_vol_2", "p175", "le50", ""),
    (
        "washington_dc_1916_vol_2",
        "p176",
        "le50",
        "pre-registered expected-miss: no pose on disk",
    ),
    (
        "washington_dc_1916_vol_2",
        "p216",
        "le50",
        "pre-registered hardest: wrong candidate outranks on verification",
    ),
    ("washington_dc_1916_vol_2", "p217", "le50", ""),
    ("philadelphia_pa_1950_vol_3", "p213", "le25", ""),
    ("fargo_nd_1958", "p63__4", "gone_or_le50", ""),
    ("fargo_nd_1958", "p58__3", "gone_or_le50", ""),
]

GOOD_FT, MID_FT, DISASTER_FT = 25.0, 50.0, 200.0
# G1 asks whether the change DESTROYS pages, so its predicate is the severe
# one (good -> >50 ft or unplaced). That is not the same question as "what did
# this cost": a page sliding 24.7 -> 25.1 ft is invisible to G1 and cost
# grand_rapids 1.89 points, more than all three G1 offenders combined. Both
# are reported; neither is presented as the other.


def band(rmse: float | None) -> str:
    if rmse is None:
        return "unplaced"
    if rmse <= GOOD_FT:
        return "good"
    if rmse >= DISASTER_FT:
        return "disaster"
    return "mid"


def compare_rows(truth: Path, iiif: Path, volume: Path) -> dict[str, float]:
    """{page key: region-graded rmse_ft} via the real compare.

    The IIIF is staged into the volume root first: compare resolves each
    split page's panels.json from the generated file's own directory, so
    grading an IIIF at its artifact path silently unmatches every split page
    (both gate runs 1-2 had this blind spot in G1/G2/G3).
    """
    staged = volume / f"reconcile-gate-{iiif.stem}-cmp.iiif.json"
    shutil.copy2(iiif, staged)
    try:
        rows, _missing = compare_pages(truth, staged, oim_dir=volume / "oim")
        return {row["page_key"]: row["rmse_ft"] for row in rows}
    finally:
        staged.unlink()


def net_score(volume: Path, iiif: Path) -> float:
    """Land-weighted net score (good share minus disaster share).

    volume_page_scores discovers centerlines and the oim/ panel dir from the
    generated file's parent, so the IIIF is scored from a temp copy in the
    volume root (removed afterwards).
    """
    staged = volume / f"reconcile-gate-{iiif.stem}.iiif.json"
    shutil.copy2(iiif, staged)
    try:
        scores = volume_page_scores(staged, truth=volume / "main.iiif.json")
        return summarize(scores).net_score * 100.0
    finally:
        staged.unlink()


def run_volume(name: str) -> dict:
    volume = DATA / name
    base_dir = volume / "artifacts" / BASE_TAG
    assert base_dir.is_dir(), f"missing {base_dir} — run the Step-0 fits first"
    if "--regrade-only" not in sys.argv:
        subprocess.run(
            [
                "mapsnap",
                "reconcile",
                str(volume),
                "--sidecars-from",
                str(base_dir),
                "--grade",
            ],
            check=True,
        )
    truth = volume / "main.iiif.json"
    base_iiif = base_dir / f"{BASE_TAG}.iiif.json"
    recon_iiif = volume / "artifacts" / "reconcile" / "reconcile.iiif.json"
    base = compare_rows(truth, base_iiif, volume)
    recon = compare_rows(truth, recon_iiif, volume)
    g1_offenders = []
    any_worse = []
    band_moves = 0
    new_disasters = []
    for key in set(base) | set(recon):
        b, r = band(base.get(key)), band(recon.get(key))
        if b != r:
            band_moves += 1
        base_good = base.get(key) is not None and base[key] <= GOOD_FT
        if base_good and (recon.get(key) is None or recon[key] > GOOD_FT):
            any_worse.append((key, base[key], recon.get(key)))
            if recon.get(key) is None or recon[key] > MID_FT:
                g1_offenders.append((key, base[key], recon.get(key)))
        if r == "disaster" and b != "disaster":
            new_disasters.append((key, base.get(key), recon[key]))
    return {
        "volume": name,
        "base_rows": base,
        "recon_rows": recon,
        "g1_offenders": sorted(g1_offenders),
        "any_worse": sorted(any_worse),
        "band_moves": band_moves,
        "new_disasters": sorted(new_disasters),
        "net_base": net_score(volume, base_iiif),
        "net_recon": net_score(volume, recon_iiif),
    }


def main() -> None:
    volumes = [a for a in sys.argv[1:] if not a.startswith("--")] or VOLUMES
    results = {name: run_volume(name) for name in volumes}

    lines = ["# reconcile v1 gate report", ""]
    # G1
    g1_pass = all(not r["g1_offenders"] for r in results.values())
    lines.append(f"## G1 safety: {'PASS' if g1_pass else 'FAIL'}")
    for r in results.values():
        for key, was, now in r["g1_offenders"]:
            lines.append(
                f"- {r['volume']} {key}: {was:.1f} -> "
                f"{now if now is None else f'{now:.1f}'} ft"
            )
    lines.append("")
    lines.append("### every page that got worse (good -> anything worse), for scale")
    total_worse = sum(len(r["any_worse"]) for r in results.values())
    lines.append(f"{total_worse} pages, of which {sum(len(r['g1_offenders']) for r in results.values())} are G1 offenders:")
    for r in results.values():
        for key, was, now in r["any_worse"]:
            severe = " (G1)" if (now is None or now > MID_FT) else ""
            lines.append(f"- {r['volume']} {key}: {was:.1f} -> "
                         f"{'unplaced' if now is None else f'{now:.1f}'} ft{severe}")
    lines.append("")
    # G2
    recovered = 0
    applicable = 0
    lines.append("## G2 recovery")
    for volume_name, page, predicate, note in RECOVERY:
        if volume_name not in results:
            continue
        applicable += 1
        r = results[volume_name]
        base_v, recon_v = r["base_rows"].get(page), r["recon_rows"].get(page)
        if predicate == "le50":
            ok = recon_v is not None and recon_v <= MID_FT
        elif predicate == "le25":
            ok = recon_v is not None and recon_v <= GOOD_FT
        else:  # gone_or_le50: the junk pose is unpublished or the region is now good
            ok = recon_v is None or recon_v <= MID_FT
        recovered += ok
        lines.append(
            f"- {'PASS' if ok else 'miss'} {volume_name} {page}: "
            f"{base_v if base_v is None else f'{base_v:.1f}'} -> "
            f"{recon_v if recon_v is None else f'{recon_v:.1f}'} ft"
            + (f"  ({note})" if note else "")
        )
    lines.append(f"\n**{recovered}/{applicable} recovered (gate: >=5 of 8)**\n")
    # G3
    la = results.get("los_angeles_ca_1949_vol_14")
    if la:
        g3 = la["band_moves"] <= 1 and not la["new_disasters"]
        lines.append(
            f"## G3 LA null: {'PASS' if g3 else 'FAIL'} "
            f"({la['band_moves']} band moves, "
            f"{len(la['new_disasters'])} new disasters)"
        )
        for key, was, now in la["new_disasters"]:
            lines.append(f"- new disaster {key}: {was} -> {now:.1f} ft")
        lines.append("")
    # G4
    lines.append("## G4 land-weighted score (real compare, both arms)")
    lines.append("| volume | baseline | reconcile | delta | gate |")
    lines.append("|---|---|---|---|---|")
    gates = {
        "washington_dc_1916_vol_2": "+1.5",
        "fargo_nd_1958": ">=0",
        "philadelphia_pa_1950_vol_3": ">=0",
        "los_angeles_ca_1949_vol_14": "within +-0.3",
    }
    for r in results.values():
        delta = r["net_recon"] - r["net_base"]
        lines.append(
            f"| {r['volume']} | {r['net_base']:.1f} | {r['net_recon']:.1f} | "
            f"{delta:+.1f} | {gates.get(r['volume'], '-')} |"
        )
    out = DATA.parent / "data" / "reconcile-gates.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
