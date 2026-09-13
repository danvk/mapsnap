"""Time the accelerator-eligible pipeline stages on this machine (#354 sizing).

``mapsnap bench --volume DIR --out results.json`` runs craft, ocr, the road and
region UNets and the key-map page-number reader exactly as the pipeline invokes
them, first on the best accelerator present (CUDA, else MPS) and again on the CPU,
and records seconds per page for each. The volume's images are symlinked into a
scratch directory beside the output file so no sidecar lands in the real volume.
``mapsnap bench --report a.json b.json`` tabulates several machines side by side.

CLI stages are timed twice, on one page and on N pages, so the per-page figure
excludes process startup and model loading (reported separately as ``startup``).
The multi-worker ocr rows are single runs and report wall time per page including
startup: that is the throughput an instance actually delivers.

See scripts/aws_bench/README.md for launching this on EC2.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

CRAFT_SIDECARS = ("boxes.json",)
OCR_SIDECARS = ("streets.json", "txt")
KEYMAP_SIDECARS = ("keymap.json", "keymap.txt", "keymap-raw.json")
IMDS = "http://169.254.169.254/latest"


@dataclass
class StageResult:
    """One timed stage: seconds per page on one device with one worker count."""

    stage: str
    device: str
    pages: int
    workers: int
    seconds: float
    per_page: float
    startup: float | None = None
    note: str = ""


def per_page_seconds(seconds_one: float, seconds_all: float, pages: int) -> float:
    """Marginal seconds per page from a one-page run and a ``pages``-page run."""
    if pages <= 1:
        return seconds_all
    return (seconds_all - seconds_one) / (pages - 1)


def startup_seconds(seconds_one: float, per_page: float) -> float:
    """Process startup plus model loading implied by a one-page run."""
    return max(0.0, seconds_one - per_page)


def parent_pages(volume: Path) -> list[Path]:
    """The volume's parent page images (no split panels), in page order."""
    from mapsnap.keymap.records import page_key_sort

    pages = [path for path in volume.glob("p*.jpg") if "__" not in path.stem]
    return sorted(pages, key=lambda path: page_key_sort(path.stem[1:]))


def raw_sheet(volume: Path) -> Path | None:
    """The first full-resolution key-map sheet under ``raw/``, if any."""
    sheets = sorted((volume / "raw").glob("p*.jpg"))
    return sheets[0] if sheets else None


def accelerator() -> str | None:
    """``cuda`` or ``mps`` when torch can use one here, else None."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def imds(path: str) -> str | None:
    """One EC2 instance-metadata value (IMDSv2), or None off EC2."""
    try:
        token_request = urllib.request.Request(
            f"{IMDS}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_request, timeout=1) as response:
            token = response.read().decode()
        request = urllib.request.Request(
            f"{IMDS}/meta-data/{path}", headers={"X-aws-ec2-metadata-token": token}
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            return response.read().decode()
    except OSError:
        return None


def cpu_model() -> str:
    """Marketing name of the CPU, from /proc/cpuinfo or sysctl."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return platform.processor()


def machine_info() -> dict[str, object]:
    """What this benchmark ran on: CPU, accelerator, instance type, versions."""
    import torch

    info: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "accelerator": accelerator(),
        "instance_type": imds("instance-type"),
        "instance_id": imds("instance-id"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        properties = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(properties.total_memory / 2**30, 1)
    return info


def machine_label(info: dict[str, object]) -> str:
    """Short column heading for a machine: instance type or host, plus accelerator."""
    name = info.get("instance_type") or info.get("hostname") or "?"
    device = info.get("gpu") or info.get("accelerator") or "cpu"
    return f"{name} ({device})"


def timed_cli(cmd: list[str], log: Path) -> float:
    """Run a pipeline command, appending its output to ``log``; return elapsed seconds."""
    with log.open("a") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        handle.flush()
        started = time.perf_counter()
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
        return time.perf_counter() - started


def clear_sidecars(images: list[Path], suffixes: tuple[str, ...]) -> None:
    """Delete ``<stem>.<suffix>`` beside each image so the next run recomputes it."""
    for image in images:
        for suffix in suffixes:
            image.with_name(f"{image.stem}.{suffix}").unlink(missing_ok=True)


def stage_scratch(volume: Path, work: Path) -> Path:
    """Symlink the volume's inputs into ``work`` so sidecars never touch ``volume``."""
    scratch = work / volume.name
    (scratch / "raw").mkdir(parents=True, exist_ok=True)
    inputs = [
        *parent_pages(volume),
        volume / "centerlines.geojson",
        volume / "mapsnap.json",
    ]
    sheet = raw_sheet(volume)
    if sheet is not None:
        inputs.append(sheet)
    for source in inputs:
        if not source.exists():
            continue
        target = scratch / source.relative_to(volume)
        if not target.is_symlink():
            target.symlink_to(source.resolve())
    return scratch


class Bench:
    """Runs the stage matrix on one scratch volume and collects StageResults."""

    def __init__(self, volume: Path, log: Path, workers: int) -> None:
        self.volume = volume
        self.log = log
        self.workers = workers
        self.centerlines = volume / "centerlines.geojson"
        self.results: list[StageResult] = []

    def record(self, result: StageResult) -> None:
        self.results.append(result)
        startup = (
            f" startup {result.startup:.1f}s" if result.startup is not None else ""
        )
        print(
            f"  {result.stage:22s} {result.device:5s} x{result.workers:<2d} "
            f"{result.pages:3d} pages {result.seconds:7.1f}s = {result.per_page:6.2f} s/page"
            f"{startup} {result.note}",
            flush=True,
        )

    def cli_stage(
        self,
        stage: str,
        device: str,
        pages: list[Path],
        command: Callable[[list[Path]], list[str]],
        sidecars: tuple[str, ...],
    ) -> None:
        """Time ``command`` on one page then on all ``pages``; report the marginal cost."""
        clear_sidecars(pages, sidecars)
        seconds_one = timed_cli(command(pages[:1]), self.log)
        clear_sidecars(pages, sidecars)
        seconds_all = timed_cli(command(pages), self.log)
        per_page = per_page_seconds(seconds_one, seconds_all, len(pages))
        self.record(
            StageResult(
                stage,
                device,
                len(pages),
                1,
                seconds_all,
                per_page,
                startup=startup_seconds(seconds_one, per_page),
            )
        )

    def single_run(
        self,
        stage: str,
        device: str,
        pages: list[Path],
        cmd: list[str],
        sidecars: tuple[str, ...],
        workers: int = 1,
        note: str = "",
    ) -> None:
        """Time one run of ``cmd``; per-page includes startup (a throughput figure)."""
        clear_sidecars(pages, sidecars)
        seconds = timed_cli(cmd, self.log)
        self.record(
            StageResult(
                stage,
                device,
                len(pages),
                workers,
                seconds,
                seconds / len(pages),
                note=note,
            )
        )

    def craft_cmd(self, device: str) -> Callable[[list[Path]], list[str]]:
        flag = ["--no-gpu"] if device == "cpu" else []
        return lambda pages: ["mapsnap", "craft", *map(str, pages), *flag]

    def ocr_cmd(
        self, device: str, workers: int = 1, extra: tuple[str, ...] = ()
    ) -> Callable[[list[Path]], list[str]]:
        flag = ["--no-gpu"] if device == "cpu" else []
        return lambda pages: [
            "mapsnap",
            "ocr",
            *map(str, pages),
            "--centerlines",
            str(self.centerlines),
            "--ignore-keymap",
            "--num-workers",
            str(workers),
            *flag,
            *extra,
        ]

    def unet_stage(
        self,
        stage: str,
        device: str,
        pages: list[Path],
        loader: Callable[[str], object],
        predict: Callable[[object, Path, str], None],
    ) -> None:
        """Time an in-process UNet: model load once, then one prediction per page."""
        started = time.perf_counter()
        model = loader(device)
        predict(model, pages[0], device)  # warm-up (kernel compilation, allocator)
        startup = time.perf_counter() - started
        started = time.perf_counter()
        for page in pages:
            predict(model, page, device)
        seconds = time.perf_counter() - started
        self.record(
            StageResult(
                stage,
                device,
                len(pages),
                1,
                seconds,
                seconds / len(pages),
                startup=startup,
            )
        )


def load_road(device: str) -> object:
    import torch

    from mapsnap.road_model import ROAD_MODEL_PATH, load_model

    return load_model(ROAD_MODEL_PATH, torch.device(device))


def predict_road(model: object, page: Path, device: str) -> None:
    import cv2
    import torch

    from mapsnap.road_model import predict_page

    gray = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(page)
    predict_page(model, gray, torch.device(device))  # type: ignore[arg-type]


def load_region(device: str) -> object:
    import torch

    from mapsnap.region_model import load_region_model

    model, _ = load_region_model(device=torch.device(device))
    return model


def predict_region_page(model: object, page: Path, device: str) -> None:
    import cv2
    import torch

    from mapsnap.region_model import predict_region

    image = cv2.imread(str(page))
    if image is None:
        raise FileNotFoundError(page)
    predict_region(model, image, torch.device(device))


def run_benchmark(
    volume: Path,
    log: Path,
    *,
    page_limit: int | None,
    cpu_pages: int,
    workers: int,
    slow: bool,
) -> list[StageResult]:
    """The full stage matrix on ``volume`` (a scratch copy); returns every StageResult."""
    from mapsnap.keymap.pipeline import valid_page_spec

    bench = Bench(volume, log, workers)
    pages = parent_pages(volume)[:page_limit]
    cpu_subset = pages[: max(2, cpu_pages)]
    sheet = raw_sheet(volume)
    accel = accelerator()
    devices = [accel, "cpu"] if accel else ["cpu"]
    print(
        f"{len(pages)} pages, accelerator: {accel or 'none'}, {workers} workers",
        flush=True,
    )

    # CRAFT: CPU subset first so the accelerator's boxes are the ones ocr reads.
    for device in reversed(devices):
        subset = pages if device != "cpu" else cpu_subset
        bench.cli_stage(
            "craft", device, subset, bench.craft_cmd(device), CRAFT_SIDECARS
        )
    if sheet is not None:
        for device in devices:
            if device == "cpu" and not slow:
                continue
            bench.single_run(
                "craft raw sheet",
                device,
                [sheet],
                bench.craft_cmd(device)([sheet]),
                CRAFT_SIDECARS,
                note="tiled at native resolution",
            )

    # ocr recognition inside the cached boxes.
    for device in devices:
        subset = pages if device != "cpu" else cpu_subset
        bench.cli_stage("ocr", device, subset, bench.ocr_cmd(device), OCR_SIDECARS)
        if workers > 1:
            bench.single_run(
                "ocr",
                device,
                subset,
                bench.ocr_cmd(device, workers)(subset),
                OCR_SIDECARS,
                workers=workers,
                note="wall time incl. startup",
            )
    if sheet is not None and accel:
        bench.single_run(
            "ocr raw sheet",
            accel,
            [sheet],
            bench.ocr_cmd(accel, extra=("--min-short-side", "60"))([sheet]),
            OCR_SIDECARS,
        )

    # UNets, in process.
    for device in devices:
        subset = pages if device != "cpu" else cpu_subset
        bench.unet_stage("road unet", device, subset, load_road, predict_road)
        bench.unet_stage(
            "region unet", device, subset, load_region, predict_region_page
        )

    # Key-map page numbers (CNN localizer + CRNN), accelerator only: it is light.
    if sheet is not None and accel:
        spec = valid_page_spec([sheet]) or "1-999"
        bench.single_run(
            "keymap numbers",
            accel,
            [sheet],
            [
                sys.executable,
                "-m",
                "mapsnap.keymap.detect_numbers_crnn",
                str(sheet),
                "--pages",
                spec,
            ],
            KEYMAP_SIDECARS,
        )
    return bench.results


def format_report(records: list[dict[str, object]]) -> str:
    """Seconds per page per stage/device/workers, one column per machine."""
    labels = [machine_label(record["machine"]) for record in records]  # type: ignore[arg-type]
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for label, record in zip(labels, records, strict=True):
        for stage in record["stages"]:  # type: ignore[union-attr]
            key = (stage["stage"], stage["device"], stage["workers"])
            rows.setdefault(key, {})[label] = stage["per_page"]
    width = max(len(label) for label in labels)
    lines = [
        f"{'stage':22s} {'device':6s} {'wkr':>3s}  "
        + "  ".join(f"{l:>{width}s}" for l in labels)
    ]
    for (stage, device, workers), cells in sorted(rows.items()):
        values = "  ".join(
            f"{cells[label]:{width}.2f}" if label in cells else f"{'-':>{width}s}"
            for label in labels
        )
        lines.append(f"{stage:22s} {device:6s} {workers:3d}  {values}")
    lines.append("(seconds per page; multi-worker rows include process startup)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--volume", type=Path, help="Volume with p*.jpg, centerlines, raw/ sheet."
    )
    parser.add_argument("--out", type=Path, help="Where to write the JSON record.")
    parser.add_argument(
        "--pages", type=int, help="Accelerator page count (default: all)."
    )
    parser.add_argument(
        "--cpu-pages", type=int, default=8, help="CPU page count (default 8)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="ocr workers for the shared-GPU row; 1 (default) skips it. The EC2 "
        "bootstrap passes the vCPU count. Keep 1 on a laptop: each worker loads its own "
        "recognizer and the machine runs out of memory.",
    )
    parser.add_argument(
        "--slow", action="store_true", help="Also CRAFT the raw sheet on the CPU."
    )
    parser.add_argument(
        "--report", nargs="*", type=Path, help="Tabulate these result JSONs."
    )
    args = parser.parse_args()

    if args.report:
        print(format_report([json.loads(path.read_text()) for path in args.report]))
        return
    if args.volume is None or args.out is None:
        parser.error("--volume and --out are required unless --report is given")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = args.out.with_suffix(".work")
    scratch = stage_scratch(args.volume, work)
    log = args.out.with_suffix(".log")
    started = datetime.now(UTC)
    print(f"machine: {json.dumps(machine_info())}", flush=True)
    results = run_benchmark(
        scratch,
        log,
        page_limit=args.pages,
        cpu_pages=args.cpu_pages,
        workers=args.workers,
        slow=args.slow,
    )
    record = {
        "machine": machine_info(),
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "args": {
            key: str(value) for key, value in vars(args).items() if value is not None
        },
        "stages": [asdict(result) for result in results],
    }
    args.out.write_text(json.dumps(record, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
