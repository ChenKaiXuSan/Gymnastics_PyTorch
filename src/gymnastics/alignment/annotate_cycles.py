"""Offline cycle and middle annotation for every dataset (``gymnastics align cycles``).

Cycle detection is a preprocessing step: it runs once, writes record files,
and training reads only those files.  This command produces the records for
the three data sources with one shared detector
(:mod:`gymnastics.alignment.cycles`, right-wrist azimuth in the body frame).

Sub-commands::

    gymnastics align cycles private [--log-root local/runs/split_cycle] [--kpt-root ...] [--person 1 2]
        Adds ``mid`` (turn-around frame) to the existing
        ``alignment_record_<id>.json`` files of the private recordings.  The
        cycle boundaries written by ``gymnastics align`` are kept unchanged;
        only the middle is computed from the two SAM3D views of each cycle.

    gymnastics align cycles freeman [--benchmark-root local/runs/freeman_benchmark_cluster] [--subjects 12 13]
        Detects cycles + middles on every session of the zero-shot benchmark
        cache and writes ``local/runs/cycle_records/freeman/subject_NN/<session>.json``.

    gymnastics align cycles unity [--benchmark-root ...] [--sam3d-cache-root ...]
        Same for the continuous Unity sequences ->
        ``local/runs/cycle_records/unity/subject_<sequence>/<sequence>.json``.

    gymnastics align cycles index [--records-root local/runs/cycle_records]
        Collects everything into one tree: exports the private alignment
        records as ``cycle_record_v1`` files under ``gymnastics/``, gathers
        the logs, and writes ``index.json`` + ``README.md`` with counts,
        detection settings and the source-data locations.  The private
        ``alignment_record_<id>.json`` stays the authoritative file (other
        pipeline stages read it); the export is a uniform read-only copy
        with a checksum of its source.

Every record stores the detection settings that produced it, and a
``theta_cycles.png`` audit plot (signal, cycle starts, middles) is written
next to it unless ``--no-plot`` is given.  Sequences in which no plausible
cycle is found still get a record with an empty ``cycles`` list, so the
training loader can distinguish "not periodic" from "not processed".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gymnastics.alignment.cycle_records import (
    FORMAT_V1,
    augment_alignment_record,
    cycle_record_path,
    read_cycle_record,
    write_cycle_record,
)
from gymnastics.alignment.cycles import (
    CycleSpan,
    DetectionSettings,
    annotate_mid_points,
    detect_cycles,
    hand_theta_unwrapped,
)
from gymnastics.common.paths import PROJECT_ROOT


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _valid_joints(points: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1) & np.any(points != 0, axis=-1)
    return finite if valid is None else finite & np.asarray(valid, dtype=bool)


def fused_wrist_theta(
    view_a: np.ndarray,
    view_b: np.ndarray,
    *,
    valid_a: Optional[np.ndarray] = None,
    valid_b: Optional[np.ndarray] = None,
    smooth_window: int = 11,
) -> np.ndarray:
    """Right-wrist azimuth of the validity-weighted body-frame fusion of two views.

    This reproduces the signal ``gymnastics align`` segments on: each view is
    mapped into its own pelvis body frame, the two are averaged where both
    are valid, and the wrist azimuth of the fused body is smoothed and
    unwrapped.  Frames without a valid body frame become NaN and are
    interpolated by :func:`hand_theta_unwrapped`.

    Args:
        view_a: ``[T, 70, 3]`` world keypoints of view A.
        view_b: ``[T, 70, 3]`` world keypoints of view B (same frames).
        valid_a: Optional ``[T, 70]`` validity of view A.
        valid_b: Optional ``[T, 70]`` validity of view B.
        smooth_window: Moving-average window in frames.

    Returns:
        ``[T]`` unwrapped angle.
    """
    from gymnastics.alignment.main import IDX, kpts_world_to_body

    if view_a.shape != view_b.shape:
        raise ValueError("both views must have shape [T, 70, 3]")
    bodies: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    for points, valid in ((view_a, valid_a), (view_b, valid_b)):
        points = np.asarray(points, dtype=np.float32)
        usable = _valid_joints(points, valid)
        masked = np.where(usable[..., None], points, np.nan)
        body = kpts_world_to_body(masked, IDX)
        weight = np.isfinite(body).all(axis=-1).astype(np.float32)
        bodies.append(np.nan_to_num(body))
        weights.append(weight)
    total = weights[0] + weights[1]
    fused = (bodies[0] * weights[0][..., None] + bodies[1] * weights[1][..., None]) / np.maximum(total, 1e-8)[..., None]
    fused = np.where((total > 0)[..., None], fused, np.nan)
    return hand_theta_unwrapped(fused, smooth_window=smooth_window)


def save_theta_plot(theta: np.ndarray, fps: float, spans: Sequence[CycleSpan], settings: DetectionSettings, out_path: Path, title: str) -> bool:
    """Audit plot: signal, reference line, cycle starts and middles."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    t = np.arange(len(theta)) / max(float(fps), 1e-6)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    ax.plot(t, theta, linewidth=1.1, label="θ right wrist (unwrapped)")
    if settings.theta_ref is not None:
        ax.axhline(settings.theta_ref, color="r", linestyle="--", linewidth=0.9, alpha=0.7, label=f"θ_ref = {settings.theta_ref:.2f}")
    if spans:
        starts = [s.start for s in spans] + [spans[-1].end]
        mids = [s.mid for s in spans]
        ax.scatter(np.array(starts) / fps, theta[np.clip(starts, 0, len(theta) - 1)], color="green", s=36, zorder=5, label=f"cycle starts ({len(spans)})")
        ax.scatter(np.array(mids) / fps, theta[mids], color="orange", marker="^", s=48, zorder=6, label="middles (turn-around)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("θ (rad)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- private
def annotate_private_person(
    person_id: str,
    *,
    kpt_root: Path,
    log_root: Path,
    smooth_window: int,
    plot: bool,
) -> Dict[str, object]:
    """Add middles to one person's alignment record (boundaries unchanged)."""
    from gymnastics.alignment.load import load_sam3d_body_sequence

    record_path = log_root / f"person_{person_id}" / f"alignment_record_{person_id}.json"
    record = read_cycle_record(record_path)
    assert record.side_cycles is not None
    face_seq = load_sam3d_body_sequence(kpt_root, person_id=person_id, subdir="face")
    side_seq = load_sam3d_body_sequence(kpt_root, person_id=person_id, subdir="side")
    face_k = face_seq.kpts3d if hasattr(face_seq, "kpts3d") else face_seq[1]
    side_k = side_seq.kpts3d if hasattr(side_seq, "kpts3d") else side_seq[1]
    face_spans: List[CycleSpan] = []
    side_spans: List[CycleSpan] = []
    ratios: List[float] = []
    thetas: List[np.ndarray] = []
    for (fs, _, fe), (ss, _, se) in zip(record.cycles, record.side_cycles):
        length = min(fe - fs, se - ss, len(face_k) - fs, len(side_k) - ss)
        if length < 3:
            raise ValueError(f"person {person_id}: cycle [{fs}, {fe}) has fewer than three usable frames")
        theta = fused_wrist_theta(face_k[fs : fs + length], side_k[ss : ss + length], smooth_window=smooth_window)
        span = annotate_mid_points(theta, [(0, length)])[0]
        face_spans.append(CycleSpan(fs, fs + span.mid, fe))
        side_spans.append(CycleSpan(ss, ss + span.mid, se))
        ratios.append(span.mid / length)
        thetas.append(theta)
    previous_ref = record.detection.get("theta_ref") if record.detection else None
    settings = DetectionSettings(
        smooth_window=smooth_window,
        theta_ref=None if previous_ref is None else float(previous_ref),
        theta_ref_mode=str(record.detection.get("theta_ref_mode", "legacy_align")) if record.detection else "legacy_align",
    )
    augment_alignment_record(record_path, face_spans, side_spans, settings)
    if plot and thetas:
        concatenated = np.concatenate(thetas)
        offsets = np.cumsum([0] + [len(t) for t in thetas[:-1]])
        spans_plot = [CycleSpan(int(o), int(o + (f.mid - f.start)), int(o + len(t))) for o, f, t in zip(offsets, face_spans, thetas)]
        save_theta_plot(concatenated, record.fps, spans_plot, settings, record_path.parent / "theta_cycles_mid.png", f"person {person_id}: cycles (concatenated) with middles")
    return {"person_id": person_id, "cycles": len(face_spans), "mid_ratio_mean": float(np.mean(ratios)) if ratios else None, "mid_ratio_min": float(np.min(ratios)) if ratios else None, "mid_ratio_max": float(np.max(ratios)) if ratios else None}


def run_private(args: argparse.Namespace) -> int:
    log_root = _resolve(args.log_root)
    kpt_root = _resolve(args.kpt_root)
    if args.person:
        person_ids = [str(p) for p in args.person]
    else:
        person_ids = sorted((p.name.split("_", 1)[1] for p in log_root.glob("person_*") if (p / f"alignment_record_{p.name.split('_', 1)[1]}.json").is_file()), key=lambda s: (len(s), s))
    print(f"[cycles/private] {len(person_ids)} persons, records under {log_root}")
    results: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    def worker(person_id: str) -> None:
        try:
            summary = annotate_private_person(person_id, kpt_root=kpt_root, log_root=log_root, smooth_window=args.smooth_window, plot=not args.no_plot)
            results.append(summary)
            print(f"  ✓ person_{person_id}: {summary['cycles']} cycles, mid ratio mean {summary['mid_ratio_mean']:.3f} [{summary['mid_ratio_min']:.2f}, {summary['mid_ratio_max']:.2f}]")
        except Exception as error:  # noqa: BLE001 - report and continue
            failures.append((person_id, repr(error)))
            print(f"  ✗ person_{person_id}: {error!r}")

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as pool:
        list(pool.map(worker, person_ids))
    summary_path = log_root / "cycle_mid_summary.json"
    summary_path.write_text(json.dumps({"persons": sorted(results, key=lambda r: (len(str(r["person_id"])), str(r["person_id"]))), "failures": failures}, indent=2), encoding="utf-8")
    print(f"[cycles/private] done: {len(results)} ok, {len(failures)} failed; summary -> {summary_path}")
    return 1 if failures else 0


# ----------------------------------------------------------------------------- public datasets
def _settings_from_args(args: argparse.Namespace) -> DetectionSettings:
    return DetectionSettings(
        smooth_window=int(args.smooth_window),
        theta_ref_mode="auto_p10",
        min_period_sec=float(args.min_period_sec),
        max_period_sec=None if args.max_period_sec is None else float(args.max_period_sec),
        min_amplitude_rad=None if args.min_amplitude_rad is None else float(args.min_amplitude_rad),
    )


def annotate_sequence(
    *,
    dataset: str,
    subject_id: str,
    sequence_id: str,
    fps: float,
    view_a: np.ndarray,
    view_b: np.ndarray,
    valid_a: Optional[np.ndarray],
    valid_b: Optional[np.ndarray],
    views: Sequence[str],
    settings: DetectionSettings,
    out_root: Path,
    plot: bool,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Path, int]:
    """Detect cycles on one two-view sequence and write its record."""
    theta = fused_wrist_theta(view_a, view_b, valid_a=valid_a, valid_b=valid_b, smooth_window=settings.smooth_window)
    spans, used = detect_cycles(theta, fps, settings=settings, both_directions=True)
    path = cycle_record_path(out_root, subject_id, sequence_id)
    write_cycle_record(path, dataset=dataset, subject_id=subject_id, sequence_id=sequence_id, fps=fps, frames=int(view_a.shape[0]), spans=spans, detection=used, views=views, extra_metadata=extra_metadata)
    if plot:
        save_theta_plot(theta, fps, spans, used, path.with_name(f"{sequence_id}_theta_cycles.png"), f"{dataset} {subject_id}/{sequence_id}: {len(spans)} cycles")
    return path, len(spans)


def run_freeman(args: argparse.Namespace) -> int:
    from gymnastics.benchmarks.freeman.training import load_manifest_pair, load_manifest_sessions

    root = _resolve(args.benchmark_root)
    out_root = _resolve(args.out_root)
    settings = _settings_from_args(args)
    subjects = sorted({int(s) for s in args.subjects}) if args.subjects else sorted(int(p.stem.split("_")[1]) for p in (root / "manifests").glob("subject_*_sessions.json"))
    print(f"[cycles/freeman] {len(subjects)} subjects from {root} -> {out_root}")
    totals = {"sessions": 0, "with_cycles": 0, "cycles": 0}
    for subject in subjects:
        sessions = load_manifest_sessions(root, subject)
        counts: List[int] = []
        for session in sessions:
            pair = load_manifest_pair(root, session)
            _, n = annotate_sequence(
                dataset="freeman",
                subject_id=f"{subject:02d}",
                sequence_id=session.session_id,
                fps=float(session.fps),
                view_a=pair.view_a.points3d,
                view_b=pair.view_b.points3d,
                valid_a=pair.view_a.valid3d,
                valid_b=pair.view_b.valid3d,
                views=(pair.view_a.view_id, pair.view_b.view_id),
                settings=settings,
                out_root=out_root,
                plot=not args.no_plot,
                extra_metadata={"split": session.split},
            )
            counts.append(n)
        totals["sessions"] += len(counts)
        totals["with_cycles"] += sum(1 for c in counts if c)
        totals["cycles"] += sum(counts)
        print(f"  subject {subject:02d}: {len(counts)} sessions, {sum(1 for c in counts if c)} periodic, {sum(counts)} cycles")
    (out_root / "summary.json").write_text(json.dumps({"settings": settings.to_dict(), **totals}, indent=2), encoding="utf-8")
    print(f"[cycles/freeman] done: {totals}")
    return 0


def run_unity(args: argparse.Namespace) -> int:
    from gymnastics.benchmarks.unity.dataset import group_evaluation_sequences, load_unity_benchmark
    from gymnastics.benchmarks.unity.sam3d import load_sam3d_camera_cache

    benchmark = load_unity_benchmark(_resolve(args.benchmark_root))
    cache_root = _resolve(args.sam3d_cache_root)
    out_root = _resolve(args.out_root)
    settings = _settings_from_args(args)
    totals = {"sequences": 0, "with_cycles": 0, "cycles": 0}
    for sequence_id, frames in group_evaluation_sequences(benchmark).items():
        if sequence_id == "static_sweep":
            continue
        sample_ids = np.asarray([frame.sample_id for frame in frames], dtype=np.int64)
        cam0 = load_sam3d_camera_cache(cache_root, "cam0", sample_ids)
        cam1 = load_sam3d_camera_cache(cache_root, "cam1", sample_ids)
        _, n = annotate_sequence(
            dataset="unity",
            subject_id=sequence_id,
            sequence_id=sequence_id,
            fps=float(args.fps),
            view_a=cam0.points_3d,
            view_b=cam1.points_3d,
            valid_a=cam0.valid_3d,
            valid_b=cam1.valid_3d,
            views=("cam0", "cam1"),
            settings=settings,
            out_root=out_root,
            plot=not args.no_plot,
        )
        totals["sequences"] += 1
        totals["with_cycles"] += int(n > 0)
        totals["cycles"] += n
        print(f"  {sequence_id}: {n} cycles")
    (out_root / "summary.json").write_text(json.dumps({"settings": settings.to_dict(), **totals}, indent=2), encoding="utf-8")
    print(f"[cycles/unity] done: {totals}")
    return 0


# ----------------------------------------------------------------------------- index
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_private_record(record_path: Path, out_root: Path) -> Tuple[Path, int]:
    """Export one private alignment record as a uniform ``cycle_record_v1`` file.

    ``frames`` carries the face (view A) video frames and ``side_frames`` the
    side video frames; ``metadata.source_record`` and ``source_sha256`` tie
    the copy to the authoritative alignment record.
    """
    record = read_cycle_record(record_path)
    assert record.side_cycles is not None
    payload: Dict[str, object] = {
        "format": FORMAT_V1,
        "metadata": {
            "dataset": "gymnastics",
            "subject_id": record.subject_id,
            "sequence_id": "all_cycles",
            "fps": record.fps,
            "frames": None,
            "views": ["face", "side"],
            "offset_side_to_face": record.metadata.get("offset_side_to_face"),
            "cycle_detection": dict(record.detection),
            "source_record": str(record_path),
            "source_sha256": _sha256(record_path),
            "audit_plot": str(record_path.parent / "theta_cycles_mid.png"),
        },
        "cycles": [
            {
                "cycle_index": i,
                "frames": {"start": fs, "mid": fm, "end": fe},
                "side_frames": {"start": ss, "mid": sm, "end": se},
            }
            for i, ((fs, fm, fe), (ss, sm, se)) in enumerate(zip(record.cycles, record.side_cycles))
        ],
    }
    path = cycle_record_path(out_root / "gymnastics", record.subject_id, "all_cycles")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, len(record.cycles)


def _dataset_stats(root: Path) -> Dict[str, object]:
    files = sorted(root.glob("subject_*/*.json"))
    sequences = len(files)
    with_cycles = cycles = 0
    with_mids = 0
    settings: Dict[str, object] = {}
    for path in files:
        record = read_cycle_record(path)
        cycles += len(record.cycles)
        with_cycles += int(bool(record.cycles))
        with_mids += int(record.has_mids)
        if not settings and record.detection:
            settings = dict(record.detection)
    return {"root": str(root), "sequences": sequences, "with_cycles": with_cycles, "with_mids": with_mids, "cycles": cycles, "detection": settings}


README_TEMPLATE = """# Cycle records

Generated by `gymnastics align cycles index` on {timestamp}. One directory per
dataset, one `cycle_record_v1` JSON per sequence under `subject_*/`, plus an
audit plot next to it. `index.json` holds the counts below in machine-readable
form.

## Definition (shared by every dataset, `src/gymnastics/alignment/cycles.py`)

* signal: right-wrist azimuth in the pelvis body frame of the two-view
  body-frame fusion, smoothed (11 frames) and unwrapped;
* cycle start: upward crossing of `theta_ref` (10th percentile) while rotating
  counter-clockwise, at least `min_period_sec` apart;
* cycle middle (`mid`): the turn-around frame = extremum of the signal inside
  the cycle; `[start, mid)` is the outward motion, `[mid, end)` the return.

## Datasets

| dataset | records | sequences with cycles | cycles | source of truth |
|---|---:|---:|---:|---|
{rows}

* `gymnastics/`: read-only export of `local/runs/split_cycle/person_<id>/alignment_record_<id>.json`
  (the file the whole pipeline reads; `frames` = face video frames, `side_frames` = side video
  frames, offset = `offset_side_to_face`). Re-export with `gymnastics align cycles index`.
* `freeman/`, `unity/`: written directly by `gymnastics align cycles freeman|unity`;
  `frames` are the shared (synchronised) frame ids.

## Regeneration

```bash
gymnastics align cycles private   # adds mid to the 137 alignment records (~1 min/person)
gymnastics align cycles freeman   # local/runs/cycle_records/freeman
gymnastics align cycles unity     # local/runs/cycle_records/unity
gymnastics align cycles index     # this tree + index.json + README.md
```

## Source data on this machine

{sources}
"""


def run_index(args: argparse.Namespace) -> int:
    records_root = _resolve(args.records_root)
    split_root = _resolve(args.log_root)
    records_root.mkdir(parents=True, exist_ok=True)
    # 1) export private records
    exported = 0
    for record_path in sorted(split_root.glob("person_*/alignment_record_*.json")):
        export_private_record(record_path, records_root)
        exported += 1
    summary = split_root / "cycle_mid_summary.json"
    if summary.is_file():
        shutil.copy2(summary, records_root / "gymnastics" / "summary.json")
    # 2) gather logs
    logs = records_root / "logs"
    logs.mkdir(exist_ok=True)
    for name in ("private_mid_annotation.log", "public_annotation.log"):
        source = records_root / name
        if source.is_file():
            shutil.move(str(source), str(logs / name))
    # 3) stats + index
    datasets = {}
    for name in ("gymnastics", "freeman", "unity"):
        root = records_root / name
        if root.is_dir():
            datasets[name] = _dataset_stats(root)
    sources = {
        "gymnastics": {
            "videos": str(_resolve(args.raw_root) / "person" / "<id>" / "ID<id>_{face,side}.MOV"),
            "keypoints_3d_sam3d": str(_resolve(args.kpt_root) / "person" / "<id>" / "{face,side}" / "*_sam3d_body.npz"),
            "keypoints_3d_reference": str(_resolve(args.triangulated_root) / "person_<id>" / "cycle_NNN" / "joints_3d_sequence.npz"),
            "cycle_records_authoritative": str(split_root / "person_<id>" / "alignment_record_<id>.json"),
        },
        "freeman": {
            "videos": str(_resolve(args.freeman_root) / "videos_extracted" / "<session>" / "vframes" / "cNN.mp4"),
            "keypoints_3d_sam3d": str(_resolve(args.freeman_benchmark_root) / "sam3d" / "subject_NN" / "<session>" / "<view>" / "prediction.npz"),
            "keypoints_3d_reference": str(_resolve(args.freeman_root) / "work" / "shared" / "30FPS" / "keypoints3d" / "<session>.npy"),
            "session_manifests": str(_resolve(args.freeman_benchmark_root) / "manifests" / "subject_NN_sessions.json"),
        },
        "unity": {
            "images": str(_resolve(args.unity_root) / "images" / "{cam0,cam1}"),
            "keypoints_3d_sam3d": str(_resolve(args.unity_cache_root) / "{cam0,cam1}" / "<sample_id>.npz"),
            "keypoints_3d_reference": str(_resolve(args.unity_root) / "manifest.jsonl (keypoints_3d, world metres)"),
        },
    }
    index = {"generated": datetime.now().isoformat(timespec="seconds"), "datasets": datasets, "sources": sources}
    (records_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    truth = {"gymnastics": "`local/runs/split_cycle/.../alignment_record_<id>.json`", "freeman": "these files", "unity": "these files"}
    rows = "\n".join(f"| {name} | {d['sequences']} | {d['with_cycles']} | {d['cycles']} | {truth[name]} |" for name, d in datasets.items())
    source_lines = []
    for name, entries in sources.items():
        source_lines.append(f"**{name}**")
        source_lines.extend(f"* {key}: `{value}`" for key, value in entries.items())
        source_lines.append("")
    (records_root / "README.md").write_text(README_TEMPLATE.format(timestamp=index["generated"], rows=rows, sources="\n".join(source_lines)), encoding="utf-8")
    print(f"[cycles/index] exported {exported} private records; index -> {records_root / 'index.json'}")
    for name, d in datasets.items():
        print(f"  {name}: {d['sequences']} records, {d['with_cycles']} with cycles, {d['cycles']} cycles")
    return 0


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gymnastics align cycles", description="Offline cycle and middle annotation.")
    sub = parser.add_subparsers(dest="dataset", required=True)

    def common(p: argparse.ArgumentParser, *, public: bool) -> None:
        p.add_argument("--smooth-window", type=int, default=11, help="moving-average window in frames (odd)")
        p.add_argument("--no-plot", action="store_true", help="skip the audit plots")
        if public:
            p.add_argument("--out-root", type=Path, default=None, help="record root (default: local/runs/cycle_records/<dataset>)")
            p.add_argument("--min-period-sec", type=float, default=0.8)
            p.add_argument("--max-period-sec", type=float, default=6.0)
            p.add_argument("--min-amplitude-rad", type=float, default=0.5, help="reject sequences whose wrist azimuth swings less than this")

    private = sub.add_parser("private", help="add middles to the private alignment records")
    private.add_argument("--log-root", type=Path, default=Path("local/runs/split_cycle"))
    private.add_argument("--kpt-root", type=Path, default=Path(os.environ.get("GYMNASTICS_DATA_ROOT", "/home/data/xchen/gymnastics")) / "sam3d_body_results")
    private.add_argument("--person", nargs="*", default=None)
    private.add_argument("--threads", type=int, default=8)
    common(private, public=False)

    freeman = sub.add_parser("freeman", help="detect cycles on the FreeMan benchmark cache")
    freeman.add_argument("--benchmark-root", type=Path, default=Path("local/runs/freeman_benchmark_cluster"))
    freeman.add_argument("--subjects", nargs="*", type=int, default=None)
    common(freeman, public=True)

    unity = sub.add_parser("unity", help="detect cycles on the Unity benchmark sequences")
    unity.add_argument("--benchmark-root", type=Path, default=Path(os.environ.get("GYMNASTICS_DATA_ROOT", "/home/data/xchen/gymnastics")) / "unity_benchmark")
    unity.add_argument("--sam3d-cache-root", type=Path, default=Path("local/runs/unity_benchmark/sam3d"))
    unity.add_argument("--fps", type=float, default=60.0)
    common(unity, public=True)

    data_root = Path(os.environ.get("GYMNASTICS_DATA_ROOT", "/home/data/xchen/gymnastics"))
    index = sub.add_parser("index", help="export private records, gather logs, write index.json and README.md")
    index.add_argument("--records-root", type=Path, default=Path("local/runs/cycle_records"))
    index.add_argument("--log-root", type=Path, default=Path("local/runs/split_cycle"))
    index.add_argument("--raw-root", type=Path, default=data_root / "raw")
    index.add_argument("--kpt-root", type=Path, default=data_root / "sam3d_body_results")
    index.add_argument("--triangulated-root", type=Path, default=data_root / "sam3d_triangulated")
    index.add_argument("--unity-root", type=Path, default=data_root / "unity_benchmark")
    index.add_argument("--unity-cache-root", type=Path, default=Path("local/runs/unity_benchmark/sam3d"))
    index.add_argument("--freeman-root", type=Path, default=Path(os.environ.get("FREEMAN_ROOT", "/home/data/xchen/public_datasets/multiview_human/FreeMan")))
    index.add_argument("--freeman-benchmark-root", type=Path, default=Path("local/runs/freeman_benchmark_cluster"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "out_root", None) is None and args.dataset in {"freeman", "unity"}:
        args.out_root = Path("local/runs/cycle_records") / args.dataset
    if args.dataset == "private":
        return run_private(args)
    if args.dataset == "freeman":
        return run_freeman(args)
    if args.dataset == "unity":
        return run_unity(args)
    return run_index(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
