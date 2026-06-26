#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path("/home/data/xchen/gymnastics/sam3d_triangulated/person")
OUT_DIR = Path("logs/analysis/triangulated_results")


def _person_key(path: Path) -> int:
    return int(path.name.split("_")[1])


def _cycle_key(path: Path) -> int:
    return int(path.name.split("_")[1])


def _stat_line(values):
    return min(values), median(values), max(values), mean(values)


def collect_rows():
    cycle_rows = []
    person_rows = []

    for person_dir in sorted(ROOT.glob("person_*"), key=_person_key):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name.split("_", 1)[1]
        cycles = []

        for cycle_dir in sorted(person_dir.glob("cycle_*"), key=_cycle_key):
            summary_path = cycle_dir / "summary.json"
            seq_path = cycle_dir / "joints_3d_sequence.npz"
            video_paths = sorted(cycle_dir.glob("*_3d.mp4"))
            joints_dir = cycle_dir / "joints_3d"
            joints_json_count = (
                len(list(joints_dir.glob("*_joints_3d.json"))) if joints_dir.exists() else 0
            )

            if not summary_path.exists():
                row = {
                    "person_id": person_id,
                    "cycle_index": _cycle_key(cycle_dir),
                    "summary_exists": False,
                    "sequence_exists": seq_path.exists(),
                    "video_exists": bool(video_paths),
                    "joints_json_count": joints_json_count,
                    "cycle_dir": str(cycle_dir),
                }
                cycle_rows.append(row)
                cycles.append(row)
                continue

            with summary_path.open() as f:
                data = json.load(f)

            processed = int(data.get("processed_frames", 0) or 0)
            missing = int(data.get("missing_pairs", 0) or 0)
            face_error = data.get("face_reprojection_error_mean_px")
            side_error = data.get("side_reprojection_error_mean_px")
            row = {
                "person_id": str(data.get("person_id", person_id)),
                "cycle_index": int(data.get("cycle_index", _cycle_key(cycle_dir))),
                "face_start": data.get("face_video_frames", {}).get("start"),
                "face_end": data.get("face_video_frames", {}).get("end"),
                "side_start": data.get("side_video_frames", {}).get("start"),
                "side_end": data.get("side_video_frames", {}).get("end"),
                "processed_frames": processed,
                "missing_pairs": missing,
                "missing_pair_ratio": (missing / processed) if processed else "",
                "num_joints": data.get("num_joints"),
                "face_reprojection_error_mean_px": face_error,
                "side_reprojection_error_mean_px": side_error,
                "mean_reprojection_error_px": mean([face_error, side_error]),
                "summary_exists": True,
                "sequence_exists": seq_path.exists(),
                "video_exists": bool(video_paths),
                "joints_json_count": joints_json_count,
                "summary_path": str(summary_path),
                "sequence_path": str(seq_path),
                "video_path": str(video_paths[0]) if video_paths else "",
                "cycle_dir": str(cycle_dir),
            }
            cycle_rows.append(row)
            cycles.append(row)

        valid = [row for row in cycles if row.get("summary_exists")]
        if valid:
            frames = [int(row["processed_frames"]) for row in valid]
            missing = [int(row["missing_pairs"]) for row in valid]
            face_errors = [float(row["face_reprojection_error_mean_px"]) for row in valid]
            side_errors = [float(row["side_reprojection_error_mean_px"]) for row in valid]
            mean_errors = [float(row["mean_reprojection_error_px"]) for row in valid]
            person_rows.append(
                {
                    "person_id": person_id,
                    "cycle_count": len(cycles),
                    "cycle_summary_count": len(valid),
                    "total_processed_frames": sum(frames),
                    "min_cycle_frames": min(frames),
                    "median_cycle_frames": median(frames),
                    "max_cycle_frames": max(frames),
                    "total_missing_pairs": sum(missing),
                    "avg_face_reprojection_error_mean_px": mean(face_errors),
                    "avg_side_reprojection_error_mean_px": mean(side_errors),
                    "avg_mean_reprojection_error_px": mean(mean_errors),
                    "max_mean_reprojection_error_px": max(mean_errors),
                    "complete_sequences": sum(
                        1 for row in cycles if row.get("sequence_exists")
                    ),
                    "complete_videos": sum(1 for row in cycles if row.get("video_exists")),
                    "person_dir": str(person_dir),
                }
            )
        else:
            person_rows.append(
                {
                    "person_id": person_id,
                    "cycle_count": len(cycles),
                    "cycle_summary_count": 0,
                    "person_dir": str(person_dir),
                }
            )

    return cycle_rows, person_rows


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def write_markdown(path: Path, cycle_rows, person_rows, cycle_csv: Path, person_csv: Path):
    valid_cycles = [row for row in cycle_rows if row.get("summary_exists")]
    face_all = [float(row["face_reprojection_error_mean_px"]) for row in valid_cycles]
    side_all = [float(row["side_reprojection_error_mean_px"]) for row in valid_cycles]
    mean_all = [float(row["mean_reprojection_error_px"]) for row in valid_cycles]
    frames_all = [int(row["processed_frames"]) for row in valid_cycles]
    missing_all = [int(row["missing_pairs"]) for row in valid_cycles]

    worst_cycles = sorted(
        valid_cycles, key=lambda row: float(row["mean_reprojection_error_px"]), reverse=True
    )[:15]
    best_cycles = sorted(
        valid_cycles, key=lambda row: float(row["mean_reprojection_error_px"])
    )[:10]
    worst_persons = sorted(
        person_rows,
        key=lambda row: float(row.get("avg_mean_reprojection_error_px") or -1),
        reverse=True,
    )[:12]
    best_persons = sorted(
        person_rows,
        key=lambda row: float(row.get("avg_mean_reprojection_error_px") or 999999),
    )[:10]

    camera_dir = ROOT / "_camera"
    camera_files = sorted(str(path) for path in camera_dir.glob("*")) if camera_dir.exists() else []

    with path.open("w") as f:
        f.write("# Triangulated SAM3D Results Report\n\n")
        f.write(f"- Source: `{ROOT}`\n")
        f.write(f"- Person directories: `{len(person_rows)}`\n")
        f.write(f"- Cycle directories with summaries: `{len(valid_cycles)}`\n")
        f.write(f"- Cycle directories total: `{len(cycle_rows)}`\n")
        f.write(f"- Total processed frames: `{sum(frames_all)}`\n")
        f.write(f"- Total missing pairs: `{sum(missing_all)}`\n")
        f.write(f"- Cycles with nonzero missing pairs: `{sum(1 for x in missing_all if x)}`\n")
        f.write(
            f"- Cycles with `joints_3d_sequence.npz`: "
            f"`{sum(1 for row in cycle_rows if row.get('sequence_exists'))}`\n"
        )
        f.write(
            f"- Cycles with `*_3d.mp4`: "
            f"`{sum(1 for row in cycle_rows if row.get('video_exists'))}`\n\n"
        )

        f.write("## Aggregate Statistics\n\n")
        stats = [
            ("Processed frames per cycle", frames_all, ""),
            ("Face reprojection error mean", face_all, " px"),
            ("Side reprojection error mean", side_all, " px"),
            ("Mean reprojection error", mean_all, " px"),
        ]
        for name, values, suffix in stats:
            mn, med, mx, avg = _stat_line(values)
            f.write(
                f"- {name}: min `{mn:.3f}{suffix}`, median `{med:.3f}{suffix}`, "
                f"max `{mx:.3f}{suffix}`, mean `{avg:.3f}{suffix}`\n"
            )
        f.write("\n")

        f.write("## Worst Persons By Average Mean Reprojection Error\n\n")
        f.write("| person | cycles | frames | avg face px | avg side px | avg mean px | max mean px |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in worst_persons:
            f.write(
                f"| {row['person_id']} | {row.get('cycle_summary_count', '')} | "
                f"{row.get('total_processed_frames', '')} | "
                f"{float(row.get('avg_face_reprojection_error_mean_px', 0)):.3f} | "
                f"{float(row.get('avg_side_reprojection_error_mean_px', 0)):.3f} | "
                f"{float(row.get('avg_mean_reprojection_error_px', 0)):.3f} | "
                f"{float(row.get('max_mean_reprojection_error_px', 0)):.3f} |\n"
            )
        f.write("\n")

        f.write("## Best Persons By Average Mean Reprojection Error\n\n")
        f.write("| person | cycles | frames | avg face px | avg side px | avg mean px |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in best_persons:
            f.write(
                f"| {row['person_id']} | {row.get('cycle_summary_count', '')} | "
                f"{row.get('total_processed_frames', '')} | "
                f"{float(row.get('avg_face_reprojection_error_mean_px', 0)):.3f} | "
                f"{float(row.get('avg_side_reprojection_error_mean_px', 0)):.3f} | "
                f"{float(row.get('avg_mean_reprojection_error_px', 0)):.3f} |\n"
            )
        f.write("\n")

        f.write("## Worst Cycles By Mean Reprojection Error\n\n")
        f.write(
            "| person | cycle | frames | face frames | side frames | face px | side px | mean px | path |\n"
        )
        f.write("|---:|---:|---:|---|---|---:|---:|---:|---|\n")
        for row in worst_cycles:
            f.write(
                f"| {row['person_id']} | {row['cycle_index']} | {row['processed_frames']} | "
                f"{row['face_start']}-{row['face_end']} | "
                f"{row['side_start']}-{row['side_end']} | "
                f"{float(row['face_reprojection_error_mean_px']):.3f} | "
                f"{float(row['side_reprojection_error_mean_px']):.3f} | "
                f"{float(row['mean_reprojection_error_px']):.3f} | "
                f"`{row['cycle_dir']}` |\n"
            )
        f.write("\n")

        f.write("## Best Cycles By Mean Reprojection Error\n\n")
        f.write("| person | cycle | frames | face px | side px | mean px | path |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---|\n")
        for row in best_cycles:
            f.write(
                f"| {row['person_id']} | {row['cycle_index']} | {row['processed_frames']} | "
                f"{float(row['face_reprojection_error_mean_px']):.3f} | "
                f"{float(row['side_reprojection_error_mean_px']):.3f} | "
                f"{float(row['mean_reprojection_error_px']):.3f} | "
                f"`{row['cycle_dir']}` |\n"
            )
        f.write("\n")

        f.write("## Camera Visualization Files\n\n")
        if camera_files:
            for camera_file in camera_files:
                f.write(f"- `{camera_file}`\n")
        else:
            f.write("- None found.\n")
        f.write("\n")

        f.write("## Detail Files\n\n")
        f.write(f"- Cycle details CSV: `{cycle_csv}`\n")
        f.write(f"- Person summary CSV: `{person_csv}`\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cycle_rows, person_rows = collect_rows()

    cycle_csv = OUT_DIR / "triangulated_cycle_details.csv"
    person_csv = OUT_DIR / "triangulated_person_summary.csv"
    report_md = OUT_DIR / "triangulated_results_report.md"

    cycle_fields = [
        "person_id",
        "cycle_index",
        "face_start",
        "face_end",
        "side_start",
        "side_end",
        "processed_frames",
        "missing_pairs",
        "missing_pair_ratio",
        "num_joints",
        "face_reprojection_error_mean_px",
        "side_reprojection_error_mean_px",
        "mean_reprojection_error_px",
        "summary_exists",
        "sequence_exists",
        "video_exists",
        "joints_json_count",
        "summary_path",
        "sequence_path",
        "video_path",
        "cycle_dir",
    ]
    person_fields = [
        "person_id",
        "cycle_count",
        "cycle_summary_count",
        "total_processed_frames",
        "min_cycle_frames",
        "median_cycle_frames",
        "max_cycle_frames",
        "total_missing_pairs",
        "avg_face_reprojection_error_mean_px",
        "avg_side_reprojection_error_mean_px",
        "avg_mean_reprojection_error_px",
        "max_mean_reprojection_error_px",
        "complete_sequences",
        "complete_videos",
        "person_dir",
    ]

    write_csv(cycle_csv, cycle_fields, cycle_rows)
    write_csv(person_csv, person_fields, person_rows)
    write_markdown(report_md, cycle_rows, person_rows, cycle_csv, person_csv)

    print(report_md)
    print(cycle_csv)
    print(person_csv)


if __name__ == "__main__":
    main()
