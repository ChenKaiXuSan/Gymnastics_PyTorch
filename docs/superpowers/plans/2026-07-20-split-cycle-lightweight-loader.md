# Split-Cycle Lightweight Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove full-frame retention from the split-cycle SAM3D loader, verify the fix with a red-green test cycle, then generate and validate split-cycle records for every available person.

**Architecture:** Keep `load_sam3d_body_sequence()` and its tuple return contract, but make each `all_info` entry an independent lightweight dictionary containing only `frame_idx` and `pred_keypoints_3d`. Run the existing alignment and segmentation pipeline unchanged, with a writable command-level Numba cache for audio alignment.

**Tech Stack:** Python 3.10, NumPy NPZ, pytest, OpenCV, librosa/Numba, `gymnastic` Conda environment.

## Global Constraints

- Preserve the public return form `(all_info, kpts3d)` of `load_sam3d_body_sequence()`.
- Do not retain `frame`, `pred_vertices`, or the original `output` dictionary in returned metadata.
- Do not modify the SAM3D-Body NPZ format, DTW, audio alignment, cycle detection, or video splitting algorithms.
- Do not modify `fuse/load.py`.
- Do not rewrite split-cycle results for persons 1–68.
- Use `conda run -n gymnastic ...` for project Python and test commands.
- Keep the existing uncommitted `AGENTS.md` change out of all commits.

---

### Task 1: Make the SAM3D sequence loader memory-safe

**Files:**
- Create: `tests/test_split_cycle_load.py`
- Modify: `split_cycle/load.py:30-104`

**Interfaces:**
- Consumes: Per-frame files at `<root>/person/<person_id>/<view>/*_sam3d_body.npz`, each with scalar object key `output` containing `frame_idx` and `pred_keypoints_3d`.
- Produces: `load_sam3d_body_sequence(...) -> tuple[list[dict], np.ndarray]`, where every dictionary has exactly `frame_idx` and `pred_keypoints_3d`, and the array has shape `(T, J, 3)`.

- [ ] **Step 1: Write the failing loader regression test**

Create `tests/test_split_cycle_load.py` with:

```python
import numpy as np

from split_cycle.load import load_sam3d_body_sequence


def _write_sam3d_frame(base, frame_idx, keypoint_value):
    base.mkdir(parents=True, exist_ok=True)
    output = {
        "frame_idx": np.int64(frame_idx),
        "pred_keypoints_3d": np.full(
            (70, 3), keypoint_value, dtype=np.float32
        ),
        "frame": np.full((8, 6, 3), frame_idx, dtype=np.uint8),
        "pred_vertices": np.full((4, 3), frame_idx, dtype=np.float32),
    }
    scalar = np.empty((), dtype=object)
    scalar[()] = output
    np.savez(base / f"{frame_idx:06d}_sam3d_body.npz", output=scalar)


def test_loader_returns_distinct_lightweight_metadata_and_stacked_keypoints(tmp_path):
    root = tmp_path / "sam3d_body_results"
    base = root / "person" / "69" / "face"
    _write_sam3d_frame(base, frame_idx=0, keypoint_value=1.0)
    _write_sam3d_frame(base, frame_idx=1, keypoint_value=2.0)

    all_info, keypoints = load_sam3d_body_sequence(
        root, person_id="69", subdir="face"
    )

    assert keypoints.shape == (2, 70, 3)
    np.testing.assert_array_equal(keypoints[0], np.ones((70, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        keypoints[1], np.full((70, 3), 2.0, dtype=np.float32)
    )
    assert all_info[0] is not all_info[1]
    assert [info["frame_idx"] for info in all_info] == [0, 1]
    assert all(set(info) == {"frame_idx", "pred_keypoints_3d"} for info in all_info)
    assert all("frame" not in info for info in all_info)
    assert all("pred_vertices" not in info for info in all_info)

    all_info[0]["frame_idx"] = 99
    assert all_info[1]["frame_idx"] == 1
```

- [ ] **Step 2: Run the new test and verify the old implementation fails**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_split_cycle_load.py -q
```

Expected: FAIL because the old loader appends the same dictionary object for both frames and retains `frame`, integer frame keys, and the complete outputs.

- [ ] **Step 3: Implement the minimal compatible loader**

Replace the accumulation block in `split_cycle/load.py` with:

```python
    all_kpts: List[np.ndarray] = []
    all_info: List[Dict] = []

    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            output = data["output"].item()
            frame_idx = int(np.asarray(output["frame_idx"]).item())
            pred_keypoints_3d = np.array(
                output["pred_keypoints_3d"], copy=True
            )

        frame_info = {
            "frame_idx": frame_idx,
            "pred_keypoints_3d": pred_keypoints_3d,
        }
        all_info.append(frame_info)
        all_kpts.append(pred_keypoints_3d)

    kpts3d = np.stack(all_kpts, axis=0)
    return all_info, kpts3d
```

Also update the return-value docstring to say that `all_info` contains lightweight frame metadata rather than complete SAM3D outputs.

- [ ] **Step 4: Run the new test and verify it passes**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_split_cycle_load.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run focused split-cycle regression tests**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_split_cycle_audio_alignment.py tests/test_split_cycle_cli.py -q
```

Expected: `7 passed` with no failures.

- [ ] **Step 6: Review and commit only the loader fix and its test**

Run:

```bash
git diff --check -- split_cycle/load.py tests/test_split_cycle_load.py
git diff -- split_cycle/load.py tests/test_split_cycle_load.py
git add split_cycle/load.py tests/test_split_cycle_load.py
git commit -m "fix: make split-cycle SAM3D loading memory-safe"
```

Expected: one commit containing only `split_cycle/load.py` and `tests/test_split_cycle_load.py`; `AGENTS.md` remains unstaged.

---

### Task 2: Re-run and validate the ID69 smoke case with audio alignment

**Files:**
- Runtime output: `logs/split_cycle/person_69/`
- Temporary backup: a new directory under `/tmp/split-cycle-id69-backup.*`

**Interfaces:**
- Consumes: ID69 face/side raw videos and complete SAM3D per-frame NPZ outputs.
- Produces: `logs/split_cycle/person_69/alignment_record_69.json` plus matching face/side cycle videos.

- [ ] **Step 1: Move the keypoint-only smoke output to a recoverable temporary backup**

Run:

```bash
split_cycle_backup_dir=$(mktemp -d /tmp/split-cycle-id69-backup.XXXXXX)
mv logs/split_cycle/person_69 "$split_cycle_backup_dir/"
```

Expected: `logs/split_cycle/person_69` no longer exists, while the previous result remains recoverable under the printed `/tmp/split-cycle-id69-backup.*` directory.

- [ ] **Step 2: Re-run ID69 with a writable Numba cache**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/gymnastics_numba_cache conda run --no-capture-output -n gymnastic python -m split_cycle.main --person 69 --threads 1
```

Expected: summary reports `1/1 persons processed successfully`; audio alignment prints a numeric offset and confidence rather than a cache error.

- [ ] **Step 3: Validate the ID69 record and cycle videos**

Run:

```bash
conda run -n gymnastic python -c '
import json
from pathlib import Path

root = Path("logs/split_cycle/person_69")
record = json.loads((root / "alignment_record_69.json").read_text())
metadata = record["metadata"]
cycles = record["cycles"]
assert isinstance(metadata["offset_side_to_face"], int)
assert metadata["offset_audio_xcorr"] is not None
assert metadata["audio_confidence"] >= 0.15
assert metadata["offset_source"] in {"kpt_audio_avg", "kpt"}
assert cycles
assert len(list((root / "face").glob("cycle_*.mp4"))) == len(cycles)
assert len(list((root / "side").glob("cycle_*.mp4"))) == len(cycles)
print({"person": 69, "metadata": metadata, "cycles": len(cycles)})
'
```

Expected: exit code 0, non-null audio offset, confidence at least `0.15`, and equal nonzero cycle counts for both views.

---

### Task 3: Process all remaining new people and validate the complete split-cycle dataset

**Files:**
- Runtime outputs: `logs/split_cycle/person_70/` through `person_138/`, excluding `person_135/`

**Interfaces:**
- Consumes: Complete raw face/side videos and SAM3D results for persons 70–134 and 136–138.
- Produces: One alignment record and matching cycle-video sets for every available person; persons 1–68 remain unchanged.

- [ ] **Step 1: Run the remaining people with conservative concurrency**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/gymnastics_numba_cache conda run --no-capture-output -n gymnastic python -m split_cycle.main --threads 6 --person {70..134} {136..138}
```

Expected: final summary reports `68/68 persons processed successfully` and `0 persons failed`.

- [ ] **Step 2: Validate every alignment record, range, and cycle-video count**

Run:

```bash
conda run -n gymnastic python -c '
import glob
import json
from pathlib import Path

import cv2

raw_root = Path("/home/data/xchen/gymnastics/raw/person")
sam_root = Path("/home/data/xchen/gymnastics/sam3d_body_results/person")
split_root = Path("logs/split_cycle")
available = []
errors = []
low_overlap = []
audio_available = 0
total_cycles = 0

for person_id in range(1, 139):
    raw = {}
    ready = True
    for view in ("face", "side"):
        videos = sorted(raw_root.joinpath(str(person_id)).glob(f"*{view}*.*"))
        outputs = list(sam_root.joinpath(str(person_id), view).glob("*_sam3d_body.npz"))
        if len(videos) != 1 or not outputs:
            ready = False
            break
        raw[view] = videos[0]
    if not ready:
        continue

    available.append(person_id)
    record_path = split_root / f"person_{person_id}" / f"alignment_record_{person_id}.json"
    if not record_path.is_file():
        errors.append((person_id, "missing_record"))
        continue

    try:
        record = json.loads(record_path.read_text())
        metadata = record["metadata"]
        cycles = record["cycles"]
        assert int(metadata["person_id"]) == person_id
        assert isinstance(metadata["offset_side_to_face"], int)
        assert cycles
        if metadata.get("offset_audio_xcorr") is not None:
            audio_available += 1

        frame_counts = {}
        for view in ("face", "side"):
            cap = cv2.VideoCapture(str(raw[view]))
            frame_counts[view] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            previous_end = 0
            for cycle in cycles:
                bounds = cycle[f"{view}_video_frames"]
                start, end = int(bounds["start"]), int(bounds["end"])
                assert 0 <= start < end <= frame_counts[view]
                assert start >= previous_end
                previous_end = end

            video_count = len(
                list((record_path.parent / view).glob("cycle_*.mp4"))
            )
            assert video_count == len(cycles)

        overlap = metadata["overlap_union_range"]
        overlap_length = int(overlap[1]) - int(overlap[0])
        ratio = overlap_length / max(frame_counts.values())
        if ratio < 0.3:
            low_overlap.append((person_id, ratio))
        total_cycles += len(cycles)
    except Exception as exc:
        errors.append((person_id, type(exc).__name__, str(exc)))

assert available == [person_id for person_id in range(1, 139) if person_id != 135]
assert not errors, errors
print({
    "people": len(available),
    "records": len(available),
    "cycles": total_cycles,
    "audio_available": audio_available,
    "low_overlap": low_overlap,
    "errors": errors,
})
'
```

Expected: `people=137`, `records=137`, `errors=[]`; every record has at least one cycle and matching face/side cycle-video counts. Report but do not silently discard any `low_overlap` entries.

- [ ] **Step 3: Run final focused regression verification**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_split_cycle_load.py tests/test_split_cycle_audio_alignment.py tests/test_split_cycle_cli.py -q
```

Expected: `8 passed` with no failures.

- [ ] **Step 4: Verify repository scope and preserve user changes**

Run:

```bash
git status --short
git log -2 --oneline
```

Expected: the loader-fix commit and design commit are present; the existing `AGENTS.md` modification remains unstaged and unchanged.
