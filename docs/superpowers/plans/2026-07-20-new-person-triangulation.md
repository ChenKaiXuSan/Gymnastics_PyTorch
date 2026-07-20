# New-Person SAM3D Triangulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely triangulate all 465 cycles for persons 69–134 and 136–138, preserve persons 1–68, and validate the resulting 137-person/923-cycle dataset.

**Architecture:** Make the triangulation entry point rebuild its root summary from all per-person summaries so chunked execution remains consistent. Add a standalone validator that treats split-cycle records as the expected inventory, performs strict sequence/file checks, excludes person 119 only from aggregate quality metrics, and writes a JSON audit report.

**Tech Stack:** Python 3.10, NumPy NPZ, JSON, pytest, OpenCV triangulation, OmegaConf, `gymnastic` Conda environment.

## Global Constraints

- Process only persons 69–134 and 136–138; person 135 is absent.
- Do not invoke triangulation for persons 1–68 or alter their per-person files.
- Triangulate person 119, mark it `excluded_low_quality`, and exclude it from aggregate metrics.
- Run production triangulation with `--no-vis --no-video`.
- Treat sequence shape, finite values, processed frame count, missing pairs, expected cycles, and JSON count as hard integrity requirements.
- Treat a per-view mean reprojection error over `60.0` px as a warning.
- Use `conda run -n gymnastic ...` for project Python and test commands.
- Keep the existing uncommitted `AGENTS.md` and `docs/superpowers/plans/2026-07-20-split-cycle-lightweight-loader.md` out of task commits.

---

### Task 1: Merge all per-person summaries after chunked triangulation

**Files:**
- Modify: `triangulation/sam3d_from_split_cycle.py`
- Create: `tests/test_triangulated_dataset_validation.py`

**Interfaces:**
- Consumes: `<output_root>/person_<id>/summary.json` files.
- Produces: `collect_person_summaries(output_root: Path) -> list[dict[str, Any]]`, sorted numerically by person ID.

- [ ] **Step 1: Write the failing summary-collection test**

Create the test helpers and test:

```python
import json

from triangulation.sam3d_from_split_cycle import collect_person_summaries


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collect_person_summaries_reads_all_people_in_numeric_order(tmp_path):
    _write_json(tmp_path / "person_10" / "summary.json", {"person_id": "10"})
    _write_json(tmp_path / "person_2" / "summary.json", {"person_id": "2"})
    (tmp_path / "_camera").mkdir()

    summaries = collect_person_summaries(tmp_path)

    assert [item["person_id"] for item in summaries] == ["2", "10"]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_triangulated_dataset_validation.py::test_collect_person_summaries_reads_all_people_in_numeric_order -q
```

Expected: collection fails with an import error because `collect_person_summaries` does not exist.

- [ ] **Step 3: Implement numeric summary collection**

Add to `triangulation/sam3d_from_split_cycle.py`:

```python
def collect_person_summaries(output_root: Path) -> List[Dict[str, Any]]:
    person_dirs = sorted(
        (path for path in output_root.glob("person_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("person_")),
    )
    summaries = []
    for person_dir in person_dirs:
        summary_path = person_dir / "summary.json"
        if summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return summaries
```

In `main()`, replace the current-batch `summaries` in the root summary with:

```python
    all_summaries = collect_person_summaries(output_root)
```

and write:

```python
            "num_persons": len(all_summaries),
            "persons": all_summaries,
```

- [ ] **Step 4: Run the test and focused regressions**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_triangulated_dataset_validation.py::test_collect_person_summaries_reads_all_people_in_numeric_order tests/test_sam3d_triangulation.py -q
```

Expected: `5 passed`.

---

### Task 2: Add strict triangulated-dataset validation

**Files:**
- Create: `triangulation/tools/validate_sam3d_triangulated.py`
- Modify: `tests/test_triangulated_dataset_validation.py`
- Modify: `triangulation/tools/README.md`

**Interfaces:**
- Consumes: split-cycle alignment JSON files and triangulated per-cycle summaries, frame JSON, and NPZ sequences.
- Produces: `validate_dataset(split_root: Path, output_root: Path, warning_threshold_px: float, excluded_person_ids: set[str]) -> dict[str, Any]` and a CLI-written JSON report.

- [ ] **Step 1: Write failing tests for a complete dataset and malformed sequence**

Add tests that construct two minimal split records and triangulated trees. The complete case must assert `passed is True`, two validated people/cycles, person 119 status `excluded_low_quality`, and one aggregate cycle. The malformed case must use `(1, 69, 3)` data containing `NaN`, `processed_frames=2`, `missing_pairs=1`, and no frame JSON, then assert error codes include:

```python
{
    "invalid_sequence_shape",
    "non_finite_sequence",
    "processed_frames_mismatch",
    "missing_pairs",
    "frame_json_count_mismatch",
}
```

- [ ] **Step 2: Run the validator tests and verify RED**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_triangulated_dataset_validation.py -q
```

Expected: import failure because `triangulation.tools.validate_sam3d_triangulated` does not exist.

- [ ] **Step 3: Implement the validator**

Create `triangulation/tools/validate_sam3d_triangulated.py` with these public behaviors:

```python
def validate_dataset(
    split_root: Path,
    output_root: Path,
    warning_threshold_px: float = 60.0,
    excluded_person_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate every split-cycle expectation against triangulated outputs."""
```

For every `person_*/alignment_record_*.json`, compare expected and actual cycle indices, load each `joints_3d_sequence.npz`, and append errors as dictionaries with at least `code`, `person_id`, and `cycle_index`. Validate exact `(T, 70, 3)` shape, finite values, `processed_frames`, `missing_pairs`, and frame JSON count. Append a `high_reprojection_error` warning when either numeric per-view error exceeds the threshold. Build per-person rows with `quality_status` equal to `excluded_low_quality`, `warning`, `error`, or `ok`; aggregate face/side errors only for non-excluded people.

The CLI must accept:

```text
--split-root
--output-root
--report
--warning-threshold-px
--exclude-person [ID ...]
```

It must always write JSON, print the report counts, and return exit code 1 when `errors` is non-empty.

- [ ] **Step 4: Run the validator tests and focused regressions**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_triangulated_dataset_validation.py tests/test_sam3d_triangulation.py tests/test_compare_fused_triangulated.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Document the validation command**

Add to `triangulation/tools/README.md`:

```bash
conda run -n gymnastic python triangulation/tools/validate_sam3d_triangulated.py \
  --exclude-person 119
```

Document that structural errors return nonzero, errors over 60 px are warnings, and person 119 remains in completeness checks but is excluded from aggregate metrics.

---

### Task 3: Preflight and person-69 production smoke

**Files:**
- Runtime output: `/home/data/xchen/gymnastics/sam3d_triangulated/person/person_69`
- Runtime manifest: `/tmp/gymnastics-triangulation-old-manifest.tsv`
- Runtime backup: a new file under `/tmp/gymnastics-triangulation-summary-backup.*.json`

**Interfaces:**
- Consumes: 137 split records, both calibration NPZs, and SAM3D face/side frame NPZs.
- Produces: an old-output protection manifest and complete person-69 outputs.

- [ ] **Step 1: Record old data metadata and back up the root summary**

Use `find` restricted to explicit `person_1` through `person_68` paths to write relative path, size, and nanosecond mtime into `/tmp/gymnastics-triangulation-old-manifest.tsv`. Copy the root `summary.json` to a uniquely named `/tmp` backup.

- [ ] **Step 2: Run person 69 without visualization**

Run:

```bash
conda run --no-capture-output -n gymnastic python -m triangulation.sam3d_from_split_cycle --person 69 --no-vis --no-video
```

Expected: exit code 0 and `[DONE]` for the configured output root.

- [ ] **Step 3: Validate the partial dataset specifically for person 69**

Run a temporary split-root containing only a link/copy of person 69's alignment record through the validator, excluding person 119. Expected: one person, the split-record cycle count, no integrity errors.

---

### Task 4: Triangulate remaining new people in bounded batches

**Files:**
- Runtime outputs: `/home/data/xchen/gymnastics/sam3d_triangulated/person/person_70` through `person_138`, excluding `person_135`

**Interfaces:**
- Consumes: the remaining 68 split-cycle records and SAM3D frame NPZ pairs.
- Produces: 464 additional triangulated cycles and refreshed merged root summaries.

- [ ] **Step 1: Run batches sequentially**

Run each group with `--no-vis --no-video`:

```text
70–79
80–89
90–99
100–109
110–119
120–129
130–134, 136–138
```

After every command, assert every person in that group has a person summary and the expected number of cycle NPZ files before starting the next group.

- [ ] **Step 2: Confirm the merged root summary after the last batch**

Assert `num_persons == 137`, `len(persons) == 137`, and sorted IDs equal `1..134,136,137,138`.

---

### Task 5: Full validation, reporting, and old-output protection check

**Files:**
- Create runtime report: `logs/analysis/triangulated_results/validation_summary.json`
- Refresh runtime reports under: `logs/analysis/triangulated_results/`

**Interfaces:**
- Consumes: all 137 person outputs, 923 expected cycles, and the preflight old-output manifest.
- Produces: strict validation JSON, reporting CSV/Markdown, and an unchanged-old-data comparison.

- [ ] **Step 1: Run strict full-dataset validation**

Run:

```bash
conda run -n gymnastic python triangulation/tools/validate_sam3d_triangulated.py --exclude-person 119
```

Expected: exit code 0, 137 validated persons, 923 validated cycles, no errors, and person 119 listed as excluded from aggregate metrics.

- [ ] **Step 2: Generate the existing consolidated report**

Run:

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

Expected: refreshed Markdown and two CSV files under `logs/analysis/triangulated_results`.

- [ ] **Step 3: Verify persons 1–68 remain unchanged**

Regenerate the exact path/size/mtime manifest for persons 1–68 and compare it byte-for-byte with `/tmp/gymnastics-triangulation-old-manifest.tsv`. Expected: no differences.

- [ ] **Step 4: Run final code verification**

Run:

```bash
conda run -n gymnastic python -m pytest tests/test_triangulated_dataset_validation.py tests/test_sam3d_triangulation.py tests/test_compare_fused_triangulated.py -q
git diff --check -- triangulation/sam3d_from_split_cycle.py triangulation/tools/validate_sam3d_triangulated.py triangulation/tools/README.md tests/test_triangulated_dataset_validation.py docs/superpowers/specs/2026-07-20-new-person-triangulation-design.md docs/superpowers/plans/2026-07-20-new-person-triangulation.md
```

Expected: all tests pass and `git diff --check` exits 0.
