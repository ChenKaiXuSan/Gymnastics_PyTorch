# Triangulation Tools

Utility scripts for inspecting and reporting triangulated SAM3D outputs.

## Result Report

Generate the consolidated Markdown report plus CSV details:

```bash
conda run -n gymnastic python -m gymnastics.analysis.reports.generate_results_report
```

Default source:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person
```

Default outputs:

```text
local/runs/analysis/triangulated_results/triangulated_results_report.md
local/runs/analysis/triangulated_results/triangulated_cycle_details.csv
local/runs/analysis/triangulated_results/triangulated_person_summary.csv
```

The cycle CSV records frame ranges, processed frame counts, missing pair counts,
joint counts, reprojection errors, generated sequence paths, visualization video
paths, and source cycle directories. The person CSV aggregates cycle counts,
frame counts, missing pairs, and reprojection error statistics per person.

## Strict Dataset Validation

Validate the triangulated tree against every split-cycle record:

```bash
conda run -n gymnastic python -m gymnastics.analysis.reports.validate_sam3d_triangulated \
  --exclude-person 119
```

The command checks cycle inventory, `(T, 70, 3)` sequence shapes, finite
coordinates, processed-frame counts, missing frame pairs, per-frame JSON
counts, and the merged root summary. Structural errors return a nonzero exit
code. A per-view mean reprojection error over 60 px is recorded as a warning.
Person 119 remains subject to every completeness check but is excluded from
aggregate quality metrics because of its known low two-view overlap.

The machine-readable report is written to:

```text
local/runs/analysis/triangulated_results/validation_summary.json
```
