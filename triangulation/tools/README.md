# Triangulation Tools

Utility scripts for inspecting and reporting triangulated SAM3D outputs.

## Result Report

Generate the consolidated Markdown report plus CSV details:

```bash
conda run -n gymnastic python triangulation/tools/generate_results_report.py
```

Default source:

```text
/home/data/xchen/gymnastics/sam3d_triangulated/person
```

Default outputs:

```text
logs/analysis/triangulated_results/triangulated_results_report.md
logs/analysis/triangulated_results/triangulated_cycle_details.csv
logs/analysis/triangulated_results/triangulated_person_summary.csv
```

The cycle CSV records frame ranges, processed frame counts, missing pair counts,
joint counts, reprojection errors, generated sequence paths, visualization video
paths, and source cycle directories. The person CSV aggregates cycle counts,
frame counts, missing pairs, and reprojection error statistics per person.
