# Neurocomputing Submission Checklist

## Required Empirical Work

- [x] Freeze 96/27/14 private train/validation/test populations.
- [x] Use the 14-person held-out set as the primary learned comparison.
- [x] Run A0--A9 paired tests with confidence intervals and Holm correction.
- [x] Run fixed-corruption and temporal-offset sensitivity analyses.
- [x] Complete person-disjoint OOF cohort inference and centered mixed models.
- [x] Add limited independent synthetic validation against Unity native 3D.
- [ ] Train the full private learned comparison with at least three seeds.
- [ ] Add diverse real public-benchmark or independent-capture validation.

## Author and Governance

- [ ] Complete the CCS institutional postal address.
- [ ] Confirm the corresponding-author metadata.
- [ ] Insert the verified ethics approval/exemption and consent statement.
- [ ] Decide whether code and anonymized keypoints can be deposited or shared under controlled access.
- [ ] Add a biography of at most 100 words and a passport-style author photograph if required at submission.
- [ ] Confirm funding and competing-interest statements.

## Files and Compliance

- [ ] Recheck the current Neurocomputing Guide for Authors.
- [ ] Keep 3--5 highlights, each at most 85 characters.
- [ ] Keep the abstract at most 250 words and use 1--7 keywords.
- [ ] Run `make -C paper/neurocomputing` from a checkout with the source metrics CSV.
- [ ] Review the final PDF for overfull tables, figure legibility, and anonymization requirements.
- [ ] Run a final citation-to-claim and data-leakage audit.
