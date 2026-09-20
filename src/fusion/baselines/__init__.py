"""Deterministic fusion comparison matrix and classical baselines.

Label-free reference methods every learned model is compared against: single
views, naive and coordinate-normalised averaging, similarity-alignment
variants, camera-assisted variants, Kalman / reliability-weighted / Butterworth
fusion and the zero-shot SmoothNet refiner. Shared by the private data and
both public benchmarks. CLI: ``python -m fusion deterministic``.

Layout:

* ``methods``            fusion building blocks and the method-name tables
* ``classical_baselines`` Kalman / RTS / jitter-weighted / Butterworth / SmoothNet
* ``data``               SAM3D, split-cycle, extrinsics and triangulated loaders; outputs
* ``evaluation``         alignment and metrics against the triangulated reference
* ``experiment_matrix``  per-person driver and the command line
"""
