"""Stage 2 - cycle alignment: face/side temporal offset and movement-cycle segmentation.

Estimates the side-to-face frame offset, splits each recording into complete
cycles with turn-around middles and writes the records that the pseudo-GT
stage and the fusion data modules read (``local/runs/split_cycle`` for the
private data, ``local/runs/cycle_records`` for the public datasets).
CLI: ``python -m cycle_alignment align``, ``python -m cycle_alignment cycles {private,freeman,unity,index}``.

Layout:

* ``features``       body-frame right-hand angle ``theta`` (the alignment signal)
* ``offset``         keypoint-DTW and audio offset estimation, common timeline
* ``segmentation``   cycle detection on the fused trajectory, timeline -> video frames
* ``cycles`` / ``cycle_records`` / ``annotate_cycles``  turn-around middles and cycle records
* ``main``           per-person driver and the ``align`` command line
"""
