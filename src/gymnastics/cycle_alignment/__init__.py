"""Stage 2 - cycle alignment: face/side temporal offset and movement-cycle segmentation.

Estimates the side-to-face frame offset, splits each recording into complete
cycles with turn-around middles and writes the records that the pseudo-GT
stage and the fusion data modules read (``local/runs/split_cycle`` for the
private data, ``local/runs/cycle_records`` for the public datasets).
CLI: ``gymnastics align``, ``gymnastics align cycles {private,freeman,unity,index}``.
"""
