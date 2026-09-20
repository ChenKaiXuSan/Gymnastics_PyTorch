"""SportsPose public benchmark (Ingwersen et al., 2023).

26 subjects (22 indoors, 4 outdoors), 7 calibrated cameras at 90 fps and a
markerless multi-view 3D reference (COCO17, metres). Every clip is one
3-second trial of one action (jump, soccer, tennis, throw_baseball, volley),
repeated about five times per subject and action.

``dataset`` enumerates clips, calibrations and references and selects the
face-like / side-like view pair; ``sam3d`` runs SAM3D-Body on the two
selected views and caches the per-view predictions; ``cli`` is the
``python -m fusion benchmark-sportspose`` entry point. Cycle records
(one clip = one cycle) are written by ``python -m cycle_alignment cycles
sportspose`` and the training adapter is ``fusion.data.sportspose``.
"""
