#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default fuse entry point.

The fuse package now runs the experiment matrix by default. The matrix rebuilds
the face/side temporal alignment from SAM3D outputs, runs fusion variants, and
evaluates them against the triangulated reference.
"""

from __future__ import annotations

from fuse.experiment_matrix import main


if __name__ == "__main__":
    main()
