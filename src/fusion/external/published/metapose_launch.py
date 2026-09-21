"""Launcher for the vendored ``metapose.train_metapose`` with two runtime patches (TF env).

* ``inference_time_optimization.procrustes`` runs ``tf.linalg.svd`` on the
  device of its inputs; inside the vectorised ``pmpjpe`` metric cuSOLVER's
  ``gesvd`` aborts the whole training step with ``info = 2`` when a 3x3
  cross-covariance is (near-)singular, e.g. for a collapsed early prediction,
  whereas the CPU kernel simply returns a solution. The metric is monitoring
  only (checkpoint selection), so the SVD is pinned to the CPU.
* The script keeps the best weights of a stage at the literal path
  ``/tmp/best-model`` (``ModelCheckpoint`` and ``WriteStageMetrics``), which
  concurrent runs on one node would overwrite; ``$METAPOSE_BEST_MODEL``
  replaces it.

The vendored files stay unmodified; everything else is the authors' script
and flags.
"""

from __future__ import annotations

import os
import sys

import tensorflow as tf
from absl import app

from metapose import inference_time_optimization as inf_opt  # type: ignore
from metapose import train_metapose  # type: ignore

RELEASED_BEST_MODEL = "/tmp/best-model"
BEST_MODEL = os.environ.get("METAPOSE_BEST_MODEL", RELEASED_BEST_MODEL)


def procrustes_cpu_svd(a, b):
    """``inference_time_optimization.procrustes`` with the SVD on the CPU."""
    tf.debugging.assert_shapes([(a, ("joints", "dim")), (b, ("joints", "dim"))])
    a_m = tf.reduce_mean(a, axis=0)
    b_m = tf.reduce_mean(b, axis=0)
    a_c = a - a_m
    b_c = b - b_m
    cross = tf.tensordot(a_c, b_c, axes=(0, 0))
    with tf.device("/cpu:0"):
        _, u, v = tf.linalg.svd(cross)
    rotation = tf.tensordot(u, v, axes=(1, 1))
    return rotation, a_m, b_m


if os.environ.get("METAPOSE_PATCH_SVD", "1") != "0":
    inf_opt.procrustes = procrustes_cpu_svd


class _ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, *args, **kwargs):
        super().__init__(BEST_MODEL if filepath == RELEASED_BEST_MODEL else filepath, *args, **kwargs)


if BEST_MODEL != RELEASED_BEST_MODEL:
    tf.keras.callbacks.ModelCheckpoint = _ModelCheckpoint  # looked up as ``tfk.callbacks.ModelCheckpoint`` at call time


def _on_train_end(self, logs=None):
    """``WriteStageMetrics.on_train_end`` with the private best-model path."""
    if self.cur_best_val_score >= self.prev_best_val_score:
        print("no improvement %.3f (new) >= %.3f, not writing to the tensorboard stage stats" % (self.cur_best_val_score, self.prev_best_val_score))
        return
    print("improved (!) %.3f (new) < %.3f, writing tensorboard stage stats" % (self.cur_best_val_score, self.prev_best_val_score))
    self.model.load_weights(BEST_MODEL)
    for _, (writer_getter, dataset) in self.dataset_writers.items():
        with writer_getter().as_default(step=self.stage):
            metric_values = self.model.evaluate(dataset, verbose=0)
            for metric_name, value in zip(self.model.metrics_names, metric_values):
                tf.summary.scalar("stage_" + metric_name, value)


if BEST_MODEL != RELEASED_BEST_MODEL:
    train_metapose.WriteStageMetrics.on_train_end = _on_train_end

if __name__ == "__main__":
    app.run(train_metapose.main, argv=sys.argv)
