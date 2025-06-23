# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import hydra

from pathlib import Path
from videopose3d.data_utils import suggest_metadata


def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print("Processing {}".format(filename))
    data = np.load(filename, encoding="latin1", allow_pickle=True)
    bb = data["boxes"]
    kp = data["keypoints"]
    metadata = data["metadata"].item()
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i]) == 0 or len(kp[i]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(
                np.full(4, np.nan, dtype=np.float32)
            )  # 4 bounding box coordinates
            results_kp.append(
                np.full((17, 4), np.nan, dtype=np.float32)
            )  # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][:, 4])
        best_bb = bb[i][best_match, :4]
        best_kp = kp[i][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)

    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2]  # Extract (x, y)

    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

    print("{} total frames processed".format(len(bb)))
    print("{} frames were interpolated".format(np.sum(~mask)))
    print("----------")

    return [
        {
            "start_frame": 0,  # Inclusive
            "end_frame": len(kp),  # Exclusive
            "bounding_boxes": bb,
            "keypoints": kp,
        }
    ], metadata


@hydra.main(config_path="../configs", config_name="inference")
def main(args):

    npz_path = Path(args.data.npz_path)
    filter_npz_path = Path(args.data.filter_npz_path)

    if not filter_npz_path.exists():
        filter_npz_path.mkdir(parents=True, exist_ok=True)

    print("Parsing 2D detections from", args.data.npz_path)

    metadata = suggest_metadata("coco")
    metadata["video_metadata"] = {}

    output = {}
    file_list = list(npz_path.rglob("*.npz"))

    for f in file_list:
        canonical_name = str(f)
        data, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]["custom"] = [data[0]["keypoints"].astype("float32")]
        metadata["video_metadata"][canonical_name] = video_metadata

    print("Saving...")
    np.savez_compressed(
        args.data.filter_npz_path, positions_2d=output, metadata=metadata
    )
    print("Done.")


if __name__ == "__main__":

    main()
