
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from glob import glob
import os
import sys

import argparse
from data_utils import suggest_metadata

output_prefix_2d = "data_2d_custom_"


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
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(
                np.full(4, np.nan, dtype=np.float32)
            )  # 4 bounding box coordinates
            results_kp.append(
                np.full((17, 4), np.nan, dtype=np.float32)
            )  # 17 COCO keypoints
            continue

        if i == 0:
            # First frame, no previous bbox/keypoints to compare
            results_bb.append(bb[i][1][0, :4])
            results_kp.append(kp[i][1][0].T)
        elif len(bb[i][1]) == 1 and len(kp[i][1]) == 1:
            # If there is only one detection, we can use it directly
            results_bb.append(bb[i][1][0, :4])
            results_kp.append(kp[i][1][0].T)
        elif len(bb[i][1]) >= 2 and len(kp[i][1]) >= 2:
            # If there are multiple detections, we need to find the best match
            # based on the highest confidence score

            # find the short distance match between the current and previous frame
            curr_xc = (bb[i][1][:, 0] + bb[i][1][:, 2]) / 2
            curr_yc = (bb[i][1][:, 1] + bb[i][1][:, 3]) / 2

            prev_xc = (results_bb[i - 1][0] + results_bb[i - 1][2]) / 2
            prev_yc = (results_bb[i - 1][1] + results_bb[i - 1][3]) / 2

            distance_list = []
            for j in range(len(bb[i][1])):
                curr_x, curr_y = curr_xc[j], curr_yc[j]
                prev_x, prev_y = prev_xc, prev_yc
                distance = np.linalg.norm(np.array([curr_x - prev_x, curr_y - prev_y]))
                distance_list.append(distance)

            best_match = np.argmin(distance_list)
            print(best_match)

            best_bb = bb[i][1][best_match, :4]
            best_kp = kp[i][1][best_match].T.copy()
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
    # print("{} frames were interpolated".format(np.sum(~mask)))
    print("----------")

    return [
        {
            "start_frame": 0,  # Inclusive
            "end_frame": len(kp),  # Exclusive
            "bounding_boxes": bb,
            "keypoints": kp,
        }
    ], metadata


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "data":
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(description="Custom dataset creator")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="",
        metavar="PATH",
        help="detections directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        metavar="PATH",
        help="output suffix for 2D detections",
    )
    args = parser.parse_args()

    if not args.input:
        print("Please specify the input directory")
        exit(0)

    if not args.output:
        print("Please specify an output suffix (e.g. detectron_pt_coco)")
        exit(0)

    print("Parsing 2D detections from", args.input)

    metadata = suggest_metadata("coco")
    metadata["video_metadata"] = {}

    output = {}
    file_list = glob(args.input + "/*.npz")
    for f in file_list:
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        data, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]["custom"] = [data[0]["keypoints"].astype("float32")]
        metadata["video_metadata"][canonical_name] = video_metadata

    print("Saving...")
    np.savez_compressed(
        output_prefix_2d + args.output, positions_2d=output, metadata=metadata
    )
    print("Done.")
