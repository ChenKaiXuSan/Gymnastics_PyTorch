# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from pathlib import Path
import hydra
import numpy as np
import cv2
from tqdm import tqdm


def video_frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 顺时针旋转90度
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        yield frame
    cap.release()


@hydra.main(config_path="../../configs/", config_name="inference")
def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)

    im_list = list(Path(args.data.path).rglob("*.MOV"))

    for video_name in im_list:

        npz_out_path = (
            Path(args.data.npz_path)
            / "/".join(video_name.parts[3:-1])
            / (video_name.stem + ".npz")
        )
        frame_out_path = (
            Path(args.data.frame_path)
            / "/".join(video_name.parts[3:-1])
            / video_name.stem
        )

        res_out_path = (
            Path(args.data.res_path)
            / "/".join(video_name.parts[3:-1])
            / video_name.stem
        )

        if not npz_out_path.parent.exists():
            npz_out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frame_out_path.exists():
            frame_out_path.mkdir(parents=True, exist_ok=True)
        if not res_out_path.exists():
            res_out_path.mkdir(parents=True, exist_ok=True)

        print("Processing {}".format(video_name))

        boxes = []
        segments = []
        keypoints = []

        for frame_i, im in tqdm(
            enumerate(video_frame_generator(video_name)),
            total=int(cv2.VideoCapture(str(video_name)).get(cv2.CAP_PROP_FRAME_COUNT)),
            leave=False,
        ):

            # save frame
            frame_out = frame_out_path / ("{:05d}.jpg".format(frame_i))
            cv2.imwrite(str(frame_out), im)

            outputs = predictor(im)["instances"].to(f"cuda:{args.gpu}")

            outputs = outputs.to("cpu")

            bbox_tensor, kps = [], []

            # Check for valid bounding boxes
            # if outputs.has("pred_boxes") and len(outputs.pred_boxes) > 0:
            #     # Extract bounding boxes and scores
            #     bbox_tensor = outputs.pred_boxes.tensor.numpy()
            #     scores = outputs.scores.numpy()

            #     # Get highest-scoring bbox
            #     best_idx = np.argmax(scores)  # * Assuming we want the best bbox
            #     best_bbox = bbox_tensor[best_idx]
            #     best_score = scores[best_idx]

            #     # Combine bbox and score
            #     bbox_tensor = np.expand_dims(np.append(best_bbox, best_score), axis=0)

            #     if outputs.has("pred_keypoints"):
            #         kps_raw = outputs.pred_keypoints.numpy()  # shape: (N, K, 3)
            #         if len(kps_raw) > best_idx:
            #             kps_xy = kps_raw[best_idx, :, :2]  # (K, 2)
            #             kps_prob = kps_raw[best_idx, :, 2:3]  # (K, 1)
            #             kps_logit = np.zeros_like(kps_prob)  # Dummy logits

            #             # Combine into (K, 4): x, y, logit, prob
            #             kps_combined = np.concatenate(
            #                 (kps_xy, kps_logit, kps_prob), axis=1
            #             )

            #             # Final shape: (1, 4, K)
            #             kps = np.expand_dims(kps_combined.T, axis=0)
            #         else:
            #             kps = []
            # else:
            #     # No valid bbox —> empty list fallback
            #     bbox_tensor = []
            #     kps = []

            has_bbox = False
            if outputs.has('pred_boxes'):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
            if has_bbox:
                kps = outputs.pred_keypoints.numpy()
                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]
                kps_logit = np.zeros_like(kps_prob) # Dummy
                kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                kps = kps.transpose(0, 2, 1)
            else:
                kps = []
                bbox_tensor = []

            # print("boxes type:", type(bbox_tensor), "len:", len(bbox_tensor))
            # print("keypoints shape:", np.shape(kps))

            boxes.append(bbox_tensor)
            segments.append(None)
            keypoints.append(kps)

            # Visualize the results (optional)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=1.0,
            )
            out = v.draw_instance_predictions(outputs)

            # Save the visualization if needed
            cv2.imwrite(
                str(res_out_path / ("{:05d}.jpg".format(frame_i))),
                out.get_image()[:, :, ::-1],
            )

        # Video resolution
        metadata = {
            "w": im.shape[1],
            "h": im.shape[0],
        }

        np.savez_compressed(
            npz_out_path,
            boxes=boxes,
            segments=segments,
            keypoints=keypoints,
            metadata=metadata,
        )


if __name__ == "__main__":
    setup_logger()
    main()
