"""Windowed, batched SAM3D-Body inference for the FreeMan benchmark.

The per-frame path (`SAM3DBodyEstimator.process_one_image`) launches the
detector, the FoV estimator, and the pose model once per frame and clears the
CUDA cache between frames, which leaves the GPU mostly idle. This module keeps
the upstream full-mode semantics (body decoder + hand decoders + wrist fusion
+ keypoint prompting) while batching whole frame windows through every model:

- frames are decoded into windows and treated as the "person" dimension of a
  single pseudo-image batch, which the upstream model already flattens;
- the two upstream hand `prepare_batch` calls are redirected through a scoped
  patch so each hand crop is taken from its own source frame (the flipped
  call is recognised by the negative column stride upstream creates);
- ViTDet runs in sub-batches over the window instead of once per frame;
- camera intrinsics are estimated once per session view (median of MoGe over
  the first probe frames) instead of once per frame — a deliberate,
  config-recorded deviation: the camera is fixed within a view, so per-frame
  re-estimation only adds jitter and one ViT-L forward per frame;
- `torch.cuda.empty_cache()` runs once per view, not per frame.

Frames within one session view share height/width, which the upstream wrist
logic relies on (scalar `width` flips); this is asserted per window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np
import torch

_DETECT_IMAGE_SIZE = 1024
_DETECT_CATEGORY_ID = 0
_DETECT_BBOX_THRESHOLD = 0.5


def _prepare_multi_image_batch(
    images: Sequence[np.ndarray],
    transform: Any,
    boxes: np.ndarray,
    cam_int: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Mirror upstream ``prepare_batch`` with one source image per crop."""
    from sam_3d_body.data.utils.prepare_batch import NoCollate
    from torch.utils.data import default_collate

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(images) != boxes.shape[0]:
        raise ValueError(
            f"batch needs one image per box, got {len(images)} images "
            f"and {boxes.shape[0]} boxes"
        )
    height, width = images[0].shape[:2]
    data_list = []
    for image, box in zip(images, boxes):
        if image.shape[:2] != (height, width):
            raise ValueError("all frames in a batch window must share a resolution")
        data_info = dict(img=image)
        data_info["bbox"] = box
        data_info["bbox_format"] = "xyxy"
        data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
        data_info["mask_score"] = np.array(0.0, dtype=np.float32)
        data_list.append(transform(data_info))

    batch = default_collate(data_list)
    max_num_person = batch["img"].shape[0]
    for key in (
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ):
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(images[0])]
    return batch


@contextmanager
def _patched_hand_prepare(frames: Sequence[np.ndarray]):
    """Route upstream hand-crop batches to each crop's own source frame.

    ``run_inference`` calls ``prepare_batch(flipped_img, ...)`` for the left
    hands and ``prepare_batch(img, ...)`` for the right hands, both with one
    box per person. Person index equals frame index here, so each hand crop
    must come from its own frame (flipped for the left-hand call, which is
    identified by the negative column stride of ``img[:, ::-1]``).
    """
    import sam_3d_body.models.meta_arch.sam3d_body as meta_arch

    original = meta_arch.prepare_batch

    def patched(img, transform, boxes, masks=None, masks_score=None, cam_int=None):
        if masks is not None or masks_score is not None:
            raise RuntimeError("batched hand preparation does not expect masks")
        flipped = img.strides[1] < 0
        sources = [
            np.ascontiguousarray(frame[:, ::-1]) if flipped else frame
            for frame in frames
        ]
        return _prepare_multi_image_batch(sources, transform, boxes, cam_int=cam_int)

    meta_arch.prepare_batch = patched
    try:
        yield
    finally:
        meta_arch.prepare_batch = original


class BatchedViewRunner:
    """Batched full-mode SAM3D inference over frame windows of one view."""

    def __init__(self, estimator: Any, options: Mapping[str, Any]):
        self.estimator = estimator
        self.window = int(options.get("window", 32))
        self.detect_batch = int(options.get("detect_batch", 8))
        self.fov_probe_frames = int(options.get("fov_probe_frames", 5))
        if self.window < 1 or self.detect_batch < 1 or self.fov_probe_frames < 1:
            raise ValueError("batched inference options must be positive")
        if estimator.detector is None:
            raise RuntimeError("batched inference requires the human detector")

    def detect(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Batched ViTDet mirroring ``run_detectron2_vitdet`` per frame."""
        import detectron2.data.transforms as T

        detector = self.estimator.detector.detector
        resize = T.ResizeShortestEdge(
            short_edge_length=_DETECT_IMAGE_SIZE,
            max_size=_DETECT_IMAGE_SIZE,
        )
        inputs = []
        for frame in frames:
            # Upstream converts RGB back to BGR before detection.
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            transformed = resize(T.AugInput(bgr)).apply_image(bgr)
            inputs.append(
                {
                    "image": torch.as_tensor(
                        transformed.astype("float32").transpose(2, 0, 1)
                    ),
                    "height": height,
                    "width": width,
                }
            )
        boxes_per_frame: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(inputs), self.detect_batch):
                chunk = inputs[start : start + self.detect_batch]
                for output in detector(chunk):
                    instances = output["instances"]
                    valid = (instances.pred_classes == _DETECT_CATEGORY_ID) & (
                        instances.scores > _DETECT_BBOX_THRESHOLD
                    )
                    boxes = instances.pred_boxes.tensor[valid].cpu().numpy()
                    if len(boxes):
                        order = np.lexsort(
                            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
                        )
                        boxes = boxes[order]
                    boxes_per_frame.append(boxes.reshape(-1, 4))
        return boxes_per_frame

    def estimate_cam_int(self, frames: Sequence[np.ndarray]) -> torch.Tensor | None:
        """Median MoGe intrinsics over probe frames; None keeps the default."""
        fov = self.estimator.fov_estimator
        if fov is None:
            return None
        probes = frames[: self.fov_probe_frames]
        estimates = []
        for frame in probes:
            value = fov.get_cam_intrinsics(frame)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            estimates.append(np.asarray(value, dtype=np.float64).reshape(3, 3))
        median = np.median(np.stack(estimates, axis=0), axis=0)
        return torch.tensor(median[None], dtype=torch.float32)

    @staticmethod
    def _largest_box(boxes: np.ndarray) -> np.ndarray | None:
        """Replicate `_best_person`: keep the largest-area detection."""
        if boxes.size == 0:
            return None
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0, boxes[:, 3] - boxes[:, 1]
        )
        if not np.isfinite(areas.max()):
            return None
        return boxes[int(np.argmax(areas))]

    @torch.no_grad()
    def process_window(
        self,
        frames: Sequence[np.ndarray],
        cam_int: torch.Tensor | None,
    ) -> list[Mapping[str, Any] | None]:
        """Run full-mode inference over one window; one result per frame."""
        boxes_per_frame = self.detect(frames)
        selected: list[tuple[int, np.ndarray, np.ndarray]] = []
        for index, boxes in enumerate(boxes_per_frame):
            box = self._largest_box(boxes)
            if box is not None:
                selected.append((index, frames[index], box))
        results: list[Mapping[str, Any] | None] = [None] * len(frames)
        if not selected:
            return results

        batch_frames = [item[1] for item in selected]
        batch_boxes = np.stack([item[2] for item in selected], axis=0)
        model = self.estimator.model
        batch = _prepare_multi_image_batch(
            batch_frames,
            self.estimator.transform,
            batch_boxes,
            cam_int=cam_int,
        )
        from sam_3d_body.utils import recursive_to

        batch = recursive_to(batch, "cuda")
        model._initialize_batch(batch)
        with _patched_hand_prepare(batch_frames):
            outputs = model.run_inference(
                batch_frames[0],
                batch,
                inference_type="full",
                transform_hand=self.estimator.transform_hand,
                thresh_wrist_angle=self.estimator.thresh_wrist_angle,
            )
        pose_output = outputs[0]
        mhr = pose_output["mhr"]
        keypoints3d = mhr["pred_keypoints_3d"].detach().float().cpu().numpy()
        keypoints2d = mhr["pred_keypoints_2d"].detach().float().cpu().numpy()
        bboxes = batch["bbox"][0].detach().float().cpu().numpy()
        for row, (frame_index, _, _) in enumerate(selected):
            results[frame_index] = {
                "bbox": bboxes[row],
                "pred_keypoints_3d": keypoints3d[row],
                "pred_keypoints_2d": keypoints2d[row],
            }
        return results


def stream_view_batched(
    runner: BatchedViewRunner,
    session: Any,
    view_id: str,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Windowed drop-in for `_stream_view` with identical outputs."""
    points3d: list[np.ndarray] = []
    points2d: list[np.ndarray] = []
    valid3d: list[np.ndarray] = []
    valid2d: list[np.ndarray] = []
    failed: list[int] = []
    cam_int: torch.Tensor | None = None
    cam_int_ready = False

    def flush(window_frames: list[np.ndarray], window_ids: list[int]) -> None:
        nonlocal cam_int, cam_int_ready
        if not window_frames:
            return
        if not cam_int_ready:
            cam_int = runner.estimate_cam_int(window_frames)
            cam_int_ready = True
        for frame_index, selected in zip(
            window_ids,
            runner.process_window(window_frames, cam_int),
        ):
            if selected is None:
                xyz = np.zeros((70, 3), dtype=np.float32)
                xy = np.zeros((70, 2), dtype=np.float32)
                xyz_valid = np.zeros(70, dtype=bool)
                xy_valid = np.zeros(70, dtype=bool)
                failed.append(frame_index)
            else:
                xyz = np.asarray(
                    selected["pred_keypoints_3d"], dtype=np.float32
                ).reshape(70, 3)
                xy = np.asarray(
                    selected["pred_keypoints_2d"], dtype=np.float32
                ).reshape(70, 2)
                xyz_valid = np.isfinite(xyz).all(axis=-1) & np.any(xyz != 0, axis=-1)
                xy_valid = np.isfinite(xy).all(axis=-1)
                xyz = np.where(xyz_valid[:, None], xyz, 0)
                xy = np.where(xy_valid[:, None], xy, 0)
            points3d.append(xyz)
            points2d.append(xy)
            valid3d.append(xyz_valid)
            valid2d.append(xy_valid)

    # Decode on a producer thread so the CPU reads the next window while the
    # GPU processes the current one. This changes throughput only: frames,
    # ordering, window boundaries, and failure semantics are identical to the
    # sequential loop, so cached outputs stay bit-comparable.
    import queue
    import threading

    wanted = {int(frame_id) for frame_id in frame_ids}
    last_wanted = int(frame_ids[-1])
    frame_queue: queue.Queue = queue.Queue(maxsize=2 * runner.window)
    stop_decoding = threading.Event()

    def decode() -> None:
        capture = cv2.VideoCapture(str(session.video_paths[view_id]))
        try:
            if not capture.isOpened():
                raise RuntimeError(
                    f"cannot open FreeMan video {session.video_paths[view_id]}"
                )
            frame_index = 0
            while frame_index <= last_wanted and not stop_decoding.is_set():
                success, frame_bgr = capture.read()
                if not success:
                    raise RuntimeError(
                        f"internal video decode failure at frame {frame_index}: "
                        f"{session.video_paths[view_id]}"
                    )
                if frame_index in wanted:
                    frame_queue.put(
                        (frame_index, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    )
                frame_index += 1
            frame_queue.put(None)
        except BaseException as error:  # propagated to the consumer
            frame_queue.put(error)
        finally:
            capture.release()

    decoder = threading.Thread(target=decode, name="freeman-decode", daemon=True)
    decoder.start()
    window_frames: list[np.ndarray] = []
    window_ids: list[int] = []
    try:
        while True:
            item = frame_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            frame_index, frame_rgb = item
            window_frames.append(frame_rgb)
            window_ids.append(frame_index)
            if len(window_frames) >= runner.window:
                flush(window_frames, window_ids)
                window_frames, window_ids = [], []
        flush(window_frames, window_ids)
    finally:
        stop_decoding.set()
        while decoder.is_alive():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            decoder.join(timeout=0.1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        np.stack(points3d, axis=0),
        np.stack(points2d, axis=0),
        np.stack(valid3d, axis=0),
        np.stack(valid2d, axis=0),
        failed,
    )
