import logging
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Set, Any, Dict, Optional, List

import torch

from prepare_dataset.model.depth_estimation import DepthEstimator
from prepare_dataset.model.optical_flow import OpticalFlow
from prepare_dataset.process.preprocess_yolo import PreprocessYOLO
from prepare_dataset.process.preprocess_d2 import PreprocessD2

logger = logging.getLogger(__name__)


def _normalize_tasks(task_field: Any) -> Set[str]:
    """
    支持:
      - "all" / "yolo" / "depth" / "optical_flow" / "detectron2"
      - 可迭代对象 (list/tuple/set)
      - None -> 空集合
    """
    if task_field is None:
        return set()
    if isinstance(task_field, str):
        t = task_field.strip().lower()
        if t == "all":
            return {"yolo", "depth", "optical_flow", "detectron2"}
        return {t}
    try:
        return {str(x).strip().lower() for x in task_field}
    except Exception:
        return set()


def _infer_hw(vframes: torch.Tensor) -> Tuple[int, int]:
    """
    从 (T,H,W,C) 或 (T,C,H,W) 推断 H,W。
    如形状异常，回退为最后两个维度。
    """
    if vframes.dim() != 4:
        raise ValueError(f"vframes must be 4D, got {tuple(vframes.shape)}")
    if vframes.shape[-1] in (1, 3):  # THWC
        return int(vframes.shape[1]), int(vframes.shape[2])
    if vframes.shape[1] in (1, 3):  # TCHW
        return int(vframes.shape[2]), int(vframes.shape[3])
    # fallback
    return int(vframes.shape[-2]), int(vframes.shape[-1])


class Preprocess:
    def __init__(self, config, person: str) -> None:
        super(Preprocess, self).__init__()

        self.tasks = _normalize_tasks(getattr(config, "task", None))
        logger.info(f"Preprocess tasks: {sorted(self.tasks)}")

        # 条件初始化，避免不必要的显存/时间开销
        self.depth_estimator: Optional[DepthEstimator] = (
            DepthEstimator(config, person) if "depth" in self.tasks else None
        )
        self.of_model: Optional[OpticalFlow] = (
            OpticalFlow(config, person) if "optical_flow" in self.tasks else None
        )
        self.yolo_model: Optional[PreprocessYOLO] = (
            PreprocessYOLO(config, person) if "yolo" in self.tasks else None
        )
        self.d2_model: Optional[PreprocessD2] = (
            PreprocessD2(config, person) if "detectron2" in self.tasks else None
        )

    def __call__(self, vframes: torch.Tensor, video_path: Path) -> Dict[str, Any]:
        """
        统一入口：对指定任务进行处理。
        返回字典结构（与原实现保持一致字段）。
        """
        device = vframes.device
        T = int(vframes.shape[0])
        H, W = _infer_hw(vframes)

        # 先准备占位张量，发生异常也能安全返回
        empty_depth = torch.empty((0, 1, H, W), dtype=torch.float32, device=device)
        # 注意：RAFT 输出 (F-1,2,H,W)，占位按 0 行对齐
        empty_of = torch.empty((0, 2, H, W), dtype=torch.float32, device=device)
        empty_bbox = torch.empty((0, 4), dtype=torch.float32, device=device)
        empty_mask = torch.empty((0, 1, H, W), dtype=torch.float32, device=device)
        empty_kpts = torch.empty((0, 17, 3), dtype=torch.float32, device=device)
        empty_kpts_score = torch.empty((0, 17), dtype=torch.float32, device=device)

        # 输出容器
        depth = empty_depth
        optical_flow = empty_of
        yolo_bbox = empty_bbox
        yolo_mask = empty_mask
        yolo_keypoints = empty_kpts
        yolo_keypoints_score = empty_kpts_score
        bbox_none_index: Sequence[int] = []

        d2_pose = empty_kpts
        d2_pose_score = empty_kpts_score
        d2_bboxes_xyxy = empty_bbox
        d2_none_index: Sequence[int] = []

        # 统一无梯度上下文
        with torch.inference_mode():
            # depth
            if self.depth_estimator is not None:
                try:
                    depth = self.depth_estimator(vframes, video_path)
                except Exception as e:
                    logger.exception(
                        "Depth estimation failed for %s: %s", video_path, e
                    )
                    depth = empty_depth

            # optical flow
            if self.of_model is not None:
                try:
                    optical_flow = self.of_model(vframes, video_path)
                except Exception as e:
                    logger.exception("Optical flow failed for %s: %s", video_path, e)
                    optical_flow = empty_of

            # yolo (bbox/mask/pose)
            if self.yolo_model is not None:
                try:
                    (
                        bbox_none_index,
                        yolo_bbox,
                        yolo_mask,
                        yolo_keypoints,
                        yolo_keypoints_score,
                    ) = self.yolo_model(vframes, video_path)
                except Exception as e:
                    logger.exception("YOLO preprocess failed for %s: %s", video_path, e)
                    bbox_none_index = []
                    yolo_bbox = empty_bbox
                    yolo_mask = empty_mask
                    yolo_keypoints = empty_kpts
                    yolo_keypoints_score = empty_kpts_score

            # detectron2
            if self.d2_model is not None:
                try:
                    d2_pose, d2_pose_score, d2_bboxes_xyxy, d2_none_index = (
                        self.d2_model(vframes, video_path)
                    )
                except Exception as e:
                    logger.exception(
                        "Detectron2 preprocess failed for %s: %s", video_path, e
                    )
                    d2_pose = empty_kpts
                    d2_pose_score = empty_kpts_score
                    d2_bboxes_xyxy = empty_bbox
                    d2_none_index = []

        # 最终统一转到 CPU（保持原有行为）
        pt_info: Dict[str, Any] = {
            "optical_flow": optical_flow.cpu(),
            "depth": depth.cpu(),
            "none_index": list(bbox_none_index),
            "YOLO": {
                "bbox": yolo_bbox.cpu(),  # (T,4) or (0,4)
                "mask": yolo_mask.cpu(),  # (T,1,H,W) or (0,1,H,W)
                "keypoints": yolo_keypoints.cpu(),  # (T,17,3) or (0,17,3)
                "keypoints_score": yolo_keypoints_score.cpu(),  # (T,17) or (0,17)
            },
            "detectron2": {
                "bbox": d2_bboxes_xyxy.cpu(),
                "keypoints": d2_pose.cpu(),
                "keypoints_score": d2_pose_score.cpu(),
                # 如需暴露 d2 的 none_index，可另外添加；当前保持向后兼容
            },
        }

        report = check_pt_info_shapes(
            pt_info, strict=False, logger=logger, check_frames=False
        )
        if not report["ok"]:
            logger.warning("pt_info shape issues: %s", report["problems"])

        return pt_info


def check_pt_info_shapes(
    pt_info: Dict[str, Any],
    *,
    T: Optional[int] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
    kp_num: int = 17,
    allow_empty: bool = True,
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
    # 新增：是否校验 frames；当只有 frames_path 时是否尝试加载
    check_frames: bool = True,
    load_external_frames: bool = True,
) -> Dict[str, Any]:
    """
    校验 Preprocess.__call__ 返回的 pt_info 的形状与一致性。
    - 当仅有光流时，可自动推断 T = flow_T + 1。
    - 现在也会校验 frames (THWC)，或在仅有 frames_path 时可选加载后校验。

    Returns:
        {"ok": bool, "T": int|None, "H": int|None, "W": int|None, "problems": [str, ...]}
    """
    _log = logger or logging.getLogger(__name__)
    problems: List[str] = []

    def _warn(msg: str):
        problems.append(msg)
        _log.warning(msg)

    def _shape(t: torch.Tensor) -> Tuple[int, ...]:
        return tuple(t.shape)

    def _is_tensor(x) -> bool:
        return isinstance(x, torch.Tensor)

    def _is_empty_tensor(x: torch.Tensor) -> bool:
        return x.numel() == 0 or x.shape[0] == 0

    # -------- gather tensors --------
    depth = pt_info.get("depth", None)
    flow = pt_info.get("optical_flow", None)
    none_index = pt_info.get("none_index", [])

    yolo = pt_info.get("YOLO", {}) or {}
    y_bbox = yolo.get("bbox", None)
    y_mask = yolo.get("mask", None)
    y_kpt = yolo.get("keypoints", None)
    y_kpt_s = yolo.get("keypoints_score", None)

    d2 = pt_info.get("detectron2", {}) or {}
    d2_bbox = d2.get("bbox", None)
    d2_kpt = d2.get("keypoints", None)
    d2_kpt_s = d2.get("keypoints_score", None)

    # frames / frames_path
    frames = pt_info.get("frames", None)
    frames_path = pt_info.get("video_path", None)

    # -------- infer T/H/W --------
    T_exact_cands: List[int] = (
        []
    )  # 这些张量的第 0 维就是 T：depth/mask/bbox/kp/kp_score/frames
    T_flow_cands: List[int] = []  # flow 的第 0 维是 T-1
    Hcands: List[int] = []
    Wcands: List[int] = []

    def _collect_THW(
        t: Optional[torch.Tensor],
        *,
        expect_c: Optional[int] = None,
        mask_like: bool = False,
        is_flow: bool = False,
    ):
        if not _is_tensor(t) or (allow_empty and _is_empty_tensor(t)):
            return
        if t.dim() == 4:  # e.g., depth/mask/flow
            if is_flow:
                T_flow_cands.append(int(t.shape[0]))
                if expect_c is not None and t.shape[1] != expect_c:
                    _warn(
                        f"optical_flow 通道数应为 {expect_c}，实际 {t.shape[1]} @ {_shape(t)}"
                    )
            else:
                T_exact_cands.append(int(t.shape[0]))
                if mask_like and t.shape[1] != 1:
                    _warn(f"Mask/Depth 通道数应为 1，实际 {t.shape[1]} @ {_shape(t)}")
                if expect_c is not None and t.shape[1] != expect_c:
                    _warn(f"通道数应为 {expect_c}，实际 {t.shape[1]} @ {_shape(t)}")
            Hcands.append(int(t.shape[-2]))
            Wcands.append(int(t.shape[-1]))
        elif t.dim() == 2:  # e.g., bbox (T,4), kpt_score (T,K)
            T_exact_cands.append(int(t.shape[0]))
        elif t.dim() == 3:  # e.g., keypoints (T,K,D)
            T_exact_cands.append(int(t.shape[0]))

    _collect_THW(depth, expect_c=1, mask_like=True, is_flow=False)
    _collect_THW(flow, expect_c=2, is_flow=True)  # (T-1,2,H,W)
    _collect_THW(y_mask, expect_c=1, mask_like=True, is_flow=False)
    _collect_THW(y_bbox)
    _collect_THW(y_kpt)
    _collect_THW(y_kpt_s)
    _collect_THW(d2_bbox)
    _collect_THW(d2_kpt)
    _collect_THW(d2_kpt_s)

    # 把 frames 也纳入 T/H/W 推断（注意它是 THWC，不走 expect_c 检查）
    if (
        check_frames
        and _is_tensor(frames)
        and (not (allow_empty and _is_empty_tensor(frames)))
    ):
        if frames.dim() == 4:
            T_exact_cands.append(int(frames.shape[0]))
            Hcands.append(int(frames.shape[1]))
            Wcands.append(int(frames.shape[2]))

    # T 推断策略
    if T is not None:
        T_infer = T
    elif T_exact_cands:
        T_infer = max(T_exact_cands)
    elif T_flow_cands:
        T_infer = max(T_flow_cands) + 1
    else:
        T_infer = None

    H_infer = H if H is not None else (max(Hcands) if Hcands else None)
    W_infer = W if W is not None else (max(Wcands) if Wcands else None)

    # -------- per-field checks --------
    # depth: (T,1,H,W)
    if _is_tensor(depth) and (not (allow_empty and _is_empty_tensor(depth))):
        if depth.dim() != 4:
            _warn(f"depth 维度应为 4，实际 {_shape(depth)}")
        else:
            if depth.shape[1] != 1:
                _warn(f"depth 通道数应为 1，实际 {depth.shape[1]}")
            if T_infer is not None and depth.shape[0] != T_infer:
                _warn(f"depth 帧数 {depth.shape[0]} 与 T={T_infer} 不一致")
            if H_infer is not None and depth.shape[2] != H_infer:
                _warn(f"depth H {depth.shape[2]} 与 H={H_infer} 不一致")
            if W_infer is not None and depth.shape[3] != W_infer:
                _warn(f"depth W {depth.shape[3]} 与 W={W_infer} 不一致")

    # optical flow: (T-1,2,H,W)
    if _is_tensor(flow) and (not (allow_empty and _is_empty_tensor(flow))):
        if flow.dim() != 4:
            _warn(f"optical_flow 维度应为 4，实际 {_shape(flow)}")
        else:
            if flow.shape[1] != 2:
                _warn(f"optical_flow 通道数应为 2，实际 {flow.shape[1]}")
            if T_infer is not None:
                exp = max(T_infer - 1, 0)
                if flow.shape[0] != exp:
                    _warn(f"optical_flow 帧数 {flow.shape[0]} 应为 T-1={exp}")
            if H_infer is not None and flow.shape[2] != H_infer:
                _warn(f"optical_flow H {flow.shape[2]} 与 H={H_infer} 不一致")
            if W_infer is not None and flow.shape[3] != W_infer:
                _warn(f"optical_flow W {flow.shape[3]} 与 W={W_infer} 不一致")

    # YOLO
    if _is_tensor(y_bbox) and (not (allow_empty and _is_empty_tensor(y_bbox))):
        if y_bbox.dim() != 2 or y_bbox.shape[1] != 4:
            _warn(f"YOLO.bbox 应为 (T,4)，实际 {_shape(y_bbox)}")
        elif T_infer is not None and y_bbox.shape[0] != T_infer:
            _warn(f"YOLO.bbox T={y_bbox.shape[0]} 与 T={T_infer} 不一致")

    if _is_tensor(y_mask) and (not (allow_empty and _is_empty_tensor(y_mask))):
        if y_mask.dim() != 4 or y_mask.shape[1] != 1:
            _warn(f"YOLO.mask 应为 (T,1,H,W)，实际 {_shape(y_mask)}")
        else:
            if T_infer is not None and y_mask.shape[0] != T_infer:
                _warn(f"YOLO.mask T={y_mask.shape[0]} 与 T={T_infer} 不一致")
            if H_infer is not None and y_mask.shape[2] != H_infer:
                _warn(f"YOLO.mask H {y_mask.shape[2]} 与 H={H_infer} 不一致")
            if W_infer is not None and y_mask.shape[3] != W_infer:
                _warn(f"YOLO.mask W {y_mask.shape[3]} 与 W={W_infer} 不一致")

    if _is_tensor(y_kpt) and (not (allow_empty and _is_empty_tensor(y_kpt))):
        if y_kpt.dim() != 3 or y_kpt.shape[1] != kp_num or y_kpt.shape[2] not in (2, 3):
            _warn(f"YOLO.keypoints 应为 (T,{kp_num},2|3)，实际 {_shape(y_kpt)}")
        elif T_infer is not None and y_kpt.shape[0] != T_infer:
            _warn(f"YOLO.keypoints T={y_kpt.shape[0]} 与 T={T_infer} 不一致")

    if _is_tensor(y_kpt_s) and (not (allow_empty and _is_empty_tensor(y_kpt_s))):
        if y_kpt_s.dim() != 2 or y_kpt_s.shape[1] != kp_num:
            _warn(f"YOLO.keypoints_score 应为 (T,{kp_num})，实际 {_shape(y_kpt_s)}")
        elif T_infer is not None and y_kpt_s.shape[0] != T_infer:
            _warn(f"YOLO.keypoints_score T={y_kpt_s.shape[0]} 与 T={T_infer} 不一致")

    # Detectron2
    if _is_tensor(d2_bbox) and (not (allow_empty and _is_empty_tensor(d2_bbox))):
        if d2_bbox.dim() != 2 or d2_bbox.shape[1] != 4:
            _warn(f"detectron2.bbox 应为 (T,4)，实际 {_shape(d2_bbox)}")
        elif T_infer is not None and d2_bbox.shape[0] != T_infer:
            _warn(f"detectron2.bbox T={d2_bbox.shape[0]} 与 T={T_infer} 不一致")

    if _is_tensor(d2_kpt) and (not (allow_empty and _is_empty_tensor(d2_kpt))):
        if (
            d2_kpt.dim() != 3
            or d2_kpt.shape[1] != kp_num
            or d2_kpt.shape[2] not in (2, 3)
        ):
            _warn(f"detectron2.keypoints 应为 (T,{kp_num},2|3)，实际 {_shape(d2_kpt)}")
        elif T_infer is not None and d2_kpt.shape[0] != T_infer:
            _warn(f"detectron2.keypoints T={d2_kpt.shape[0]} 与 T={T_infer} 不一致")

    if _is_tensor(d2_kpt_s) and (not (allow_empty and _is_empty_tensor(d2_kpt_s))):
        if d2_kpt_s.dim() != 2 or d2_kpt_s.shape[1] != kp_num:
            _warn(
                f"detectron2.keypoints_score 应为 (T,{kp_num})，实际 {_shape(d2_kpt_s)}"
            )
        elif T_infer is not None and d2_kpt_s.shape[0] != T_infer:
            _warn(
                f"detectron2.keypoints_score T={d2_kpt_s.shape[0]} 与 T={T_infer} 不一致"
            )

    # frames 校验（THWC）
    if check_frames:
        _frames = frames
        if _frames is None and isinstance(frames_path, str) and load_external_frames:
            try:
                _frames = torch.load(frames_path, map_location="cpu")
            except Exception as e:
                _warn(f"无法加载 frames_path: {frames_path}，原因：{e}")

        if _is_tensor(_frames) and (not (allow_empty and _is_empty_tensor(_frames))):
            if _frames.dim() != 4:
                _warn(f"frames 维度应为 4(THWC)，实际 {_shape(_frames)}")
            else:
                t, h, w, c = _frames.shape
                if c not in (1, 3):
                    _warn(f"frames 通道 C 应为 1 或 3，实际 {c}")
                # dtype 建议为 uint8；若为 float，也给出提示
                if _frames.dtype != torch.uint8:
                    _warn(f"frames dtype 建议为 uint8，实际 {_frames.dtype}")

                if T_infer is not None and t != T_infer:
                    _warn(f"frames T={t} 与 T={T_infer} 不一致")
                if H_infer is not None and h != H_infer:
                    _warn(f"frames H={h} 与 H={H_infer} 不一致")
                if W_infer is not None and w != W_infer:
                    _warn(f"frames W={w} 与 W={W_infer} 不一致")

    # none_index 范围检查
    if isinstance(none_index, (list, tuple)):
        if T_infer is not None:
            bad = [int(i) for i in none_index if not (0 <= int(i) < T_infer)]
            if bad:
                _warn(f"none_index 存在越界索引: {bad}（T={T_infer}）")
    else:
        _warn("none_index 应为 list/tuple")

    ok = len(problems) == 0
    report = {"ok": ok, "T": T_infer, "H": H_infer, "W": W_infer, "problems": problems}
    if strict and not ok:
        raise AssertionError("pt_info 形状检查未通过: \n- " + "\n- ".join(problems))
    return report
