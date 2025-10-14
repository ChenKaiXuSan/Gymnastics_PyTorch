# d2_infer.py
from pathlib import Path
from typing import Optional, Tuple, List, Union, Sequence
import numpy as np
import torch
import cv2
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

COCO_SKELETON = [
    (5, 7),
    (7, 9),  # 左臂
    (6, 8),
    (8, 10),  # 右臂
    (5, 6),  # 肩膀连线
    (11, 13),
    (13, 15),  # 左腿
    (12, 14),
    (14, 16),  # 右腿
    (11, 12),  # 髋部
    (5, 11),
    (6, 12),  # 上身到下身
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),  # 头部
    (1, 2),  # 左右眼
]


# ---------- 1) 构建 cfg / predictor ----------
def build_cfg_and_predictor(
    cfg_path_or_zoo: str,
    score_thresh: float = 0.7,
    device: str = "cuda",
    weights_url_or_path: Optional[str] = None,
):
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path_or_zoo))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = weights_url_or_path or model_zoo.get_checkpoint_url(
        cfg_path_or_zoo
    )
    cfg.INPUT.FORMAT = "BGR"
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


# ---------- 工具：把帧变成 numpy BGR uint8 ----------
def to_numpy_bgr(
    frame: Union[np.ndarray, torch.Tensor], assume_rgb: bool = False
) -> np.ndarray:
    """
    支持:
      - torch.Tensor [T?,H,W,3] 或 [T?,3,H,W]，float(0..1/255) 或 uint8
      - np.ndarray   [T?,H,W,3] 或 [T?,3,H,W]
    返回: 单帧 np.uint8 BGR
    """
    if isinstance(frame, torch.Tensor):
        f = frame.detach().cpu()
        if f.ndim == 3 and f.shape[0] in (1, 3):  # [C,H,W] -> [H,W,C]
            f = f.permute(1, 2, 0)
        arr = f.numpy()
    elif isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # [C,H,W] -> [H,W,C]
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)

    if assume_rgb:
        arr = arr[:, :, ::-1]  # RGB->BGR
    return np.ascontiguousarray(arr)


# ---------- 2) 单帧推理 ----------
def infer_instances(predictor: DefaultPredictor, image_bgr) -> "Instances":
    img = to_numpy_bgr(image_bgr)
    outputs = predictor(img)
    return outputs["instances"].to("cpu")


# ---------- 3) 提取 bbox / kpt 数组 ----------
def instances_to_arrays(instances) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      bboxes    : [N,5]  -> x1,y1,x2,y2,score (float32)
      kpts_xy   : [N,K,2]  (float32)
      kpts_conf : [N,K]    (float32)
    """
    if hasattr(instances, "pred_boxes") and len(instances) > 0:
        xyxy = instances.pred_boxes.tensor.numpy().astype(np.float32)
        scores = instances.scores.numpy().astype(np.float32)[:, None]
        bboxes = np.concatenate([xyxy, scores], axis=1)
    else:
        bboxes = np.zeros((0, 5), dtype=np.float32)

    if hasattr(instances, "pred_keypoints") and len(instances) > 0:
        kpts = instances.pred_keypoints.numpy().astype(np.float32)  # [N,K,3] (x,y,prob)
        kpts_xy = kpts[..., :2]
        kpts_conf = kpts[..., 2]
    else:
        kpts_xy = np.zeros((0, 0, 2), dtype=np.float32)
        kpts_conf = np.zeros((0, 0), dtype=np.float32)

    return bboxes, kpts_xy, kpts_conf


# ---------- 4) 可视化（Detectron2 Visualizer） ----------
def visualize_instances(
    image_bgr: Union[np.ndarray, torch.Tensor],
    instances,
    cfg,
    dataset_name_fallback: Optional[str] = "coco_2017_train",
    scale: float = 1.0,
) -> np.ndarray:
    img_bgr = to_numpy_bgr(image_bgr)
    img_rgb = img_bgr[:, :, ::-1]

    if len(cfg.DATASETS.TRAIN) > 0:
        md = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    else:
        if (
            dataset_name_fallback
            and dataset_name_fallback not in MetadataCatalog.list()
        ):
            MetadataCatalog.get(dataset_name_fallback)
        md = MetadataCatalog.get(dataset_name_fallback)

    v = Visualizer(img_rgb, metadata=md, scale=scale, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances)
    vis_rgb = out.get_image()
    return vis_rgb[:, :, ::-1]  # -> BGR


# ---------- 5) 简易关键点可视化（骨架） ----------
def draw_and_save_keypoints(
    vframes: Union[torch.Tensor, np.ndarray, List],
    keypoints: Union[List[np.ndarray], torch.Tensor, np.ndarray],
    save_root: Union[str, Path],
    video_path: Union[str, Path],
    radius: int = 3,
    color: tuple = (0, 255, 0),
    line_color: tuple = (255, 0, 0),
    draw_indices: bool = False,
    assume_rgb: bool = False,
    stride: int = 1,
):
    """
    vframes: [T,H,W,3] / [T,3,H,W] / list
    keypoints: list of (K,2) np.ndarray 或 [T,K,2] array/tensor（允许包含 NaN）
    保存到: save_root/filter_img/pose/<video_stem>/*.jpg
    """
    video_path = Path(video_path)
    video_stem = video_path.stem
    out_dir = Path(save_root) / "filter_img" / "pose" / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 标准化 keypoints 访问
    if isinstance(keypoints, torch.Tensor):
        kp_seq = [keypoints[i].detach().cpu().numpy() for i in range(len(keypoints))]
    elif isinstance(keypoints, np.ndarray) and keypoints.ndim == 3:
        kp_seq = [keypoints[i] for i in range(keypoints.shape[0])]
    elif isinstance(keypoints, list):
        kp_seq = keypoints
    else:
        raise TypeError(f"Unsupported keypoints type: {type(keypoints)}")

    # 获取帧迭代器和 T
    if isinstance(vframes, (torch.Tensor, np.ndarray)):
        T = len(vframes)
        frames_iter = (vframes[i] for i in range(T))
    else:
        T = len(vframes)
        frames_iter = iter(vframes)

    for idx, (frame, kpt) in tqdm(
        enumerate(zip(frames_iter, kp_seq)),
        total=T,
        desc="Draw & Save KPT",
        leave=False,
    ):
        if idx % stride != 0:
            continue

        img = to_numpy_bgr(frame, assume_rgb=assume_rgb)  # BGR uint8

        # kpt: (K,2) 可能含 NaN
        if kpt is None or (
            isinstance(kpt, np.ndarray) and (kpt.size == 0 or np.isnan(kpt).all())
        ):
            # 直接保存原图
            cv2.imwrite(str(out_dir / f"{idx:04d}_pose_filter.jpg"), img)
            continue

        # 画骨架与点
        H, W = img.shape[:2]

        def _valid(p):
            return (
                np.isfinite(p[0])
                and np.isfinite(p[1])
                and (0 <= p[0] < W)
                and (0 <= p[1] < H)
            )

        # 画线
        for i, j in COCO_SKELETON:
            if i < kpt.shape[0] and j < kpt.shape[0]:
                pi, pj = kpt[i], kpt[j]
                if _valid(pi) and _valid(pj):
                    cv2.line(
                        img,
                        (int(pi[0]), int(pi[1])),
                        (int(pj[0]), int(pj[1])),
                        line_color,
                        2,
                    )

        # 画点
        for j in range(kpt.shape[0]):
            p = kpt[j]
            if _valid(p):
                cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)
                if draw_indices:
                    cv2.putText(
                        img,
                        str(j),
                        (int(p[0]) + 3, int(p[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        cv2.imwrite(str(out_dir / f"{idx:04d}_pose_filter.jpg"), img)


def _to_numpy_bgr(
    frame: Union[np.ndarray, torch.Tensor], assume_rgb: bool = False
) -> np.ndarray:
    """单帧 -> np.uint8 BGR"""
    if isinstance(frame, torch.Tensor):
        f = frame.detach().cpu()
        if f.ndim == 3 and f.shape[0] in (1, 3):  # [C,H,W] -> [H,W,C]
            f = f.permute(1, 2, 0)
        img = f.numpy()
    else:
        img = frame
        if img.ndim == 3 and img.shape[0] in (1, 3):  # [C,H,W]
            img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
    if assume_rgb:
        img = img[:, :, ::-1]  # RGB->BGR
    return img


def _xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> tuple:
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return x1, y1, x2, y2


def draw_and_save_boxes(
    vframes: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    bboxes: Sequence,  # 每帧一个 bbox 或 (bbox, score)；也可 None/NaN
    save_root: Union[str, Path],
    video_path: Union[str, Path],
    *,
    fmt: Optional[str] = None,  # 'xyxy' / 'xywh' / None(自动)
    assume_rgb: bool = False,
    stride: int = 1,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    draw_score: bool = True,
):
    """
    vframes: [T,H,W,3] / [T,3,H,W] / list[H,W,3]
    bboxes:  长度 T；元素可为
             - (4,) -> 框
             - (5,) -> 框+score（最后一个当 score）
             - dict/tuple/list 中含 'xyxy' 或 'xywh'
             - None 或 含 NaN 时跳过
    fmt:     明确提供 'xyxy' 或 'xywh' 更快；不提供则自动判断
    """
    video_path = Path(video_path)
    save_root = Path(save_root)

    # 组织输出路径
    out_dir = save_root / "filter_img" / "bbox" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一长度 & 迭代器
    if isinstance(vframes, (torch.Tensor, np.ndarray)):
        T = len(vframes)
        frame_iter = (vframes[i] for i in range(T))
    else:
        T = len(vframes)
        frame_iter = iter(vframes)

    # 安全取 bbox
    def _parse_bbox(bb) -> tuple:
        """返回 (x1,y1,x2,y2,score_or_None)；若不可用返回 None"""
        if bb is None:
            return None
        # 允许 (bbox, score)
        score = None
        if (
            isinstance(bb, (list, tuple))
            and len(bb) == 2
            and isinstance(bb[1], (float, int, np.floating, np.integer))
        ):
            bb, score = bb[0], float(bb[1])

        # dict 带键
        if isinstance(bb, dict):
            if "xyxy" in bb:
                arr = np.asarray(bb["xyxy"], dtype=float).reshape(-1)
                if arr.size >= 4:
                    xyxy = arr[:4]
                else:
                    return None
            elif "xywh" in bb:
                arr = np.asarray(bb["xywh"], dtype=float).reshape(-1)
                if arr.size >= 4:
                    xyxy = np.array(_xywh_to_xyxy(*arr[:4]), dtype=float)
                else:
                    return None
            else:
                return None
        else:
            arr = np.asarray(bb, dtype=float).reshape(-1)
            if arr.size < 4:
                return None
            if fmt == "xyxy":
                xyxy = arr[:4]
                if arr.size >= 5 and score is None:
                    score = float(arr[4])
            elif fmt == "xywh":
                xyxy = np.array(_xywh_to_xyxy(*arr[:4]), dtype=float)
                if arr.size >= 5 and score is None:
                    score = float(arr[4])
            else:
                # 自动判断：若 x2>x1 且 y2>y1 可能是 xyxy，否则按 xywh
                if arr[2] > arr[0] and arr[3] > arr[1]:
                    xyxy = arr[:4]
                else:
                    xyxy = np.array(_xywh_to_xyxy(*arr[:4]), dtype=float)
                if arr.size >= 5 and score is None:
                    score = float(arr[4])

        if not np.all(np.isfinite(xyxy)):
            return None
        return (*xyxy.tolist(), score)

    # 主循环
    for i, (frame, bb) in tqdm(
        enumerate(zip(frame_iter, bboxes)),
        total=T,
        desc="Draw & Save BBoxes",
        leave=False,
    ):
        if i % stride != 0:
            continue

        img = _to_numpy_bgr(frame, assume_rgb=assume_rgb)  # BGR
        H, W = img.shape[:2]

        parsed = _parse_bbox(bb)
        if parsed is None:
            # 无框：直接保存原图
            cv2.imwrite(str(out_dir / f"{i:04d}_bbox_filter.jpg"), img)
            continue

        x1, y1, x2, y2, sc = parsed
        # 裁剪到边界并转 int
        x1 = int(np.clip(x1, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            cv2.imwrite(str(out_dir / f"{i:04d}_bbox_filter.jpg"), img)
            continue

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 画分数
        if draw_score and sc is not None and np.isfinite(sc):
            label = f"{sc:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(out_dir / f"{i:04d}_bbox_filter.jpg"), img)


def _center_xyxy(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return np.stack([cx, cy], axis=-1)


def _pick_person_by_prev_center(
    bboxes_xyxy: np.ndarray, prev_center_xy: np.ndarray
) -> int:
    if bboxes_xyxy.shape[0] == 1:
        return 0
    centers = _center_xyxy(bboxes_xyxy)
    d2 = np.sum((centers - prev_center_xy[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


class Detectron2Wrapper:
    def __init__(self, config, person: str) -> None:
        self.cfg, self.predictor = build_cfg_and_predictor(
            config.detectron2.cfg,
            getattr(config, "score_thresh", 0.7),
            getattr(config, "device", "cuda"),
            getattr(config, "weights", None),
        )
        root = Path(config.extract_dataset.save_path)
        self.save_dir = root / "vis" / "d2" / person

        self.save_vis = bool(getattr(config, "vis", True))
        self.vis_stride = int(getattr(config, "vis_stride", 1))
        self.assume_rgb = bool(getattr(config, "assume_rgb", True))

    def __call__(
        self,
        vframes: Union[torch.Tensor, np.ndarray, List],
        video_path: Union[str, Path],
    ):
        """
        返回:
          pose:          [T, K, 2]  (float32)   — 所选人的关键点坐标
          pose_score:    [T, K]     (float32)   — 所选人的关键点置信度
          bboxes_xyxy:   [T, 4]     (float32)   — 所选人的 bbox (x1,y1,x2,y2)
          none_index:    List[int]             — 未检出帧索引（NaN 占位）
        """
        video_path = Path(video_path)
        stem = video_path.stem
        out_dir = self.save_dir / "img" / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 统一可迭代和长度
        if isinstance(vframes, (torch.Tensor, np.ndarray)):
            T = len(vframes)
            iterable = (vframes[i] for i in range(T))
        else:
            T = len(vframes)
            iterable = iter(vframes)

        poses_xy_list: List[np.ndarray] = []  # 每帧 [K,2] 或 NaN
        poses_sc_list: List[np.ndarray] = []  # 每帧 [K]   或 NaN
        bboxes_list: List[np.ndarray] = []  # 每帧 [4]   或 NaN
        none_index: List[int] = []

        prev_center = None  # 上一帧选中人的 bbox 中心

        for idx, frame in tqdm(
            enumerate(iterable), total=T, desc="Detectron2 Inference"
        ):
            img = to_numpy_bgr(frame, assume_rgb=self.assume_rgb)
            instances = infer_instances(self.predictor, img)

            # 当前帧候选
            bboxes, kpts_xy_all, kpts_conf_all = instances_to_arrays(
                instances
            )  # bboxes [N,5], kpts [N,K,2], conf [N,K]
            N = bboxes.shape[0]

            if idx == 0:
                if N == 0:
                    none_index.append(idx)
                    poses_xy_list.append(np.full((0, 2), np.nan, dtype=np.float32))
                    poses_sc_list.append(np.full((0,), np.nan, dtype=np.float32))
                    bboxes_list.append(np.full((4,), np.nan, dtype=np.float32))
                    prev_center = None
                else:
                    pick = int(np.argmax(bboxes[:, 4])) if N > 1 else 0
                    xyxy = bboxes[pick, :4][None, :]
                    prev_center = _center_xyxy(xyxy)[0]
                    poses_xy_list.append(kpts_xy_all[pick])
                    poses_sc_list.append(kpts_conf_all[pick])
                    bboxes_list.append(bboxes[pick, :4])
            else:
                if N == 0:
                    none_index.append(idx)
                    poses_xy_list.append(np.full_like(poses_xy_list[0], np.nan))
                    poses_sc_list.append(np.full_like(poses_sc_list[0], np.nan))
                    bboxes_list.append(np.full((4,), np.nan, dtype=np.float32))
                else:
                    pick = (
                        int(np.argmax(bboxes[:, 4]))
                        if prev_center is None
                        else _pick_person_by_prev_center(bboxes[:, :4], prev_center)
                    )
                    xyxy = bboxes[pick, :4][None, :]
                    prev_center = _center_xyxy(xyxy)[0]
                    poses_xy_list.append(kpts_xy_all[pick])
                    poses_sc_list.append(kpts_conf_all[pick])
                    bboxes_list.append(bboxes[pick, :4])

            # 可视化（抽帧）
            if self.save_vis and (idx % self.vis_stride == 0):
                vis = visualize_instances(
                    img, instances, self.cfg, dataset_name_fallback="coco_2017_train"
                )
                cv2.imwrite(str(out_dir / f"frame_{idx:04d}.jpg"), vis)

        # —— 统一 K 大小（找一个非空帧确定 K）——
        K = None
        for arr in poses_xy_list:
            if arr.size != 0 and not np.isnan(arr).all():
                K = arr.shape[0]
                break
        if K is None:
            # 全部缺失
            return (
                torch.empty((T, 0, 2), dtype=torch.float32),
                torch.empty((T, 0), dtype=torch.float32),
                torch.full((T, 4), np.nan, dtype=torch.float32),
                none_index,
            )

        # pad 到 (T,K,2) / (T,K)
        def _pad_xy(a):
            if a.size == 0 or np.isnan(a).all():
                return np.full((K, 2), np.nan, dtype=np.float32)
            if a.shape[0] == K:
                return a.astype(np.float32)
            out = np.full((K, 2), np.nan, dtype=np.float32)
            m = min(K, a.shape[0])
            out[:m] = a[:m]
            return out

        def _pad_sc(s):
            if s.size == 0 or np.isnan(s).all():
                return np.full((K,), np.nan, dtype=np.float32)
            if s.shape[0] == K:
                return s.astype(np.float32)
            out = np.full((K,), np.nan, dtype=np.float32)
            m = min(K, s.shape[0])
            out[:m] = s[:m]
            return out

        poses_xy_arr = np.stack([_pad_xy(a) for a in poses_xy_list], axis=0)  # [T,K,2]
        poses_sc_arr = np.stack([_pad_sc(s) for s in poses_sc_list], axis=0)  # [T,K]
        bboxes_arr = np.stack(
            [
                b if b.shape == (4,) else np.full((4,), np.nan, dtype=np.float32)
                for b in bboxes_list
            ],
            axis=0,
        )  # [T,4]

        # （可选）关键点骨架图：默认仅在 save_vis==True 且 stride 一致时启用
        if self.save_vis:
            draw_and_save_keypoints(
                vframes=vframes,
                keypoints=poses_xy_list,
                save_root=self.save_dir,
                video_path=video_path,
                radius=3,
                color=(0, 255, 0),
                line_color=(255, 0, 0),
                draw_indices=False,
                assume_rgb=self.assume_rgb,
                stride=self.vis_stride,
            )
            draw_and_save_boxes(
                vframes=vframes,
                bboxes=bboxes_list,
                save_root=self.save_dir,
                video_path=video_path,
                fmt="xyxy",
            )

        pose = torch.from_numpy(poses_xy_arr)  # [T,K,2]
        pose_score = torch.from_numpy(poses_sc_arr)  # [T,K]
        bboxes_xyxy = torch.from_numpy(bboxes_arr)  # [T,4]
        return pose, pose_score, bboxes_xyxy, none_index
