#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Union
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------- Core types ----------------------------


@dataclass
class Extrinsics:
    """
    OpenCV 约定：
      X_cam = R_wc * X_world + t_wc
    同时提供 camera->world：
      X_world = R_cw * X_cam + t_cw，且 t_cw == C (相机中心, world)
    """

    R_wc: np.ndarray  # (3,3)
    t_wc: np.ndarray  # (3,)
    R_cw: np.ndarray  # (3,3)
    t_cw: np.ndarray  # (3,)
    C: np.ndarray  # (3,)


# ---------------------------- Math utils ----------------------------


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def _proj_to_so3(R: np.ndarray) -> np.ndarray:
    """将 3x3 矩阵投影到最近的 SO(3)，稳定正交化。"""
    U, _, Vt = np.linalg.svd(R)
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:
        U[:, -1] *= -1
        R_ = U @ Vt
    return R_


def rodrigues_to_R(rvec: Iterable[float]) -> np.ndarray:
    """Rodrigues 向量(弧度, 世界->相机) -> 旋转矩阵(3x3)"""
    r = np.asarray(rvec, float).reshape(3)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], float)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return _proj_to_so3(R)


def ypr_deg_to_R(
    yaw_deg: float, pitch_deg: float, roll_deg: float, order: str = "ZYX"
) -> np.ndarray:
    """
    yaw/pitch/roll(度) -> R_cw(相机->世界)；默认 ZYX：R = Rz(yaw)*Ry(pitch)*Rx(roll)
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    mapping = {"X": Rx, "Y": Ry, "Z": Rz}
    R = np.eye(3)
    for ax in order:
        R = mapping[ax] @ R
    return _proj_to_so3(R)


# ------------------------- Pose construction ------------------------


def lookat_Rt(
    C: Iterable[float], T: Iterable[float], up: Iterable[float] = (0, 0, 1), filp_y=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    由相机位置 C 和目标点 T 生成世界->相机 (R_wc, t_wc)
    相机坐标：+Z 前、+X 右、+Y 下 (OpenCV)
    """
    C = np.asarray(C, float).reshape(3)
    T = np.asarray(T, float).reshape(3)
    up = _normalize(np.asarray(up, float).reshape(3))

    z_cam = T - C
    if np.linalg.norm(z_cam) < 1e-12:
        raise ValueError("Camera and target coincide")
    z_cam = _normalize(z_cam)

    x_cam = np.cross(z_cam, up)
    if np.linalg.norm(x_cam) < 1e-6:  # up ~ z_cam，换一个应急 up
        alt_up = np.array([0, 1, 0]) if abs(z_cam[1]) < 0.9 else np.array([1, 0, 0])
        x_cam = np.cross(z_cam, alt_up)
    x_cam = _normalize(x_cam)

    y_cam = np.cross(x_cam, z_cam)
    y_cam = _normalize(y_cam)
    if filp_y:
        y_cam = -y_cam

    R_cw = np.stack([x_cam, y_cam, z_cam], axis=1)  # cam->world
    R_wc = R_cw.T
    t_wc = -R_wc @ C
    return R_wc, t_wc


def compose_extrinsics_from(
    cam_pos: Iterable[float],
    orientation: Dict[str, Union[Tuple[float, float, float], float]],
    orientation_mode: str = "lookat",
    up: Iterable[float] = (0, 0, 1),
) -> Extrinsics:
    """
    orientation_mode:
      - 'lookat': orientation['target'] = (x,y,z)
      - 'ypr'   : orientation['yaw','pitch','roll'] (度)，得到 R_cw 再转置
      - 'rodrigues': orientation['rvec'] 世界->相机
    """
    C = np.asarray(cam_pos, float).reshape(3)

    if orientation_mode == "lookat":
        R_wc, t_wc = lookat_Rt(C, orientation["target"], up=up)
    elif orientation_mode == "ypr":
        R_cw = ypr_deg_to_R(
            orientation["yaw"], orientation["pitch"], orientation["roll"], order="ZYX"
        )
        R_wc = R_cw.T
        t_wc = -R_wc @ C
    elif orientation_mode == "rodrigues":
        R_wc = rodrigues_to_R(orientation["rvec"])
        t_wc = -R_wc @ C
    else:
        raise ValueError(f"Unknown orientation_mode: {orientation_mode}")

    R_cw = R_wc.T
    t_cw = C
    return Extrinsics(R_wc=R_wc, t_wc=t_wc, R_cw=R_cw, t_cw=t_cw, C=C)


def build_extrinsics_map(
    camera_layout: Dict[int, Dict], default_up: Iterable[float] = (0, 0, 1)
) -> Dict[int, Extrinsics]:
    """
    camera_layout[id] = {
        'pos': (x,y,z),
        one of: 'target' | 'ypr' | 'rvec',
        optional: 'up'
    }
    """
    out: Dict[int, Extrinsics] = {}
    for cid, spec in camera_layout.items():
        pos = spec["pos"]
        if "target" in spec:
            ext = compose_extrinsics_from(
                pos,
                {"target": spec["target"]},
                orientation_mode="lookat",
                up=spec.get("up", default_up),
            )
        elif "ypr" in spec:
            yaw, pitch, roll = spec["ypr"]
            ext = compose_extrinsics_from(
                pos,
                {"yaw": yaw, "pitch": pitch, "roll": roll},
                orientation_mode="ypr",
                up=spec.get("up", default_up),
            )
        elif "rvec" in spec:
            ext = compose_extrinsics_from(
                pos,
                {"rvec": spec["rvec"]},
                orientation_mode="rodrigues",
                up=spec.get("up", default_up),
            )
        else:
            raise ValueError(
                f"Camera {cid} must contain one of: 'target' | 'ypr' | 'rvec'"
            )
        out[cid] = ext
    return out


def make_projection_matrices(
    extrinsics_map: Dict[int, Extrinsics], K_map: Optional[Dict[int, np.ndarray]] = None
) -> Dict[int, np.ndarray]:
    """P = K [R|t]；若 K_map=None 则用 I3。"""
    P_map: Dict[int, np.ndarray] = {}
    for cid, ext in extrinsics_map.items():
        K = np.eye(3) if K_map is None else K_map[cid]
        Rt = np.hstack([ext.R_wc, ext.t_wc.reshape(3, 1)])
        P_map[cid] = K @ Rt
    return P_map


# ------------------------- Layout helpers ---------------------------


def angles_to_position(
    target: Tuple[float, float, float],
    distance: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    z_override: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    由 (distance, yaw, pitch) 计算相机中心 C，使相机 +Z 指向 target。
    约定：Z上,yaw 绕 Z轴，（逆时针为 +），pitch 仰角。
    注意，这里的相机是需要看向 target 的。

    Args:
        target (Tuple[float, float, float]): 目标点坐标
        distance (float): 相机与目标点的距离
        yaw_deg (float): 相机绕 Z 轴旋转的角度
        pitch_deg (float, optional): 相机绕 X 轴旋转的角度。默认为 0.0。
        z_override (Optional[float], optional): 覆盖相机 Z 轴坐标。默认为 None。

    Returns:
        Tuple[float, float, float]: 相机中心 C 的坐标
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    f = np.array(
        [np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], float
    )  # cam forward(+Z)
    T = np.asarray(target, float)
    C = T - distance * f
    if z_override is not None:
        C[2] = z_override
    return tuple(C.tolist())


# --------------------------- Visualization -------------------------


def _make_axes_points(
    ext: Extrinsics, axis_len: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    Xc = np.eye(3) * axis_len  # cam axes endpoints in cam
    # Xc[:,1] *= -1                      # 这里渲染的时候让 Y 轴反向更直观
    Xw = (ext.R_cw @ Xc) + ext.t_cw.reshape(3, 1)
    return Xw, ext.C


def _frustum_corners_world(
    ext: Extrinsics,
    K: Optional[np.ndarray],
    img_size: Optional[Tuple[int, int]],
    depth: float,
) -> np.ndarray:
    """
    返回 (3,4) 的世界坐标四角点。img_size = (width, height)
    """
    if K is None or img_size is None:
        fov = np.deg2rad(60 / 2)
        w = depth * np.tan(fov)
        h = w * 0.75
        corners_cam = np.array(
            [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]], float
        ).T
    else:
        W, H = img_size
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pixels = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
        x = (pixels[:, 0] - cx) * (depth / fx)
        y = (pixels[:, 1] - cy) * (depth / fy)
        z = np.full_like(x, depth)
        corners_cam = np.vstack([x, y, z])

    corners_world = (ext.R_cw @ corners_cam) + ext.t_cw.reshape(3, 1)
    return corners_world


def draw_cameras_matplotlib(
    extrinsics_map: Dict[int, Extrinsics],
    K_map: Optional[Dict[int, np.ndarray]] = None,
    img_size: Optional[Tuple[int, int]] = None,
    frustum_depth: float = 0.4,
    axis_len: float = 0.2,
    figsize: Tuple[int, int] = (8, 8),
    elev: float = 20,
    azim: float = 40,
    auto_equal: bool = True,
    save_path: Optional[str] = None,
):
    """画相机中心、坐标轴、视锥；img_size=(width,height)"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []

    for cid, ext in sorted(extrinsics_map.items()):
        # center
        ax.scatter(ext.C[0], ext.C[1], ext.C[2], marker="o")
        all_pts.append(ext.C)

        # axes
        axes_end, Cw = _make_axes_points(ext, axis_len=axis_len)
        ax.plot(
            [Cw[0], axes_end[0, 0]],
            [Cw[1], axes_end[1, 0]],
            [Cw[2], axes_end[2, 0]],
            color="r",
        )  # X
        ax.plot(
            [Cw[0], axes_end[0, 1]],
            [Cw[1], axes_end[1, 1]],
            [Cw[2], axes_end[2, 1]],
            color="g",
        )  # Y
        ax.plot(
            [Cw[0], axes_end[0, 2]],
            [Cw[1], axes_end[1, 2]],
            [Cw[2], axes_end[2, 2]],
            color="b",
        )  # Z

        # frustum
        K = None if K_map is None else K_map.get(cid, None)
        corners = _frustum_corners_world(ext, K, img_size, depth=frustum_depth)  # (3,4)

        order = [0, 1, 2, 3, 0]  # rim
        ax.plot(corners[0, order], corners[1, order], corners[2, order], color="orange")
        for j in range(4):  # rays
            ax.plot(
                [Cw[0], corners[0, j]],
                [Cw[1], corners[1, j]],
                [Cw[2], corners[2, j]],
                color="orange",
            )

        all_pts.append(corners.T)
        ax.text(ext.C[0], ext.C[1], ext.C[2], f"Cam {cid}", fontsize=9)

    all_pts = np.vstack(all_pts)
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_zlabel("Z (world)")
    ax.view_init(elev=elev, azim=azim)

    if auto_equal and all_pts.size > 0:
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        centers = (mins + maxs) / 2.0
        ranges = maxs - mins
        radius = float(np.max(ranges)) * 0.6 if np.all(ranges > 0) else 1.0
        ax.set_xlim([centers[0] - radius, centers[0] + radius])
        ax.set_ylim([centers[1] - radius, centers[1] + radius])
        ax.set_zlim([centers[2] - radius, centers[2] + radius])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=180)
    return fig, ax


def save_multi_views(
    extrinsics_map: Dict[int, Extrinsics],
    K_map: Optional[Dict[int, np.ndarray]] = None,
    img_size: Optional[Tuple[int, int]] = None,
    save_prefix: str = "cameras",
    views: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    一次性导出多视图（可自定义）
    默认：front/left/top
    """
    if views is None:
        views = {
            "front": dict(elev=0, azim=0),  # 从 +X 看
            "left": dict(elev=0, azim=90),  # 从 +Y 看
            "top": dict(elev=90, azim=-90),  # 俯视
        }
    for name, view in views.items():
        fig, _ = draw_cameras_matplotlib(
            extrinsics_map,
            K_map=K_map,
            img_size=img_size,
            elev=view["elev"],
            azim=view["azim"],
            save_path=f"{save_prefix}_{name}.png",
        )
        plt.close(fig)


def prepare_camera_position(
    K: np.array,
    yaws: Dict[int, float],
    T: Tuple[float, float, float],
    r: float,
    z: float,
    output_path: Optional[str] = None,
    img_size: Optional[Tuple[int, int]] = None,
) -> Dict[int, Dict]:
    """
    准备相机位置数据，返回字典格式，包含相机ID、位置和朝向信息。

    Args:
        extrinsics_map (Dict[int, Extrinsics]): 包含相机外参的字典，键为相机ID，值为Extrinsics对象。

    Returns:
        Dict[int, Dict]: 包含相机位置和朝向信息的字典，键为相机ID，值为包含位置和朝向的字典。
    """
    CAMERA_LAYOUT: Dict[int, Dict] = {}
    for cid, yaw in yaws.items():
        C = angles_to_position(T, r, yaw, pitch_deg=0.0, z_override=z)
        CAMERA_LAYOUT[cid] = {"pos": C, "target": T}

    # 由布局生成外参
    extr_map = build_extrinsics_map(CAMERA_LAYOUT)

    # —— 打印外参摘要 ——
    for cid, ext in sorted(extr_map.items()):
        print(f"[Cam {cid}]")
        print(
            "R_wc=\n",
            np.array2string(ext.R_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "t_wc=",
            np.array2string(ext.t_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "C(w) =",
            np.array2string(ext.C, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print()

    rt_info = dict()
    for cid, ext in extr_map.items():
        rt_info[cid] = {
            "R": ext.R_wc,
            "t": ext.t_wc,
            "C": ext.C,
        }

    # —— 可视化 & 多视图导出 ——
    K_map = {cid: K for cid in CAMERA_LAYOUT.keys()}
    
    draw_cameras_matplotlib(
        extr_map,
        K_map=K_map,
        img_size=img_size,
        frustum_depth=0.6,
        axis_len=0.25,
        save_path=os.path.join(output_path, "camera_poses.png"),
    )
    save_multi_views(
        extr_map,
        K_map=K_map,
        img_size=img_size,
        save_prefix=os.path.join(output_path, "camera_poses"),
    )

    return {
        "layout": CAMERA_LAYOUT,  # {cid: {"pos": (x, y, z), "target": T}}
        "extrinsics_map": extr_map,  # {cid: Extrinsics}
        "K_map": K_map,  # {cid: K}
        "rt_info": rt_info,  # {cid: {"R": R_wc, "t": t_wc, "C": C}}
    }


# ------------------------------- Demo -------------------------------

if __name__ == "__main__":
    # —— 相机布局（按角度放置，全部看向原点） ——
    T = (0.0, 0.0, 1.5)  # 目标（人）位置
    r = 3.5  # 半径 (m)
    z = 1.5  # 高度 (m)
    # * 按照3号相机为原点进行旋转
    yaws = {1: -90.0, 2: -45.0, 3: 0.0, 4: -135.0}

    CAMERA_LAYOUT: Dict[int, Dict] = {}
    for cid, yaw in yaws.items():
        C = angles_to_position(T, r, yaw, pitch_deg=0.0, z_override=z)
        CAMERA_LAYOUT[cid] = {"pos": C, "target": T}

    extr_map = build_extrinsics_map(CAMERA_LAYOUT)

    # —— 内参（若多相机不同，可分别给） ——
    K = np.array(
        [
            [1710.4629148432577, 0.0, 550.1152435515663],
            [0.0, 1711.318414718867, 896.8609628805682],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    K_map = {cid: K for cid in CAMERA_LAYOUT.keys()}

    # —— 打印外参摘要 ——
    for cid, ext in sorted(extr_map.items()):
        print(f"[Cam {cid}]")
        print(
            "R_wc=\n",
            np.array2string(ext.R_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "t_wc=",
            np.array2string(ext.t_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "C(w) =",
            np.array2string(ext.C, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print()

    # —— 可视化 & 多视图导出 ——
    img_size = (1080, 1920)  # (width, height)
    draw_cameras_matplotlib(
        extr_map,
        K_map=K_map,
        img_size=img_size,
        frustum_depth=0.6,
        axis_len=0.25,
        save_path="camera_poses.png",
    )
    save_multi_views(
        extr_map, K_map=K_map, img_size=img_size, save_prefix="camera_poses"
    )
