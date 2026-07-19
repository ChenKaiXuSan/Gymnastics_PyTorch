"""YAML-backed skeletal contracts for rotation-aware fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from fuse.metadata.mhr70 import mhr_names


@dataclass(frozen=True)
class RoleSpec:
    """A real joint or virtual anatomical role resolved from real joints."""

    kind: str
    joints: tuple[str, ...]
    fallback: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkeletonSpec:
    """Joint ordering, bones, and anatomical roles for one pose format."""

    name: str
    joint_names: tuple[str, ...]
    bones: tuple[tuple[int, int], ...]
    roles: Mapping[str, RoleSpec]
    required_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SkeletonSpec name must be non-empty")
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError("SkeletonSpec joint_names must be unique")
        if not self.joint_names:
            raise ValueError("SkeletonSpec requires at least one joint")
        if any(index < 0 or index >= len(self.joint_names) for bone in self.bones for index in bone):
            raise ValueError("SkeletonSpec bone index is outside joint_names")
        missing = [role for role in self.required_roles if role not in self.roles]
        if missing:
            raise ValueError(f"SkeletonSpec is missing required roles: {missing}")
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))

    def joint_index(self, joint_name: str) -> int:
        try:
            return self.joint_names.index(joint_name)
        except ValueError as error:
            raise KeyError(f"Unknown joint name: {joint_name}") from error

    def role(self, role_name: str) -> RoleSpec:
        try:
            return self.roles[role_name]
        except KeyError as error:
            raise KeyError(f"Unknown skeleton role: {role_name}") from error


def _as_names(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError(f"{field} must be a list of joint names")
    return tuple(value)


def load_skeleton_spec(path: str | Path) -> SkeletonSpec:
    """Load and validate a SkeletonSpec without inventing virtual joints."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Skeleton config must be a mapping: {config_path}")

    joint_names = _as_names(raw.get("joint_names"), "joint_names")
    if joint_names != tuple(mhr_names):
        raise ValueError("MHR70 skeleton joint_names must exactly match fuse.metadata.mhr70.mhr_names")

    raw_roles = raw.get("roles")
    if not isinstance(raw_roles, dict):
        raise ValueError("roles must be a mapping")
    roles: dict[str, RoleSpec] = {}
    for name, role in raw_roles.items():
        if not isinstance(name, str) or not isinstance(role, dict):
            raise ValueError("roles must map names to mappings")
        kind = role.get("kind")
        if kind == "joint":
            joints = (role.get("joint"),)
        else:
            joints = tuple(role.get("joints", ()))
        fallback = tuple(role.get("fallback", ()))
        if kind not in {"joint", "midpoint"} or not all(isinstance(item, str) for item in joints + fallback):
            raise ValueError(f"Invalid role definition for {name}")
        if kind == "joint" and len(joints) != 1:
            raise ValueError(f"Joint role {name} needs exactly one joint")
        if kind == "midpoint" and len(joints) != 2:
            raise ValueError(f"Midpoint role {name} needs exactly two joints")
        unknown = set(joints + fallback) - set(joint_names)
        if unknown:
            raise ValueError(f"Role {name} references unknown joints: {sorted(unknown)}")
        roles[name] = RoleSpec(kind=kind, joints=joints, fallback=fallback)

    raw_bones = raw.get("bones", [])
    if not isinstance(raw_bones, list):
        raise ValueError("bones must be a list")
    bones: list[tuple[int, int]] = []
    for bone in raw_bones:
        if not isinstance(bone, list) or len(bone) != 2 or not all(isinstance(name, str) for name in bone):
            raise ValueError("Each bone must be a two-joint list")
        try:
            bones.append((joint_names.index(bone[0]), joint_names.index(bone[1])))
        except ValueError as error:
            raise ValueError(f"Bone references an unknown joint: {bone}") from error

    required_roles = _as_names(raw.get("required_roles"), "required_roles")
    return SkeletonSpec(
        name=str(raw.get("name", "")),
        joint_names=joint_names,
        bones=tuple(bones),
        roles=roles,
        required_roles=required_roles,
    )
