"""Bounded online image and pending-depth storage for loop validation."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypose as pp
import torch

from DataLoader import StereoData
from Module.Frontend.StereoDepth import IStereoDepth


@dataclass(frozen=True)
class StoredLoopKeyframe:
    frame_id: int
    time_ns: int
    pose: torch.Tensor
    left_path: Path
    right_path: Path
    K: torch.Tensor
    T_BS: torch.Tensor
    baseline: torch.Tensor
    height: int
    width: int


@dataclass(frozen=True)
class LoadedLoopKeyframe:
    frame_id: int
    pose: torch.Tensor
    stereo: StereoData


def _image_to_uint8(image):
    image = image.detach().cpu().float()[0].permute(1, 2, 0)
    return (image.clamp(0, 1) * 255).round().byte().numpy()


def _write_png(path, image):
    rgb = _image_to_uint8(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Failed to write loop keyframe image {path}")


class LoopKeyframeStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = {}

    def add(self, frame_id, stereo, pose):
        folder = self.root / f"{frame_id:06d}"
        folder.mkdir(exist_ok=False)
        left_path = folder / "left.png"
        right_path = folder / "right.png"
        _write_png(left_path, stereo.imageL)
        _write_png(right_path, stereo.imageR)
        record = StoredLoopKeyframe(
            frame_id=int(frame_id),
            time_ns=int(stereo.frame_ns),
            pose=torch.as_tensor(pose).detach().cpu().double().clone(),
            left_path=left_path,
            right_path=right_path,
            K=stereo.K.detach().cpu().clone(),
            T_BS=stereo.T_BS.tensor().detach().cpu().clone(),
            baseline=stereo.baseline.detach().cpu().clone(),
            height=stereo.height,
            width=stereo.width,
        )
        self.records[frame_id] = record
        return record

    def load(self, frame_id):
        record = self.records[frame_id]
        left = cv2.cvtColor(
            cv2.imread(str(record.left_path), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        right = cv2.cvtColor(
            cv2.imread(str(record.right_path), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        image_left = torch.from_numpy(left.copy()).permute(2, 0, 1)[None].float() / 255
        image_right = torch.from_numpy(right.copy()).permute(2, 0, 1)[None].float() / 255
        stereo = StereoData(
            T_BS=pp.SE3(record.T_BS.clone()),
            K=record.K.clone(),
            baseline=record.baseline.clone(),
            time_ns=[record.time_ns],
            height=record.height,
            width=record.width,
            imageL=image_left,
            imageR=image_right,
        )
        return LoadedLoopKeyframe(record.frame_id, record.pose.clone(), stereo)


@dataclass(frozen=True)
class PendingLoopTarget:
    frame_id: int
    stereo: StereoData
    depth: IStereoDepth.Output


def _cpu_depth(output):
    def move(value):
        return None if value is None else value.detach().cpu().clone()

    return IStereoDepth.Output(
        depth=move(output.depth),
        disparity=move(output.disparity),
        cov=move(output.cov),
        mask=move(output.mask),
        disparity_uncertainty=move(output.disparity_uncertainty),
    )


class PendingLoopTargetStore:
    def __init__(self, capacity=8):
        if capacity < 1:
            raise ValueError("pending target capacity must be positive")
        self.capacity = capacity
        self._targets = OrderedDict()

    @property
    def frame_ids(self):
        return tuple(self._targets)

    def add(self, packet):
        stored = PendingLoopTarget(
            int(packet.frame_id), packet.stereo, _cpu_depth(packet.depth)
        )
        self._targets[stored.frame_id] = stored
        self._targets.move_to_end(stored.frame_id)
        if len(self._targets) > self.capacity:
            _, expired = self._targets.popitem(last=False)
            return expired
        return None

    def get(self, frame_id):
        return self._targets.get(frame_id)

    def pop(self, frame_id):
        return self._targets.pop(frame_id, None)
