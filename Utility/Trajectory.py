from __future__ import annotations
import torch
import pypose as pp
import numpy  as np

from pathlib import Path
from typing import Generic, Literal, TypeVar, Any, Callable, cast
from evo.core.trajectory import PoseTrajectory3D

from Utility.Math import interpolate_pose, NormalizeQuat
from Utility.Sandbox import Sandbox

T = TypeVar("T")
T2 = TypeVar("T2")


class Plotable(Generic[T]):
    def __init__(self, data: T, name: str, plot_kwargs: dict[str, Any]):
        self.data = data
        self.name = name
        self.plot_kwargs: dict[str, Any] = {
            # Default kwargs for matplotlib, can be overwritten by plot_kwargs
            "alpha": 0.8,
            } | plot_kwargs

    def apply(self, func: Callable[[T,], T2]) -> Plotable[T2]:
        return Plotable(func(self.data), self.name, self.plot_kwargs)
    
    def __repr__(self) -> str:
        return f"Plotable<{self.data.__class__.__name__}>(data={self.data}, name='{self.name}')"


class Trajectory:
    def __init__(self, poses: pp.LieTensor, time: torch.Tensor, frame_status: torch.Tensor):
        self.poses = poses
        self.time  = time
        self.frame_status = frame_status
    
    def __getitem__(self, index) -> Trajectory:
        return Trajectory(
            pp.LieTensor(self.poses.__getitem__(index), ltype=self.poses.ltype),
            self.time.__getitem__(index),
            self.frame_status.__getitem__(index),
        )
    
    @classmethod
    def from_SE3_txt(cls, file: Path,
                     timestamp: torch.Tensor | None = None,
                     frame_status: torch.Tensor | None = None) -> Trajectory:
        SE3_matrix = pp.SE3(np.loadtxt(file))
        if timestamp is None:
            timestamp = torch.arange(0, SE3_matrix.size(0), step=1)
        if frame_status is None:
            frame_status = torch.zeros((SE3_matrix.size(0),), dtype=torch.bool)
        else:
            assert frame_status.dtype == torch.bool
        return cls(SE3_matrix, timestamp, frame_status)
    
    @classmethod
    def from_SE3_numpy(cls, file: Path,
                       timestamp: torch.Tensor | None = None,
                       frame_status: torch.Tensor | None = None) -> Trajectory:
        SE3_matrix = pp.SE3(np.load(file).astype(float))
        if timestamp is None:
            timestamp = torch.arange(0, SE3_matrix.size(0), step=1)
        if frame_status is None:
            frame_status = torch.zeros((SE3_matrix.size(0),), dtype=torch.bool)
        else:
            assert frame_status.dtype == torch.bool
        return cls(SE3_matrix, timestamp, frame_status)

    @classmethod
    def from_timed_SE3_txt(cls, file: Path, frame_status: torch.Tensor | None = None) -> Trajectory:
        data = np.loadtxt(file, dtype=float)
        timestamp, SE3_matrix = torch.tensor(data[:, 0]), pp.SE3(data[:, 1:])
        if frame_status is None:
            frame_status = torch.zeros((SE3_matrix.size(0),), dtype=torch.bool)
        else:
            assert frame_status.dtype == torch.bool
        return cls(SE3_matrix, timestamp, frame_status)
    
    @classmethod
    def from_timed_SE3_numpy(cls, file: Path, frame_status: torch.Tensor | None = None) -> Trajectory:
        data = np.load(file).astype(float)
        timestamp, SE3_matrix = torch.tensor(data[:, 0]), pp.SE3(data[:, 1:])
        if frame_status is None:
            frame_status = torch.zeros((SE3_matrix.size(0),), dtype=torch.bool)
        return cls(SE3_matrix, timestamp, frame_status)
    
    @classmethod
    def from_sandbox(cls, box: Sandbox, align_time: Literal["est->gt", "gt->est", None]="est->gt") -> tuple[Plotable[Trajectory], Plotable[Trajectory]]:
        """
        Returns (GT_Trajectory, Est_Trajectory) pair with name.
        """
        assert box.path("poses.npy").exists()       , f"Unable to load poses.npy from {box.folder}"
        assert box.path("ref_poses.npy").exists()   , f"Unable to load ref_poses.npy from {box.folder}"

        flag_path = box.path("frame_status.pth")
        if flag_path.exists():
            frame_status = torch.load(flag_path, weights_only=True)
        else:
            frame_status = None
        est_traj = cls.from_timed_SE3_numpy(box.path("poses.npy"), frame_status=frame_status)
        gt_traj  = cls.from_timed_SE3_numpy(box.path("ref_poses.npy"))
        est_traj = est_traj.align_origin(gt_traj)
        
        gt_traj.time  = gt_traj.time - gt_traj.time[0]   #FIXME: this is only fore debugging purpose
        est_traj.time = est_traj.time - est_traj.time[0] #FIXME: this is only fore debugging purpose
        
        match align_time:
            case "est->gt":
                est_traj = est_traj.align_time(gt_traj.time)
            case "gt->est":
                gt_traj  = gt_traj.align_time(est_traj.time)
            case None:
                pass

        est_name = box.config.Project if hasattr(box.config, "Project") else box.folder.parent.name
        return Plotable(gt_traj, "Ref",  dict()), Plotable(est_traj, est_name, dict())
    
    @classmethod
    def from_sandbox_mayberef(cls, box: Sandbox) -> tuple[PlotableTrajectory | None, PlotableTrajectory]:
        if box.path("ref_poses.npy").exists():
            return cls.from_sandbox(box)
        
        flag_path = box.path("frame_status.pth")
        if flag_path.exists():
            frame_status = torch.load(flag_path)
        else:
            frame_status = None
        est_traj = cls.from_timed_SE3_numpy(box.path("poses.npy"), frame_status=frame_status)
        return None, Plotable(est_traj, box.config.Project, dict())

    @classmethod
    def from_evo(cls, evo_traj: PoseTrajectory3D, frame_status: torch.Tensor | None=None) -> Trajectory:
        positions = torch.tensor(evo_traj.positions_xyz)
        rotations_xyzw = torch.tensor(evo_traj.orientations_quat_wxyz).roll(shifts=-1, dims=-1)
        poses_SE3 = pp.SE3(torch.cat([positions, rotations_xyzw], dim=1).float())
        if frame_status is None:
            frame_status = torch.zeros((poses_SE3.size(0),))
        return Trajectory(poses_SE3, torch.tensor(evo_traj.timestamps), frame_status)

    @property
    def length(self) -> int: return self.poses.size(0)

    @property
    def translation(self) -> torch.Tensor:
        return self.poses.translation()

    @property
    def rotation(self) -> pp.LieTensor:
        return self.poses.rotation()

    @property
    def as_evo(self) -> PoseTrajectory3D:
        positions = self.poses.translation().numpy()
        rotation_wxyz = self.poses.rotation().roll(shifts=1, dims=-1).numpy()
        evo_traj = PoseTrajectory3D(
            positions_xyz=positions,
            orientations_quat_wxyz=rotation_wxyz,
            timestamps=self.time.numpy()
        )
        return evo_traj

    @property
    def as_motion(self) -> MotionSequence:
        start_time = self.time[0].item()
        duration = self.time[1:] - self.time[:-1]
        motions  = cast(pp.LieTensor, self.poses[:-1]).Inv() @ self.poses[1:]
        return MotionSequence(motions, duration, self.frame_status[1:], pp.SE3(self.poses[0]), start_time)

    def align_time(self, new_time: torch.Tensor) -> Trajectory:
        aligned_pose, frame_status = interpolate_pose(self.poses, self.time, new_time)
        # TODO: may want to also do 'interpolate' on frame_status (e.g. find nearest neighbor and 
        # get corresponding frame_status flag).
        return Trajectory(aligned_pose, new_time, frame_status)

    def align_SE3(self, ref_traj: Trajectory) -> Trajectory:
        self_evo, ref_evo = self.as_evo, ref_traj.as_evo
        result = self_evo.align(ref_evo, correct_scale=False)
        return Trajectory.from_evo(self_evo, self.frame_status)

    def align_scale(self, ref_traj: Trajectory) -> Trajectory:
        self_evo, ref_evo = self[~self.frame_status].as_evo, ref_traj[~self.frame_status].as_evo
        _, _, scale = self_evo.align(ref_evo, correct_scale=True, correct_only_scale=True)
        return self.scale(scale)
    
    def align_origin(self, ref_traj: Trajectory) -> Trajectory:
        self_evo, ref_evo = self.as_evo, ref_traj.as_evo
        self_evo.align_origin(ref_evo)
        return Trajectory.from_evo(self_evo, self.frame_status)
    
    def crop(self, from_idx: int | None = None, to_idx: int | None = None):
        return Trajectory(
            cast(pp.LieTensor, self.poses[from_idx:to_idx]),
            self.time[from_idx:to_idx],
            self.frame_status[from_idx:to_idx]
        )
    
    def scale(self, s: float) -> Trajectory:
        self_evo = self.as_evo
        self_evo.scale(s)
        return Trajectory.from_evo(self_evo)
    
    def __repr__(self) -> str:
        return f"Trajectory(length={self.length})"


class MotionSequence:
    def __init__(self, motions: pp.LieTensor, durations: torch.Tensor, frame_status: torch.Tensor,
                 initial_pose: pp.LieTensor = pp.identity_SE3(), start_time: float = 0.0):
        self.motions = motions
        self.duration = durations
        self.frame_status = frame_status
        assert self.motions.size(0) == self.duration.size(0)
        assert self.motions.size(0) == self.frame_status.size(0)
        
        self.initial_pose = initial_pose
        self.start_time = start_time

    @property
    def length(self) -> int:
        return self.motions.size(0)

    @property
    def translation(self) -> torch.Tensor:
        return self.motions.translation()
    
    @property
    def rotation(self) -> pp.LieTensor:
        return self.motions.rotation()

    @property
    def as_trajectory(self) -> Trajectory:
        ident_poses = pp.cumops(self.motions, dim=0, ops=lambda a, b: NormalizeQuat(a) @ NormalizeQuat(b))
        ident_times = pp.cumops(self.duration, dim=0, ops=lambda a, b: a + b)
        poses = self.initial_pose @ ident_poses
        times = self.start_time + ident_times
        frame_status = torch.cat([torch.tensor([0.]), self.frame_status], dim=0)
        return Trajectory(poses, times, frame_status)

    def __repr__(self) -> str:
        return f"MotionSequence(length={self.length})"

PlotableMotions = Plotable[MotionSequence]
PlotableTrajectory = Plotable[Trajectory]
