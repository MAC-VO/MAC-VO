import torch
import traceback
import numpy as np
import pypose as pp

from typing import Callable, Generic
from typing_extensions import Self
from abc import ABC, abstractmethod

from Utility.Sandbox import Sandbox
from DataLoader import SequenceBase, T_Data
from Module.Map import VisualMap
from Utility.PrettyPrint import ColoredTqdm, Logger

from torch.profiler import profile, ProfilerActivity


class IOdometry(ABC, Generic[T_Data]):
    def __init__(self, profile: bool = False) -> None:
        super().__init__()
        self.terminated = False
        self.profile    = profile
        self.profile_save_path = "trace_parallel.json"
    
    def receive_frames(self, sequence: SequenceBase[T_Data], saveto: Sandbox, on_frame_finished: None | Callable[[T_Data, Self, ColoredTqdm], None]=None):
        try:
            reference_poses, reference_time = [], []
            pb = ColoredTqdm(sequence)
            frame: T_Data
            for frame in pb:
                if self.profile and frame.frame_idx == 2:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, with_flops=True) as prof:
                        self.run(frame)
                    prof.export_chrome_trace(self.profile_save_path)
                else:
                    self.run(frame)
                
                if frame.gt_pose is not None:
                    reference_poses.append(frame.gt_pose)
                    reference_time.append(frame.time_ns[0])
                
                if on_frame_finished is not None: on_frame_finished(frame, self, pb)
            
            self.terminate()
            global_map = self.get_map()
            
            sensor_poses = pp.SE3(global_map.frames.data["pose"].tensor)
            T_BS         = pp.SE3(global_map.frames.data["T_BS"].tensor)
            body_poses: np.ndarray = (T_BS @ sensor_poses @ T_BS.Inv()).tensor().cpu().numpy()
            time_ns   : np.ndarray = global_map.frames.data["time_ns"].tensor.cpu().numpy()[:, np.newaxis]
            
            np.save(saveto.path("poses.npy"), np.concatenate([time_ns, body_poses], axis=-1))
            np.savez_compressed(saveto.path("tensor_map.npz"), **global_map.serialize())
            
            if len(reference_poses) > 1:    # At least two poses for a non-trivial trajectory
                ref_body_poses: np.ndarray = torch.cat(reference_poses, dim=0).numpy()
                ref_time_ns   : np.ndarray = np.array(reference_time, dtype=np.float64)[:, np.newaxis]
                np.save(saveto.path("ref_poses.npy"), np.concatenate([ref_time_ns, ref_body_poses], axis=-1))
            else:
                Logger.write("warn", f"Did not write {saveto.path('ref_poses.npy')} since less than 2 ground truth poses in sequence.")
            
        except KeyboardInterrupt as e:
            self.terminate()
            Logger.write("fatal", f"Experiment at {saveto.folder} is interrupted.")
            raise e
        except Exception as e:
            self.terminate()
            Logger.show_exception()
            Logger.write("fatal", f"Failed to execute experiment at {saveto.folder}.")
        
    @abstractmethod
    def run(self, frame: T_Data) -> None:
        """
        Core method for IVisualOdometry. This method handles the incoming frames and perform tracking/mapping internally.
        """
        ...
    
    @abstractmethod
    def get_map(self) -> VisualMap:
        """
        Provides the VisualMap built across multiple calls of .run(...).
        """
        ...

    def terminate(self) -> None: 
        """
        You can define additional operations on terminate. For instance, smoothing trajectory / interpolate bad frames etc.
        """
        self.terminated = True
