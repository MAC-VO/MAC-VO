import torch
import traceback
import numpy as np

from typing import TypeVar, Type, Callable, Generic
from typing_extensions import Self
from types import SimpleNamespace
from abc import ABC, abstractmethod

from Utility.Sandbox import Sandbox
from DataLoader import GenericSequence, DataFrame
from Module.Map import TensorMap
from Utility.PrettyPrint import ColoredTqdm, Logger

from torch.profiler import profile, ProfilerActivity


T = TypeVar("T", bound="IVisualOdometry")
T_Frame = TypeVar("T_Frame", bound="DataFrame")


class IVisualOdometry(ABC, Generic[T_Frame]):
    def __init__(self, profile: bool = False) -> None:
        super().__init__()
        self.terminated = False
        self.profile    = profile
        self.profile_save_path = "trace_parallel.json"
    
    def receive_frames(self, sequence: GenericSequence[T_Frame], saveto: Sandbox, on_frame_finished: None | Callable[[T_Frame, Self, ColoredTqdm], None]=None):
        try:
            reference_poses = []
            pb = ColoredTqdm(sequence)
            for frame in pb:
                if self.profile and frame.meta.idx == 2:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, with_flops=True) as prof:
                        self.run(frame)
                    prof.export_chrome_trace(self.profile_save_path)
                else:
                    self.run(frame)
                
                reference_poses.append(frame.gtPose)
                
                if on_frame_finished is not None: on_frame_finished(frame, self, pb)
            
            self.terminate()
            global_map = self.get_map()
            global_map.save_poses(saveto.path("poses.npy"))
            global_map.save_flags(saveto.path("frame_status.pth"))
            global_map.save_map(saveto.path("tensor_map.pth"))
            if all([p is not None for p in reference_poses]):
                np.save(saveto.path("ref_poses.npy"), torch.stack(reference_poses).numpy())
            
            saveto.finish()
        except KeyboardInterrupt as e:
            self.terminate()
            Logger.write("fatal", f"Experiment at {saveto.folder} is interrupted.")
            raise e
        except Exception as e:
            self.terminate()
            Logger.write("fatal", f"Failed to execute experiment at {saveto.folder}.")
            Logger.write("fatal", traceback.format_exc())
        
    @abstractmethod
    def run(self, frame: T_Frame) -> None:
        """
        Core method for IVisualOdometry. This method handles the incoming frames and perform tracking/mapping internally.
        """
        ...
    
    @abstractmethod
    def get_map(self) -> TensorMap:
        """
        Provides the TensorMap built across multiple calls of .run(...).
        """
        ...

    def terminate(self) -> None: 
        """
        You can define additional operations on terminate. For instance, smoothing trajectory / interpolate bad frames etc.
        """
        self.terminated = True
