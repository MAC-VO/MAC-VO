from typing import Callable
from types  import SimpleNamespace

import warnings
import torch.multiprocessing as mp

from .SequenceBase import GenericSequence
from .Interface import FramePair


class TrainDataset(GenericSequence[FramePair]):
    def __init__(self, from_idx: int = 0, to_idx: int = -1, **kwargs) -> None:
        self.sequence = GenericSequence.instantiate(**kwargs).clip(
            start_idx=from_idx,
            end_idx=to_idx if to_idx != -1 else None
        )
        self.length = len(self.sequence) - 1
        self.transforms = []
    
        super().__init__(self.length)
    
    def __getitem__(self, local_index: int) -> FramePair:
        index = self.get_index(local_index)
        frame, next_frame = self.sequence[index], self.sequence[index + 1]
        data = FramePair(frame, next_frame)
        for fn in self.transforms:
            data = fn(data)
        return data

    def transform(self, f: Callable[[FramePair], FramePair]) -> "TrainDataset":
        self.transforms.append(f)
        return self
    
    @staticmethod
    def _instantiate_dataset(from_idx: int, to_idx: int, config: SimpleNamespace, do_instantiate: bool) -> "TrainDataset | None":
        if not do_instantiate: return None
        try:
            return TrainDataset(from_idx, to_idx, **vars(config.args))
        except Exception as e:
            warnings.warn(f"Failed to load dataset with config {config} - Reason: {e}")
            return None

    @classmethod
    def mp_instantiation(cls, data_config: list[SimpleNamespace], from_idx: int, to_idx: int, pred: Callable[[SimpleNamespace,], bool]) -> "list[TrainDataset | None]":
        """
        Instantiate a huge set of TrainDataset. If not supported it will just return None.
        
        pred: SimpleNamespace -> bool
            A function to check whether to instantiate a TrainDataset for this configuration. 
            (True = include, False = exclude and corresponding list item will be None)
        """
        num_worker = 4
        
        # Start loading everything
        args: list[tuple[int, int, SimpleNamespace, bool]] = [(from_idx, to_idx, data_cfg, pred(data_cfg)) for data_cfg in data_config]
        with mp.Pool(processes=num_worker) as pool:
            return pool.starmap(TrainDataset._instantiate_dataset, args, chunksize=2)
    
