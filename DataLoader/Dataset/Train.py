import typing as T
import multiprocessing as mp
from types import SimpleNamespace

from Utility.PrettyPrint import Logger
from ..SequenceBase import SequenceBase
from ..Interface    import DataFramePair, T_Data


class TrainDataset(SequenceBase[DataFramePair[T_Data]]):
    def __init__(self, cfg: SimpleNamespace, from_idx: int = 0, to_idx: int = -1):
        self.sequence = SequenceBase.instantiate(cfg.type, cfg.args).clip(
            start_idx=from_idx,
            end_idx=to_idx if to_idx != -1 else None
        )
        self.length   = len(self.sequence) - 1
        super().__init__(self.length)
    
    def __getitem__(self, local_index: int) -> DataFramePair[T_Data]:
        index = self.get_index(local_index)
        frame, next_frame = self.sequence[index], self.sequence[index + 1]
        data = DataFramePair(idx=frame.idx, cur=frame, nxt=next_frame, time_ns=frame.time_ns)
        return data

    def transform_source(self, actions: list[T.Callable[[T_Data], T_Data]]):
        self.sequence = self.sequence.transform(actions)
        return self
    
    @staticmethod
    def _instantiate_dataset(config: SimpleNamespace, from_idx: int, to_idx: int, do_instantiate: bool) -> "TrainDataset[T_Data] | None":
        if not do_instantiate: return None
        try:
            return TrainDataset[T_Data](config, from_idx, to_idx)
        except Exception as e:
            Logger.show_exception()
            return None

    @classmethod
    def mp_instantiation(cls, data_config: list[SimpleNamespace], from_idx: int, to_idx: int, pred: T.Callable[[SimpleNamespace,], bool]) -> "T.Sequence[TrainDataset[T_Data] | None]":
        """
        Instantiate a huge set of TrainDataset. If not supported it will just return None.
        
        pred: SimpleNamespace -> bool
            A function to check whether to instantiate a TrainDataset for this configuration. 
            (True = include, False = exclude and corresponding list item will be None)
        """
        num_worker = 4
        
        # Start loading everything
        args: list[tuple[SimpleNamespace, int, int, bool]] = [(data_cfg, from_idx, to_idx, pred(data_cfg)) for data_cfg in data_config]
        with mp.Pool(processes=num_worker) as pool:
            return pool.starmap(TrainDataset[T_Data]._instantiate_dataset, args, chunksize=2)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        SequenceBase.is_valid_config(config)
