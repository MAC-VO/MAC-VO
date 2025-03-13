import torch
import signal
import typing as T
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from abc import ABC, abstractmethod

import torch.multiprocessing as mp
from multiprocessing.context import SpawnProcess
from multiprocessing.connection import _ConnectionBase as Conn_Type

from Module.Map import VisualMap, TensorBundle
from Utility.PrettyPrint import Logger
from Utility.Extensions import SubclassRegistry

if T.TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = object


T_GraphInput  = T.TypeVar("T_GraphInput", bound=DataclassInstance)
T_Context     = T.TypeVar("T_Context")
T_GraphOutput = T.TypeVar("T_GraphOutput", bound=DataclassInstance)


def move_dataclass_to_local(obj: T_GraphInput) -> T_GraphInput:
    data_dict = asdict(obj)
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.clone()
        elif isinstance(value, TensorBundle):
            value.apply(lambda x: x.clone())
            data_dict[key] = value
        else:
            data_dict[key] = value
    return type(obj)(**data_dict)


class IOptimizer(ABC, T.Generic[T_GraphInput, T_Context, T_GraphOutput], SubclassRegistry):
    """
    Interface for optimization module. When config.parallel set to `true`, will spawn a child process
    to run optimization loop in "background".
    
    `IOptimizer.optimize(global_map: TensorMap, frames: BatchFrames) -> None`
    
    * In sequential mode, will run optimization loop in blocking mannor and return when optimization is finished.
    
    * In parallel mode, will send optimization job to child process and return immediately (non-blocking).
    
    `IOptimizer.write_back(global_map: TensorMap) -> None`
    
    * In sequential mode, will write back optimization result to global_map immediately and return.
    
    * In parallel mode, will wait for child process to finish optimization job and write back result to global_map. (blocking)

    `IOptimizer.terminate() -> None`
    
    Force terminate child process if in parallel mode. no-op if in sequential mode.
    """
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        
        self.config: SimpleNamespace = config
        self.is_parallel_mode = config.parallel
        
        # For sequential mode
        self.context     : None | T_Context  = None
        self.optimize_res: None | T_GraphOutput = None
        
        # For parallel mode
        self.main_conn : None | Conn_Type = None   # For parent-end
        self.child_conn: None | Conn_Type = None   # For child-end
        self.child_proc: None | SpawnProcess = None   # child process running optimization loop
        
        # Keep track if there is a job on child to avoid deadlock (wait forever for child to finish
        # a non-exist job)
        self.has_job_on_child: bool = False
        
        if self.is_parallel_mode:
            ctx = mp.get_context("spawn")
            torch.set_num_threads(1)
        
            # Generate Pipe for parent-end and child-end
            self.main_conn, self.child_conn = ctx.Pipe(duplex=True)
            # Spawn child process
            self.child_proc = ctx.Process(
                target=IOptimizerParallelWorker,
                args=(config, self.init_context, self._optimize, self.child_conn),
                
            )
            assert self.child_proc is not None
            self.child_proc.start()
            
            torch.set_num_threads(4)
        else:
            self.context = self.init_context(config)
    
    ### Internal Interface to be implemented by the user
    @abstractmethod
    def _get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor) -> T_GraphInput:
        """
        Given current global map and frames of interest (actual meaning depends on the implementation),
        return T_GraphArgs that will be used by optimizer to construct optimization problem.
        """
        ...

    @staticmethod
    @abstractmethod
    def init_context(config) -> T_Context:
        """
        Given config, initialize a *mutable* context object that is preserved between optimizations.
        
        Can also be used to avoid repetitive initialization of some objects (e.g. optimizer, robust kernel).
        """
        ...
    
    @staticmethod
    @abstractmethod
    def _optimize(context: T_Context, graph_data: T_GraphInput) -> tuple[T_Context, T_GraphOutput]:
        """
        Given context and argument, construct the optimization problem, solve it and return the 
        updated context and result.
        """
        ...
    
    @staticmethod
    @abstractmethod
    def _write_map(result: T_GraphOutput | None, global_map: VisualMap) -> None:
        """
        Given the result, write back the result to global_map.
        """
        ...
    
    @property
    def is_running(self) -> bool:
        """
        Returns immediately, indicate the status of optimizer:
        - true if there is a job on child process running.
        - false otherwise - which includes the following cases:
            1. The optimizer is not running in parallel mode (no child process)
            2. The job on child process is finished.
            3. The child process have not received optimization job yet.
        """
        if self.main_conn is None: return False         # Not in Parallel Mode
        if not self.has_job_on_child: return False      # No Job on Child
        return (not self.main_conn.poll(timeout=0))     # Job on Child but not finished

    ### Implementation detail
    
    ## Sequential Version
    def __optimize_sequential(self, global_map: VisualMap, frame_idx: torch.Tensor) -> None:
        graph_args = self._get_graph_data(global_map, frame_idx)
        self.context, self.optimize_res = self._optimize(self.context, graph_args)  #type: ignore
    
    def __writemap_sequential(self, global_map: VisualMap) -> None:
        self._write_map(self.optimize_res, global_map)
        self.optimize_res = None

    ## Parallel Version
    
    def __optimize_parallel(self, global_map: VisualMap, frame_idx: torch.Tensor) -> None:
        assert self.main_conn is not None
        assert self.child_proc and self.child_proc.is_alive()
        
        graph_args: T_GraphInput = self._get_graph_data(global_map, frame_idx)
        self.main_conn.send(graph_args)
        self.has_job_on_child = True

    def __writemap_parallel(self, global_map: VisualMap) -> None:
        assert self.main_conn is not None
        assert self.child_proc and self.child_proc.is_alive()
        
        if not self.has_job_on_child: return
        
        graph_res: T_GraphOutput = self.main_conn.recv()
        graph_res_local = move_dataclass_to_local(graph_res)
        del graph_res
        
        self.has_job_on_child = False
        self._write_map(graph_res_local, global_map)

    ### External Interface used by Users
    def optimize(self, global_map: VisualMap, frame_idx: torch.Tensor) -> None:
        if self.is_parallel_mode:
            # Send T_GraphArg to child process using Pipe
            self.__optimize_parallel(global_map, frame_idx)
        else:
            self.__optimize_sequential(global_map, frame_idx)
    
    def write_map(self, global_map: VisualMap):
        if self.is_parallel_mode:
            # Recv T_GraphArg from child process using Pipe
            self.__writemap_parallel(global_map)
        else:
            self.__writemap_sequential(global_map)

    def terminate(self):
        if self.child_proc and self.child_proc.is_alive():
            self.child_proc.terminate()


def IOptimizerParallelWorker(
    config: SimpleNamespace,
    init_context: T.Callable[[SimpleNamespace,], T_Context],
    optimize: T.Callable[[T_Context, T_GraphInput], tuple[T_Context, T_GraphOutput]],
    child_conn: Conn_Type,
):
    Logger.write("info", "OptimizationParallelWorker started")
    # NOTE: child process ignore keyboard interrupt to ignore deadlock
    # (parent process will terminate child process on exit in MAC-VO implementation)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(8)
    context = init_context(config)
    while True:
        try:
            graph_args: T_GraphInput = child_conn.recv()
        except EOFError:
            continue
        
        graph_args_local = move_dataclass_to_local(graph_args)
        del graph_args
        
        context, graph_res = optimize(context, graph_args_local)
        child_conn.send(graph_res)


# Empty Optimizer
#   A specific implementation of the IOptimizer interface that does nothing
#   Helpful in debugging process.


@dataclass
class EmptyMessageType:
    pass


class EmptyOptimizer(IOptimizer[EmptyMessageType, None, EmptyMessageType]):
    """
    This is a trivial optimizer that do no operations at all. It does not modify the map.
    
    This is used only for debugging and mapping mode VO.
    """
    
    def _get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor) -> EmptyMessageType:
        return EmptyMessageType()
    
    @staticmethod
    def init_context(config) -> None:
        return None
    
    @staticmethod
    def _optimize(context: None, graph_data: EmptyMessageType) -> tuple[None, EmptyMessageType]:
        return None, EmptyMessageType()
    
    @staticmethod
    def _write_map(result: EmptyMessageType | None, global_map: VisualMap) -> None:
        return None
