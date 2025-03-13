import torch
import time
from pathlib import Path
from typing import ClassVar
from functools import wraps
from contextlib import contextmanager

from Utility.PrettyPrint import Logger

T_CUDAEvent = torch._C._CudaEventBase

class Timer:
    ACTIVE: ClassVar[bool] = False
    CPU_TIME_STREAM: ClassVar[dict[str, tuple[list[float], list[float]]]] = dict()
    GPU_TIME_STREAM: ClassVar[dict[str, tuple[list[T_CUDAEvent], list[T_CUDAEvent]]]] = dict()
    GPU_STREAMS: ClassVar[set[torch.cuda.Stream]] = set()

    @classmethod
    def setup(cls, active: bool):
        cls.ACTIVE = active
        if active: Logger.write("info", "Timer is set to active.")

    @classmethod
    def cpu_timeit(cls, name: str):        
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                if not cls.ACTIVE: return func(*args, **kwargs)
                time_stream = cls.CPU_TIME_STREAM.get(name, ([], []))
                
                time_stream[0].append(time.time() * 1000)
                result = func(*args, **kwargs)
                time_stream[1].append(time.time() * 1000)
                
                cls.CPU_TIME_STREAM[name] = time_stream
                return result
            return wrapped
        return decorator

    @classmethod
    def gpu_timeit(cls, name: str):
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                if not cls.ACTIVE: return func(*args, **kwargs)
                stream = torch.cuda.current_stream()
                time_stream = cls.GPU_TIME_STREAM.get(name, ([], []))
                start_event = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
                end_event   = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
                time_stream[0].append(start_event)
                time_stream[1].append(end_event)
                cls.GPU_STREAMS.add(stream)
                cls.GPU_TIME_STREAM[name] = time_stream
                
                start_event.record(stream)
                result = func(*args, **kwargs)
                end_event.record(stream)
                
                return result
            return wrapped
        return decorator

    @classmethod
    @contextmanager
    def CPUTimingContext(cls, name: str):
        if not cls.ACTIVE:
            yield
            return
        time_stream = cls.CPU_TIME_STREAM.get(name, ([], []))
        time_stream[0].append(time.time())
        yield
        time_stream[1].append(time.time())
        cls.CPU_TIME_STREAM[name] = time_stream

    @classmethod
    @contextmanager
    def GPUTimingContext(cls, name: str, stream: torch.cuda.Stream):
        if not cls.ACTIVE:
            yield
            return
        cls.GPU_STREAMS.add(stream)
        time_stream = cls.GPU_TIME_STREAM.get(name, ([], []))
        time_stream[0].append(start_event := torch.cuda.Event(enable_timing=True))
        time_stream[1].append(end_event   := torch.cuda.Event(enable_timing=True))
        
        start_event.record(stream)
        yield
        end_event.record(stream)
        cls.GPU_TIME_STREAM[name] = time_stream

    @classmethod
    def report(cls):
        if not cls.ACTIVE: return
        report_str = "\n--- Timer Report ---\n"
        
        # CPU Side timing
        report_str += "CPU Timers:\n"
        for name, (starts, ends) in cls.CPU_TIME_STREAM.items():
            elapsed = list(map(lambda pair: pair[1] - pair[0], zip(starts, ends)))
            
            if len(starts) == 0:
                report_str += f"\t{name}".ljust(20) +  \
                    f" | #Call= 0".ljust(20) + \
                    f" | AvgTime=N/A".ljust(20) + \
                    f" | MedianTime=N/A\n"
            else:
                median_elapsed = torch.tensor(elapsed).median().item()
                report_str += f"\t{name}".ljust(20) + \
                    f" | #Call={len(starts)}".ljust(20) + \
                    f" | AvgTime={(sum(elapsed) / len(starts)) : 3f} ms".ljust(20) + \
                    f" | MedianTime={median_elapsed : 3f} ms\n"

        # GPU Side timing
        
        # Ensure everything is finished
        report_str += "GPU Timers:\n"
        for stream in cls.GPU_STREAMS: stream.synchronize()
        for name, (starts, ends) in cls.GPU_TIME_STREAM.items():
            elapsed = list(map(lambda pair: pair[0].elapsed_time(pair[1]), zip(starts, ends)))
            if len(starts) == 0:
                report_str += f"\t{name}".ljust(20) +  \
                    f" | #Call= 0".ljust(20) + \
                    f" | AvgTime=N/A".ljust(20) + \
                    f" | MedianTime=N/A\n"
            else:
                median_elapsed = torch.tensor(elapsed).median().item()
                report_str += f"\t{name}".ljust(20) + \
                    f" | #Call={len(starts)}".ljust(20) + \
                    f" | AvgTime={(sum(elapsed) / len(starts)) : 3f} ms".ljust(20) + \
                    f" | MedianTime={median_elapsed : 3f} ms\n"
        
        Logger.write("info", report_str)

    @classmethod
    def save_elapsed(cls, json_file: str | Path):
        if not cls.ACTIVE: return
        import json
        for stream in cls.GPU_STREAMS: stream.synchronize()
        with open(json_file, "w") as f:
            json.dump({
                "CPU_ElapsedTime": {
                    name : [end - start for start, end in zip(*cls.CPU_TIME_STREAM[name])]
                    for name in cls.CPU_TIME_STREAM
                },
                "GPU_ElapsedTime": {
                    name : [start.elapsed_time(end) for start, end in zip(*cls.GPU_TIME_STREAM[name])]
                    for name in cls.GPU_TIME_STREAM
                }
            }, f)
            Logger.write("info", f"Elapsed time information write to {json_file}")
