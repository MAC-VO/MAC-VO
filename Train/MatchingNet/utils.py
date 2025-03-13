import torch
import warnings
from typing import get_args, get_origin, Literal, TypeGuard, TypeVar


T_DataType  = Literal["fp32", "fp16", "bf16"]
T_TrainType = Literal["flow", "cov", "flow+cov", "finalcov"]
T_Scheduler = Literal["OneCycleLR"]
T_Optimizer = Literal["AdamW"]

T_Type = TypeVar("T_Type")

def AssertLiteralType(value: str | float | int | bool, t: T_Type) -> TypeGuard[T_Type]:
    if get_origin(t) is Literal:
        if value in get_args(t): return True
        else: raise ValueError(f"AssertLiteralType does not match - get {value}, expect to be one of {get_args(t)}")
    else:
        raise Exception(f"AssertLiteralType can only check for literal types.")


def get_datatype(type_string: T_DataType) -> torch.dtype:
    AssertLiteralType(type_string, T_DataType)
    match type_string:
        case "fp32":
            return torch.float32
        case "fp16":
            warnings.warn(f"Training with fp16 may cause loss to become nan. Consider use bf16/fp32 for better stability.")
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _: raise Exception("Impossible")


def get_scheduler(scheduler_string: T_Scheduler) -> type[torch.optim.lr_scheduler.LRScheduler]:
    AssertLiteralType(scheduler_string, T_Scheduler)
    
    match scheduler_string:
        case "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR
        case _: raise Exception("Impossible")


def get_optimizer(optim_string: T_Optimizer) -> type[torch.optim.Optimizer]:
    AssertLiteralType(optim_string, T_Optimizer)
    
    match optim_string:
        case "AdamW":
            return torch.optim.AdamW
        case _: raise Exception("Impossible")
