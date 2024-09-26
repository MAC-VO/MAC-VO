from Module.Map import TensorMap
from .Interface import IOptimizer


class EmptyMessageType:
    def move_self_to_local(self): return EmptyMessageType()
    def release(self): return


class TrivialOptimizer(IOptimizer[EmptyMessageType, None, EmptyMessageType]):
    """
    This is a trivial optimizer that do no operations at all. It does not modify the map.
    
    This is used only for debugging and mapping mode VO.
    """
    
    def _get_graph_data(self, global_map: TensorMap, frame_idx: list[int]) -> EmptyMessageType:
        return EmptyMessageType()
    
    @staticmethod
    def init_context(config) -> None:
        return None
    
    @staticmethod
    def _optimize(context: None, graph_data: EmptyMessageType) -> tuple[None, EmptyMessageType]:
        return None, EmptyMessageType()
    
    @staticmethod
    def _write_map(result: EmptyMessageType | None, global_map: TensorMap) -> None:
        return None
