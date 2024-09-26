import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from Module.Frontend.Frontend import IFrontend, ModelContext
from Utility.Config import build_dynamic_config

def GetFlops(model: IFrontend[ModelContext]):
    input_A = torch.rand(1, 3, 640, 640, device='cuda')
    input_B = torch.rand(1, 3, 640, 640, device='cuda')
        
    flops = FlopCountAnalysis(model.context["model"], (input_A, input_B))
    print(f"Total Flops: {flops.total()/1e9} G")
    print(flop_count_table(flops))



flowformer_cfg, _ = build_dynamic_config({
    "type": "FlowFormerCovFrontend",
    "args": {
        "device": "cuda",
        "weight": "./Model/MACVO_FrontendCov.pth",
        "use_jit": False
    }
})

frontend = IFrontend.instantiate(flowformer_cfg.type, flowformer_cfg.args)
GetFlops(frontend)

