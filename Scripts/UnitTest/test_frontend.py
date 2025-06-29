import pytest
import torch
from pathlib import Path

from DataLoader import TartanAirV2_StereoSequence
from Module.Frontend.Frontend import IFrontend
from Utility.Config import load_config



@pytest.mark.local
@pytest.mark.parametrize(
    ["config"],
    [(str(f),) for f in Path("./Scripts/UnitTest/assets/test_module_config/Frontend").glob("*.yaml")]
)
def test_frontend(config: str):
    cfg, _ = load_config(Path(config))
    seq    = TartanAirV2_StereoSequence(dict(
        root="./Scripts/UnitTest/assets/test_sequence/TartanAir2_abs_P000",
        compressed=True,
        gtFlow=True, gtDepth=False, gtPose=True,
    ))
    
    frameA, frameB = seq[0], seq[1]
    frontend       = IFrontend.instantiate(cfg.type, cfg.args)
    depth_output, flow_output    = frontend.estimate_pair(frameA.stereo, frameB.stereo)
    B, C, H, W = seq[0].stereo.imageL.shape

    # Depth shape validity
    assert depth_output.depth.shape == torch.Size([B, 1, H, W])
    
    if depth_output.cov is not None:
        assert depth_output.cov.shape == torch.Size([B, 1, H, W])

    if depth_output.mask is not None:
        assert depth_output.mask.shape == torch.Size([B, 1, H, W])
        assert depth_output.mask.dtype == torch.bool

    # Flow shape validity
    assert flow_output.flow.shape == torch.Size([B, 2, H, W])
    
    if flow_output.cov is not None:
        assert flow_output.cov.shape == torch.Size([B, 3, H, W])

    if flow_output.mask is not None:
        assert flow_output.mask.shape == torch.Size([B, 1, H, W])
        assert flow_output.mask.dtype == torch.bool
    
    del frontend
    torch.cuda.empty_cache()
