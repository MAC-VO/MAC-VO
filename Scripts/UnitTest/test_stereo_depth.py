import pytest
import torch
from pathlib import Path

from DataLoader import TartanAirV2_StereoSequence
from Module.Frontend.StereoDepth import IStereoDepth
from Utility.Config import load_config



@pytest.mark.local
@pytest.mark.parametrize(
    ["config"],
    [(str(f),) for f in Path("./Scripts/UnitTest/assets/test_module_config/Depth").glob("*.yaml")]
)
def test_matching(config: str):
    cfg, _ = load_config(Path(config))
    seq    = TartanAirV2_StereoSequence(dict(
        root="./Scripts/UnitTest/assets/test_sequence/TartanAir2_abs_P000",
        compressed=True,
        gtFlow=False, gtDepth=True, gtPose=True
    ))
    
    frameA         = seq[0]
    stereo_depth   = IStereoDepth.instantiate(cfg.type, cfg.args)
    depth_output   = stereo_depth.estimate(frameA.stereo)

    assert depth_output.depth is not None
    B, C, H, W = seq[0].stereo.imageL.shape
    
    # Check flow shape validity
    assert depth_output.depth.shape == torch.Size([B, 1, H, W])
    
    if depth_output.cov is not None:
        assert depth_output.cov.shape == torch.Size([B, 1, H, W])

    if depth_output.mask is not None:
        assert depth_output.mask.shape == torch.Size([B, 1, H, W])
        assert depth_output.mask.dtype == torch.bool

    del stereo_depth
    torch.cuda.empty_cache()
