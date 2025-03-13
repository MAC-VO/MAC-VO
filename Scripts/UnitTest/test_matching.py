import pytest
import torch
from pathlib import Path

from DataLoader import TartanAirV2_StereoSequence
from Module.Frontend.Matching import IMatcher
from Utility.Config import load_config



@pytest.mark.local
@pytest.mark.parametrize(
    ["config"],
    [(str(f),) for f in Path("./Scripts/UnitTest/assets/test_module_config/Flow").glob("*.yaml")]
)
def test_matching(config: str):
    cfg, _ = load_config(Path(config))
    seq    = TartanAirV2_StereoSequence(dict(
        root="./Scripts/UnitTest/assets/test_sequence/TartanAir2_abs_P000",
        compressed=True,
        gtFlow=True, gtDepth=False, gtPose=True,
    ))
    
    frameA, frameB = seq[0], seq[1]
    matcher        = IMatcher.instantiate(cfg.type, cfg.args)
    flow_output    = matcher.estimate(frameA.stereo, frameB.stereo)

    assert flow_output.flow is not None
    B, C, H, W = seq[0].stereo.imageL.shape
    
    # Check flow shape validity
    assert flow_output.flow.shape == torch.Size([B, 2, H, W])
    
    if flow_output.cov is not None:
        assert flow_output.cov.shape == torch.Size([B, 3, H, W])

    if flow_output.mask is not None:
        assert flow_output.mask.shape == torch.Size([B, 1, H, W])
        assert flow_output.mask.dtype == torch.bool

    del matcher
    torch.cuda.empty_cache()
