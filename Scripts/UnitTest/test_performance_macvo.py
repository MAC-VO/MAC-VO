import pytest
from pathlib import Path

from Odometry.MACVO import MACVO
from DataLoader import TartanAirV2_StereoSequence, StereoFrame
from Evaluation.EvalSeq import EvaluateSequences
from Utility.Config import load_config
from Utility.Sandbox  import Sandbox

@pytest.mark.local
@pytest.mark.parametrize(["file_name", "data", "expect_ate", "expect_rte", "expect_roe"], [
    # Performance Test (with rounding up to allow minor performance fluctuation) 
    ("./Scripts/UnitTest/assets/test_config/MACVO/MACVO.yaml", "./Scripts/UnitTest/assets/test_sequence/TartanAir2_abs_P000", 0.002, 0.0025, 0.045),      # at 2024 Sept. 22
])
def test_macvo_performance(file_name: str, data: str, expect_ate: float, expect_rte: float, expect_roe: float):
    seq    = TartanAirV2_StereoSequence(dict(
        root=data,
        compressed=True, gtFlow=False, gtDepth=False, gtPose=True
    ))
    cfg, cfg_dict = load_config(Path(file_name))
    result = Sandbox(Path("./cache/macvo_test_temp"))
    result.set_autoremove()
    result.config = cfg_dict | {"Project": "MACVO_Test"}
    
    system = MACVO[StereoFrame].from_config(cfg)
    system.receive_frames(seq, result)
    
    headers, metrics = EvaluateSequences([str(result.folder)], False)
    
    avg_ATE, avg_RTE, avg_ROE = metrics[0][1], metrics[0][4], metrics[0][7]
    
    assert avg_ATE <= expect_ate, f"Perofmance did not match expectation! get {avg_ATE=}, but expect {expect_ate}"
    assert avg_RTE <= expect_rte, f"Perofmance did not match expectation! get {avg_RTE=}, but expect {expect_rte}"
    assert avg_ROE <= expect_roe, f"Perofmance did not match expectation! get {avg_ROE=}, but expect {expect_roe}"
