import pytest
from pathlib import Path

from Odometry.MACVO import MACVO
from DataLoader import TartanAirV2Sequence
from Evaluation.EvalSeq import EvaluateSequences
from Utility.Config import load_config
from Utility.Space  import Sandbox

@pytest.mark.local
def test_macvo_performance():
    seq    = TartanAirV2Sequence(
        root="./Scripts/UnitTest/assets/test_sequence/TartanAir2_abs_P000",
        ips=100, compressed=True,
        gtFlow=False, gtDepth=False, gtPose=True,
        useIMU=False
    )
    cfg, cfg_dict = load_config(Path("./Scripts/UnitTest/assets/test_config/MACVO_unittest.yaml"))
    result = Sandbox(Path("./cache/macvo_test_temp"))
    result.set_autoremove()
    result.config = cfg_dict | {"Project": "MACVO_Test"}
    
    system = MACVO.from_config(cfg, seq)
    system.receive_frames(seq, result)
    
    headers, metrics = EvaluateSequences([str(result.folder)], False)
    
    avg_ATE, avg_RTE, avg_ROE = metrics[0][1], metrics[0][4], metrics[0][7]
    
    # NOTE: Result as of 2024 Sept 22.
    # ATE 0.0016286766772153987
    # RTE 0.002320774545358606
    # ROE 0.040772523764985066
    
    assert avg_ATE <= 0.002
    assert avg_RTE <= 0.003
    assert avg_ROE <= 0.06
