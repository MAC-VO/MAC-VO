import pytest
from pathlib import Path

from Utility.Config import load_config
from Odometry.MACVO import MACVO
from Odometry.OnlineLoopWindowMACVO import OnlineLoopWindowMACVO
from Odometry.WindowMACVO import WindowMACVO


@pytest.mark.parametrize(
    argnames=["file_name"],
    argvalues=[
        (str(f),) for f in Path("./Config/Experiment/MACVO").rglob("*.yaml")
    ] + [
        (str(f),) for f in Path("./Scripts/UnitTest/assets/test_config/MACVO").rglob("*.yaml")
    ])
def test_macvo_config(file_name: str):
    cfg, _ = load_config(Path(file_name))
    system_type = getattr(cfg.Odometry, "type", "MACVO")
    system_class = {
        "MACVO": MACVO,
        "WindowMACVO": WindowMACVO,
        "OnlineLoopWindowMACVO": OnlineLoopWindowMACVO,
    }[system_type]
    system_class.is_valid_config(cfg.Odometry)
