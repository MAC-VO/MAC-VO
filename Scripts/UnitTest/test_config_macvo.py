import pytest
from pathlib import Path

from Utility.Config import load_config
from Odometry.MACVO import MACVO


@pytest.mark.parametrize(
    argnames=["file_name"],
    argvalues=[
        (str(f),) for f in Path("./Config/Experiment/MACVO").rglob("*.yaml")
    ] + [
        (str(f),) for f in Path("./Scripts/UnitTest/assets/test_config/MACVO").rglob("*.yaml")
    ])
def test_macvo_config(file_name: str):
    cfg, _ = load_config(Path(file_name))
    MACVO.is_valid_config(cfg.Odometry)
