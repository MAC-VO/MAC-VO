import pytest
from pathlib import Path

from Utility.Config import load_config
from Odometry.BaselineTartanVO import TartanVO


@pytest.mark.parametrize(["file_name"], [(str(f),) for f in Path("./Config/Experiment/Baseline/TartanVO/").glob("*.yaml")])
def test_tartanvo_config(file_name: str):
    cfg, _ = load_config(Path(file_name))
    TartanVO.is_valid_config(cfg.Odometry)
