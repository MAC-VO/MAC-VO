import pytest
from pathlib import Path

from Utility.Config import load_config
from DataLoader import SequenceBase


@pytest.mark.parametrize(["file_name"], [(str(f),) for f in Path("./Config/Sequence").glob("**/*.yaml")])
def test_sequence_cfg(file_name: str):
    cfg, _ = load_config(Path(file_name))
    if isinstance(cfg, list):
        for c in cfg: SequenceBase.is_valid_config(c)
    else:
        SequenceBase.is_valid_config(cfg)
