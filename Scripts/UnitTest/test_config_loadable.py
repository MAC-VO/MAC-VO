import pytest
from pathlib import Path

from Utility.Config import load_config


@pytest.mark.parametrize(argnames=["file_name"], argvalues=[(str(f),) for f in Path("./Config").rglob("*.yaml")])
def test_config_loadable(file_name: str):
    load_config(Path(file_name))
