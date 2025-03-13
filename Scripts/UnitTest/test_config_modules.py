import pytest
from pathlib import Path

from Utility.Config import load_config
from Module.Frontend.Matching import IMatcher
from Module.Frontend.StereoDepth import IStereoDepth
from Module.Frontend.Frontend import IFrontend


@pytest.mark.parametrize(
    argnames=["file_name"],
    argvalues=[(str(f),) for f in Path("./Scripts/UnitTest/assets/test_module_config/Flow").rglob("*.yaml")])
def test_matcher_config(file_name: str):
    cfg, _ = load_config(Path(file_name))
    IMatcher.is_valid_config(cfg)


@pytest.mark.parametrize(
    argnames=["file_name"],
    argvalues=[(str(f),) for f in Path("./Scripts/UnitTest/assets/test_module_config/Depth").rglob("*.yaml")])
def test_stereo_config(file_name: str):
    cfg, _ = load_config(Path(file_name))
    IStereoDepth.is_valid_config(cfg)
    

@pytest.mark.parametrize(
    argnames=["file_name"],
    argvalues=[(str(f),) for f in Path("./Scripts/UnitTest/assets/test_module_config/Frontend").rglob("*.yaml")])
def test_frontend_config(file_name: str):
    cfg, _ = load_config(Path(file_name))
    IFrontend.is_valid_config(cfg)
