import cv2
import torch
import numpy as np
import pypose as pp
import glob, re, os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from torch.utils.data import Dataset

from ..Interface import StereoFrame, StereoData
from ..SequenceBase import SequenceBase


class ZedSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "Zed"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.raw_width = cfg.width
        self.raw_height = cfg.height
        
        K = torch.tensor([[cfg.fx, 0.0, cfg.cx], [0.0, cfg.fy, cfg.cy], [0.0, 0.0, 1.0]]).unsqueeze(0)
        
        width = self.raw_width
        height = self.raw_height
        
        # metadata
        self.seqRoot = Path(cfg.root)
        self.K = K
        self.baseline = cfg.bl
        self.width = width
        self.height = height
        
        self.ImageL = ZedMonocularDataset(Path(self.seqRoot), left=True)
        self.ImageR = ZedMonocularDataset(Path(self.seqRoot), left=False)

        self.length = len(self.ImageL)
        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        
        return StereoFrame(
            idx    = [local_index],
            time_ns= [local_index * 1000],   # FIXME: a fake timestamp.
            stereo = StereoData(
                T_BS     = pp.identity_SE3(1),
                K        = self.K,
                baseline = torch.tensor([self.baseline]),
                width    = self.width,
                height   = self.height,
                time_ns  = [local_index * 1000],   # FIXME: a fake timestamp.
                imageL   = self.ImageL[index],
                imageR   = self.ImageR[index]
            )
        )
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root": lambda s: isinstance(s, str),
            "fx"  : lambda v: isinstance(v, float),
            "fy"  : lambda v: isinstance(v, float),
            "cx"  : lambda v: isinstance(v, float),
            "cy"  : lambda v: isinstance(v, float),
            "bl"  : lambda v: isinstance(v, float),
            "width": lambda v: isinstance(v, int),
            "height": lambda v: isinstance(v, int)
        })  


class ZedMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path, left: bool = True) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"

        def natural_sort_key(s):
            """Create a list of integer and non-integer substrings to enable natural alphanumeric sorting."""
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', os.path.basename(s))]
        self.file_names = []
        if left:
            self.file_names = glob.glob(str(self.directory) + "/rgb_l/*.png") + glob.glob(str(self.directory) + "/rgb_l/*.jpg")
        else:
            self.file_names = glob.glob(str(self.directory) + "/rgb_r/*.png") + glob.glob(str(self.directory) + "/rgb_r/*.jpg")
        
        self.file_names = sorted(self.file_names, key=natural_sort_key)
            
        self.length = len(self.file_names)
        assert self.length > 0, f"No file with '.png' suffix is found under {self.directory}"

    @staticmethod
    def load_png_format(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        image = self.load_png_format(self.file_names[index])
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.
        return image


if __name__ == "__main__":
    import argparse
    from Utility.Config import load_config
    from torch.utils.data import DataLoader

    args = argparse.ArgumentParser()
    args.add_argument("--data", default="Config/Sequence/Zed/kit2kit_manual_dyna.yaml", type=str)
    args = args.parse_args()
    datacfg, datacfg_dict = load_config(Path(args.data))
    dataset = SequenceBase[StereoFrame].instantiate(**vars(datacfg)).clip(16, 32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    
    batch: StereoFrame
    for batch in dataloader:
        print(batch.stereo.imageL.shape, batch.stereo.imageR.shape, batch.idx, batch.stereo.K.shape)
