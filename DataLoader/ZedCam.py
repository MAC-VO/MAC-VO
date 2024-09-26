import cv2
import torch
import numpy as np
import glob, re, os
from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset

from .SequenceBase import GenericSequence
from .Interface import MetaInfo, SourceDataFrame


class ZedMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path, transform = None, left: bool = True) -> None:
        super().__init__()
        self.directory = directory
        self.transform = transform
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
        if self.transform:
            image = self.transform(image)

        return image


class ZedSequence(GenericSequence[SourceDataFrame]):
    @classmethod
    def name(cls) -> str: return "Zed"

    def __init__(self, root, 
                 fx: float, fy: float, cx: float, cy: float, bl: float, 
                 width: int, height: int, scaleu: float = 1, scalev: float = 1, 
                 crop: int = 0, **_) -> None:
        self.scaleu=scaleu
        self.scalev=scalev
        self.crop = crop
        self.raw_width = width
        self.raw_height = height
        transformations = []
        
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        
        if (self.scaleu >1) or (self.scalev>1):
            width=int(self.raw_width / self.scaleu)
            height=int(self.raw_height / self.scalev)
            round_scaleu = self.raw_width/width
            round_scalev = self.raw_height/height
            transformations.append(transforms.Resize((height, width)))
            K[0] /= round_scaleu
            K[1] /= round_scalev
        else:
            width = self.raw_width
            height = self.raw_height
        
        if crop > 0:
            assert (width - self.crop) > 0, f"Width is {width}, but trying to crop {self.crop}."
            assert (height - self.crop) > 0, f"Height is {height}, but trying to crop {self.crop}."
            
            K[0, 2] -= (width - self.crop)/2
            K[1, 2] -= (height - self.crop)/2
            height = int(self.crop)
            width = int(self.crop)
            transformations.append(transforms.CenterCrop(int(crop)))
        
        # metadata
        self.seqRoot = Path(root)
        self.K = K
        self.g = 9.81
        self.baseline = 0.119981
        self.width = width
        self.height = height
        
        transformations = transforms.Compose(transformations)

        self.ImageL = ZedMonocularDataset(Path(self.seqRoot), transform=transformations, left=True)
        self.ImageR = ZedMonocularDataset(Path(self.seqRoot), transform=transformations, left=False)

        self.length = len(self.ImageL)
        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> SourceDataFrame:
        index = self.get_index(local_index)
        
        return SourceDataFrame(
            meta=MetaInfo(
                local_index,
                self.K,
                self.baseline,
                width=self.width,
                height=self.height
            ),
            imageL=self.ImageL[index],
            imageR=self.ImageR[index],
            gtDepth=None,
            gtFlow=None,
            flowMask=None,
            gtPose=None,
            imu=None,
        )
    

if __name__ == "__main__":
    import argparse
    from Utility.Config import load_config
    from torch.utils.data import DataLoader

    args = argparse.ArgumentParser()
    args.add_argument("--data", default="Config/Sequence/Zed/kit2kit_manual_dyna.yaml", type=str)
    args = args.parse_args()
    datacfg, datacfg_dict = load_config(Path(args.data))
    dataset = GenericSequence[SourceDataFrame].instantiate(**vars(datacfg)).clip(16, 32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    
    batch: SourceDataFrame
    for batch in dataloader:
        print(batch.imageL.shape, batch.imageR.shape, batch.meta.idx)

