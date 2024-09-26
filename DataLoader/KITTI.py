import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

from .Interface import MetaInfo, SourceDataFrame
from .SequenceBase import GenericSequence

EDN2NED = pp.from_matrix(torch.tensor([
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
]), pp.SE3_type)
NED2EDN = EDN2NED.Inv()

class KITTIMonocularDataset(Dataset):
    def __init__(self, image_dir: Path, target_size: list[int]) -> None:
        super().__init__()
        self.file_names = sorted([file for file in image_dir.glob("*.png")])
        self.length = len(self.file_names)
        
        self.orig_width  = 1240
        self.orig_height = 376
        
        
        if len(target_size) == 0:
            # Crop to even shape to make it work on many DL models (with pooling layer).
            self.crop = transforms.Resize((376, 1240))
            self.width = self.orig_width
            self.height = self.orig_height
        else:
            self.width = target_size[1]
            self.height = target_size[0]
            self.crop = transforms.CenterCrop(target_size)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
        image_tensor /= 255.
        
        return self.crop(image_tensor)


def loadKITTIGTPoses(pose_dir: Path) -> pp.LieTensor:
    poses = torch.tensor(np.loadtxt(pose_dir)).reshape((-1, 3, 4))
    
    padding = torch.tensor([[[0., 0., 0., 1.]]]).repeat(poses.size(0), 1, 1)
    poses = torch.cat([poses, padding], dim=1)
    
    # System: NED
    # KITTI : EDN
    poses = pp.from_matrix(poses, pp.SE3_type).float()
    poses = EDN2NED @ poses @ NED2EDN
    
    return poses



class KITTISequence(GenericSequence):
    def __init__(self, root: str, fx: float, fy: float, cx: float, cy: float, bl: float,
                 target_size: list[int],
                 gtFlow: bool, gtDepth: bool, gtPose: bool, **_):
        assert not gtFlow, "KITTI has no gtFlow provided."
        assert not gtDepth, "KITTI has no gtDepth provided."
        
        self.root = Path(root)
        self.sequence_name = self.root.name
        
        self.imageL = KITTIMonocularDataset(Path(self.root, "image_2"), target_size)
        self.imageR = KITTIMonocularDataset(Path(self.root, "image_3"), target_size)
        assert len(self.imageL) == len(self.imageR)
        
        if gtPose:
            self.gtPose_data = loadKITTIGTPoses(Path(self.root.parent.parent, "poses", self.sequence_name + ".txt"))
        else:
            self.gtPose_data = None
        
        if len(target_size) != 0:
            cx = (cx - (self.imageL.orig_width - self.imageL.width) / 2)
            cy = (cy - (self.imageL.orig_height - self.imageL.height) / 2)
        

        self.K=torch.tensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0., 1.]])
        self.g=9.81
        self.baseline=bl
        self.width=self.imageL.width
        self.height=self.imageL.height
        self.gtFlow = gtFlow
        self.gtPose = gtPose
        self.gtDepth = gtDepth
            
        self.length = len(self.imageL)
        super().__init__(self.length)
    
    @staticmethod
    def SequenceType() -> str: return "KITTI"
    
    def __getitem__(self, local_index: int) -> SourceDataFrame:
        index = self.get_index(local_index)
        return SourceDataFrame(
            MetaInfo(idx = local_index, 
                     K = self.K, 
                     baseline = self.baseline, 
                     width = self.width, 
                     height = self.height),
            self.imageL[index],
            self.imageR[index],
            None,
            None,
            None,
            None,
            self.gtPose_data[index] if self.gtPose_data is not None else None #type: ignore
        )


if __name__ == "__main__":
    import argparse
    from Utility.Visualizer import PLTVisualizer
    from Utility.Config import load_config
    from torch.utils.data import DataLoader
    PLTVisualizer.setup(PLTVisualizer.State.SAVE_FILE, Path("./Results"))

    args = argparse.ArgumentParser()
    args.add_argument("--data", default="Config/Sequence/KITTI/KITTI_00-02.yaml", type=str)
    args = args.parse_args()
    datacfg, datacfg_dict = load_config(Path(args.data))
    dataset = GenericSequence[SourceDataFrame].instantiate(**vars(datacfg))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False ,num_workers=1, collate_fn=dataset.collate_fn)             
    
    for i, frame in enumerate(dataloader):
        PLTVisualizer.visualize_stereo("stereo", frame.imageL, frame.imageR)
        if i == 10:
            break
