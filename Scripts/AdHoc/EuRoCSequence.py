from pathlib import Path

from DataLoader import EuRoCSequence
from Utility.Visualizer import RerunVisualizer
from Utility.PrettyPrint import ColoredTqdm

RerunVisualizer.setup("EuRoC Test", False, Path("Results", "rrvis.rrd"), useRR=True)
seq = EuRoCSequence(Path("/data2/datasets/yuhengq/Euroc/V2_03_difficult/mav0"), False, False, True, useIMU=False).clip(0, 100)
assert seq.gtPose_data is not None

RerunVisualizer.visualizePath("/map/Trajectory", seq.gtPose_data, colors=(244, 97, 221), radii=0.01)

for frame in ColoredTqdm(seq):
    assert frame.gtPose is not None
    
    frame_position = frame.gtPose.translation()
    frame_rotation = frame.gtPose.rotation()
    RerunVisualizer.visualizeAnnotatedCamera("/map/Frame", seq.K, frame.gtPose)
    RerunVisualizer.visualizeImageOnCamera(frame.imageL[0].permute(1, 2, 0))
