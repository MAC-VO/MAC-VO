import typing as T
from .Graph import TensorBundle, AutoScalingBundle

# Define storage of interest
FrameFeature = T.Literal[
    "K",            # Nx3x3 , dtype=float32
    "baseline",     # Nx1   , dtype=float32
    "pose",         # Nx7   , dtype=float32, pose of sensor under world frame.
    "T_BS",         # Nx7   , dtype=float32, body-to-sensor SE3 transformation.
    "need_interp",  # Nx1   , dtype=bool
    "time_ns"       # Nx1   , dtype=long
]
MatchingFeature = T.Literal[
    "pixel1_uv",    # Nx2   , dtype=float32
    "pixel1_d",     # Nx1   , dtype=float32
    "pixel2_uv",    # Nx2   , dtype=float32
    "pixel2_d",     # Nx1   , dtype=float32
    "pixel1_disp",  # Nx1   , dtype=float32
    "pixel2_disp",  # Nx1   , dtype=float32
    "pixel1_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel2_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel1_d_cov" ,# Nx1   , dtype=float32
    "pixel2_d_cov" ,# Nx1   , dtype=float32
    "pixel1_disp_cov",    # Nx1   , dtype=float32
    "pixel2_disp_cov",    # Nx1   , dtype=float32
    "obs1_covTc",   # Nx3x3 , dtype=float64
    "obs2_covTc",   # Nx3x3 , dtype=float64
]
PointFeature = T.Literal[
    "pos_Tw",       # Nx3   , dtype=float32
    "cov_Tw",       # Nx3x3 , dtype=float64
    "color" ,       # Nx3   , dtype=uint8
]


FrameNode    = TensorBundle[FrameFeature]
FrameStore   = AutoScalingBundle[FrameFeature]

MatchObs     = TensorBundle[MatchingFeature]
MatchStore   = AutoScalingBundle[MatchingFeature]

PointNode    = TensorBundle[PointFeature]
PointStore   = AutoScalingBundle[PointFeature]
