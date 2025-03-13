from functools import wraps
from Utility.PrettyPrint import Logger
import typing as T
import pypose as pp
import numpy  as np
import torch

try:
    import rerun as rr
except ImportError:
    rr = None


T_Mode    = T.Literal["none", "rerun"]
T_Input   = T.ParamSpec("T_Input")
T_Output  = T.TypeVar("T_Output")


# NOTE: Since rerun does not ensure compatibilty between different versions,
#       We explicitly constrain the version of rerun sdk
if (rr is not None) and (rr.__version__ <= "0.20.0"):
    Logger.write("warn", f"Please re-install rerun_sdk to have version of 0.20.0. Current version is {rr.__version__}")
    rr = None

class Rerun_Visualizer:    
    func_mode: T.ClassVar[dict[str, T_Mode | T.Literal["default"]]] = dict()
    default_mode: T.ClassVar[T_Mode] = "none"
    
    @staticmethod
    def init_connect(application_id: str):
        assert rr is not None, "Can't initialize rerun since rerun is not installed or have incorrect version."
        rr.init(application_id, spawn=True)
        rr.connect_tcp()
        rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
    
    @staticmethod
    def init_save(application_id: str, save_rrd: str):
        assert rr is not None, "Can't initialize rerun since rerun is not installed or have incorrect version."
        rr.init(application_id, spawn=True)
        rr.save(save_rrd)
        rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)

    @staticmethod
    def set_fn_mode(func: T.Callable[T.Concatenate[str, T_Input], None], mode: T_Mode | T.Literal["default"]):
        Rerun_Visualizer.func_mode[func.__name__] = mode

    @staticmethod
    def get_fn_mode(func: T.Callable[T.Concatenate[str, T_Input], None]) -> T_Mode:
        assert func.__name__ in Rerun_Visualizer.func_mode
        func_mode = Rerun_Visualizer.func_mode[func.__name__]
        if func_mode == "default": return Rerun_Visualizer.default_mode
        return func_mode
    
    @staticmethod
    def register(func: T.Callable[T.Concatenate[str, T_Input], None]) -> T.Callable[T.Concatenate[str, T_Input], None]:
        @wraps(func)
        def implement(rerun_path: str, *args: T_Input.args, **kwargs: T_Input.kwargs) -> None:
            if func.__name__ not in Rerun_Visualizer.func_mode:
                Rerun_Visualizer.func_mode[func.__name__] = "default"
            
            func_mode = Rerun_Visualizer.get_fn_mode(func)
            match func_mode:
                case "none": return None
                case "rerun": func(rerun_path, *args, **kwargs)
        return implement

    @register
    @staticmethod
    def log_trajectory(rerun_path: str, trajectory: pp.LieTensor | torch.Tensor, **kwargs):
        assert rr is not None
        if not isinstance(trajectory, pp.LieTensor):
            trajectory = pp.SE3(trajectory)
        
        position = trajectory.translation().detach().cpu().numpy()
        from_pos = position[:-1]
        to_pos = position[1:]
        rr.log(rerun_path, rr.LineStrips3D(np.stack([from_pos, to_pos], axis=1), **kwargs))

    @register
    @staticmethod
    def log_camera(rerun_path: str, pose: pp.LieTensor | torch.Tensor, K: torch.Tensor, **kwargs):
        assert rr is not None
        cx = K[0][2].item()
        cy = K[1][2].item()
        
        if not isinstance(pose, pp.LieTensor):
            pose = pp.SE3(pose)
        frame_position = pose.translation().detach().cpu().numpy()
        frame_rotation = pose.rotation().detach().cpu().numpy()

        rr.log(
            "/".join(rerun_path.split("/")[:-1]),
            rr.Transform3D(
                translation=frame_position,
                rotation=rr.datatypes.Quaternion(xyzw=frame_rotation),
            ),
        )
        rr.log(
            rerun_path,
            rr.Pinhole(
                resolution=[cx * 2, cy * 2],
                image_from_camera=K.detach().cpu().numpy(),
                camera_xyz=rr.ViewCoordinates.FRD,
                image_plane_distance=0.25
            ),
        )

    @register
    @staticmethod
    def log_points(rerun_path: str, position: torch.Tensor, color: torch.Tensor | None, cov_Tw: torch.Tensor | None, cov_mode: T.Literal["none", "axis", "sphere", "color"]="sphere"):
        assert rr is not None
        rr.log(
            rerun_path, 
            rr.Points3D(positions=position, colors=color.detach().cpu().numpy() if (color is not None) else None)
        )
        
        match cov_Tw, cov_mode:
            case None, _: return
            case _, "none": return
            case _, "axis":
                eigen_val, eigen_vec = torch.linalg.eig(cov_Tw)
                eigen_val, eigen_vec = eigen_val.real, eigen_vec.real

                delta = position.repeat(1, 3, 1).reshape(-1, 3)
                eigen_vec_Tw = eigen_vec.transpose(-1, -2).reshape(-1, 3)
                eigen_val = eigen_val.unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3)
                eigen_vec_Tw = eigen_vec_Tw * eigen_val.sqrt()
                eigen_vec_Tw_a = delta + .1 * eigen_vec_Tw
                eigen_vec_Tw_b = delta - .1 * eigen_vec_Tw
                rr.log(
                    rerun_path + "/cov",
                    rr.LineStrips3D(
                        torch.stack([eigen_vec_Tw_a, eigen_vec_Tw_b], dim=1).numpy(),
                        radii=[0.003],
                        colors=color.unsqueeze(0).repeat(3, 1, 1).reshape(-1, 3) if (color is not None) else None
                    ),
                )
            case _, "sphere":
                radii  = (cov_Tw.det().sqrt() * 1e2).clamp(min=0.03, max=0.5)
                rr.log(
                    rerun_path + "/cov", 
                    rr.Points3D(positions=position, colors=color.detach().cpu().numpy() if (color is not None) else None,
                                radii=radii)
                )
            case _, "color":
                import matplotlib.pyplot as plt
                from matplotlib.colors import Normalize
                cov_value = cov_Tw.det()
                cov_det_normalized = Normalize(vmin=0, vmax=cov_value.quantile(0.99).item())(cov_value)
                colormap = plt.cm.plasma    #type: ignore
                c = colormap(cov_det_normalized)[..., :3]        
                rr.log(rerun_path + "/cov", rr.Points3D(position, colors=c))

    @register
    @staticmethod
    def log_image(rerun_path: str, image: torch.Tensor | np.ndarray):
        assert rr is not None
        if isinstance(image, torch.Tensor): np_image = image.cpu().numpy()
        else: np_image = image
        
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        rr.log(rerun_path, rr.Image(np_image).compress())
