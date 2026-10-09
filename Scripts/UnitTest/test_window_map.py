import unittest
import torch
import pypose as pp
from types import SimpleNamespace
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from Odometry.WindowMACVO import reanchor_points, WindowMACVO
from Module.Optimization.WindowICP import WindowResult, EdgeWindow
from Module.MapProcessor import MotionInterpolate


class ReanchorTests(unittest.TestCase):
    def test_updates_owned_points_and_covariance_only(self):
        self.check_reanchor(1.0)

    def test_reanchoring_normalizes_roundoff_before_rotation(self):
        self.check_reanchor(1.00005)

    def check_reanchor(self, quaternion_scale):
        graph = VisualMap()
        for i in range(2):
            graph.frames.push(FrameNode.init({
                "pose": pp.identity_SE3(1), "K": torch.eye(3)[None],
                "baseline": torch.tensor([.1]), "T_BS": pp.identity_SE3(1),
                "need_interp": torch.tensor([False]), "time_ns": torch.tensor([i]),
            }))
        points = torch.tensor([[2., 1., 0.], [3., 0., 1.]])
        cov = torch.tensor([[[1., .2, 0.], [.2, .5, 0.], [0., 0., .1]]]).repeat(2, 1, 1).double()
        pidx = graph.points.push(PointNode.init({"pos_Tw": points.clone(), "cov_Tw": cov.clone(),
                                               "color": torch.zeros(2, 3, dtype=torch.uint8)}))
        fields = {k: torch.zeros((2,) + tuple(v.shape[1:]), dtype=v.dtype)
                  for k, v in graph.match.data.items()}
        midx = graph.match.push(MatchObs.init(fields))
        graph.match2point.set(midx, pidx)
        graph.match2frame1.set(midx, torch.tensor([0, 1]))
        graph.frame2match.add(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        graph.frame2match.add(torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
        before = pp.identity_SE3(1).tensor().double()
        before[..., 3:] *= quaternion_scale
        after = pp.se3(torch.tensor([[.1, .2, 0., 0., 0., .3]], dtype=torch.float64)).Exp().tensor()
        reanchor_points(graph, [0], before, after)
        transform = pp.SE3(after[0])
        self.assertTrue(torch.allclose(graph.points.data["pos_Tw"][0].double(),
                                       transform.Act(points[0].double()), atol=1e-6))
        R = transform.rotation().matrix()
        self.assertTrue(torch.allclose(graph.points.data["cov_Tw"][0], R @ cov[0] @ R.T))
        self.assertTrue(torch.equal(graph.points.data["pos_Tw"][1], points[1]))
        self.assertTrue(torch.equal(graph.points.data["cov_Tw"][1], cov[1]))

    def test_skip_recovered_poses_survive_termination(self):
        system = WindowMACVO.__new__(WindowMACVO)
        system.graph = VisualMap()
        x = [0., .1, .2, .8, .9, 1., 1.1]
        for i, position in enumerate(x):
            pose = pp.identity_SE3(1)
            pose[0, 0] = position
            system.graph.frames.push(FrameNode.init({
                "pose": pose, "K": torch.eye(3)[None], "baseline": torch.tensor([.1]),
                "T_BS": pp.identity_SE3(1), "need_interp": torch.tensor([i == 3]),
                "time_ns": torch.tensor([i]),
            }))
        # Model an already connected window solve recovering frame 3 via skip edges.
        ids = [1, 2, 3, 4, 5]
        solved = system.graph.frames.data["pose"][torch.tensor(ids)].double().clone()
        system._apply_refinement(ids, solved.clone(), WindowResult(solved, {"accepted_steps": 1}))
        system.Optimizer = SimpleNamespace(optimize_res=None, terminate=lambda: None)
        system.MapRefiner = MotionInterpolate(SimpleNamespace())
        system.frame_cache = {}
        system.edge_window = EdgeWindow(5)
        system.terminated = False
        system.terminate()
        saved = system.graph.frames.data["pose"].tensor[:, 0]
        self.assertTrue(torch.allclose(saved, torch.tensor(x), atol=1e-6))
        self.assertFalse(system.graph.frames.data["need_interp"][3].item())


if __name__ == "__main__":
    unittest.main()
