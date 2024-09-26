import evo.main_ape as main_ape
import evo.main_rpe as main_rpe

from evo.core.trajectory import PosePath3D
from evo.core.metrics import PoseRelation, Unit

NEED_ALIGN_SCALE = {"dpvo", "droid", "tartanvo_mono"}

def evaluateRTE(gt_traj: PosePath3D, est_traj: PosePath3D, correct_scale=False):
    result = main_rpe.rpe(gt_traj, est_traj, 
                          pose_relation=PoseRelation.translation_part,
                          align_origin=True,
                          align=True,
                          correct_scale=correct_scale,
                          delta=1, delta_unit=Unit.frames)
    return result

def evaluateATE(gt_traj: PosePath3D, est_traj: PosePath3D, correct_scale=False):
    result = main_ape.ape(gt_traj, est_traj, 
                          pose_relation=PoseRelation.translation_part,
                          align_origin=True,
                          align=True,
                          correct_scale=correct_scale)
    return result

def evaluateROE(gt_traj: PosePath3D, est_traj: PosePath3D, correct_scale=False):
    """
    Evaluates error of rotation
    """
    # Due to the numerical precision of trajectory file, sometimes this method call
    # may raise evo.LieAlgebraException. In this case, you need to move to 
    #   .../evo/code/lie_algebra.py (Library code)
    # and manually change the atol in is_so3(...) method from 1e-6 (library default)
    # to 1e-4 (or a more relaxed value)
    result = main_rpe.rpe(gt_traj, est_traj,
                          pose_relation=PoseRelation.rotation_angle_deg,
                          align_origin=True,
                          align=True,
                          correct_scale=correct_scale,
                          delta=1, delta_unit=Unit.frames)
    return result

def evaluateRPE(gt_traj: PosePath3D, est_traj: PosePath3D, correct_scale=False):
    """
    Evaluates error of se(3) pose
    """
    result = main_rpe.rpe(gt_traj, est_traj,
                          pose_relation=PoseRelation.full_transformation,
                          align_origin=True, align=True, correct_scale=correct_scale,
                          delta=1, delta_unit=Unit.frames)
    return result
