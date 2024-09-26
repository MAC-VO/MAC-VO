import torch.nn as nn
import pypose.module as pm

from Utility.Utils import StructuralMove


class Integrator(nn.Module):
    def __init__(self, gravity, prop_cov=True, reset=True):
        super().__init__()
        self.integrator = pm.IMUPreintegrator(gravity=gravity, prop_cov=prop_cov, reset=reset)

    def inte_through_graph(self, edges, data, device="cpu"):
        """
        Output:
            IMU inte state: the IMU increments corresponding to the edges
        """
        out_state = {'Dp':[], 'Dr':[], 'Dv':[],'Dt':[]}
        data = StructuralMove(data, device)
        assert isinstance(data, dict)

        for start_i, end_i in edges:
            # i stands for node idx stands for the frame 

            acc = self.integrator._check(data["acc"][start_i:end_i])
            gyro = self.integrator._check(data['gyro'][start_i:end_i])
            dt = self.integrator._check(data['dt'][start_i:end_i])
            rot = self.integrator._check(data['gt_orientation'][start_i:end_i])

            obs = self.integrator.integrate(dt = dt, gyro = gyro, acc = acc, rot = rot)
            obs["acc_cov"] =  self.integrator._check(data["acc_cov"][start_i:end_i])
            obs["gyro_cov"] =  self.integrator._check(data["gyro_cov"][start_i:end_i])
            obs["dt"] =  self.integrator._check(data["dt"][start_i:end_i])
            # save_state(out_state, obs) #TODO

        return out_state
    
    def intefromFrame(self, frameData, init_state):
        assert init_state is not None
        
        acc = self.integrator._check(frameData.imu.acc)
        gyro = self.integrator._check(frameData.imu.gyro)
        dt = self.integrator._check(frameData.imu.dt)
        imu_obs = self.integrator.integrate(
            dt = dt,
            gyro = gyro,
            acc = acc,
            init_rot = init_state["rot"],
        )
        imu_state = self.integrator(
            init_state=init_state,
            dt=dt, 
            gyro=gyro, 
            acc=acc
        )

        return imu_obs, imu_state

