import torch
import torch.nn as nn
import torch.nn.functional as F


class StereoVONet(nn.Module):
    def __init__(self, flowNormFactor=1.0, stereoNormFactor=1.0, poseDepthNormFactor=0.25):
        """
        flowNormFactor: difference between flownet and posenet
        stereoNormFactor: norm value used in stereo training
        poseDepthNormFactor: distance is normalized in posenet training ()
        """
        super(StereoVONet, self).__init__()

        from Module.Network.PWCNet.pwc import PWCDCNet_Adapted as FlowNet
        from .StereoNet import StereoNet7 as StereoNet
        from .FlowPoseNet import VOFlowRes as FlowPoseNet

        self.flowNet = FlowNet()
        self.stereoNet = StereoNet()
        self.flowPoseNet = FlowPoseNet(config=1, stereo=True, autoDistTarget=0., down_scale=True, out_feature=False, intrinsic=True)
        self.flowNormFactor = flowNormFactor
        self.stereoNormFactor = stereoNormFactor
        self.poseDepthNormFactor = poseDepthNormFactor
    
    def forward_flow(self, x0_flow, x0n_flow):
        inputTensor = torch.cat((x0_flow, x0n_flow), dim=1).contiguous()
        return self.flowNet(inputTensor)[0][0]

    def forward(self, x0_flow, x0n_flow, x0_stereo, x1_stereo, intrin=None, blxfx=80.):
        """
        flow_out: pwcnet: 5 scale predictions up to 1/4 size
                  flownet: 1/1 size prediction
        stereo_out: psmnet: 3 scale predictions up to 1/1 size
                    stereonet: 1/1 size prediction
        scale_w: the x-direction scale factor in data augmentation
        scale_depth: scale input depth and output motion to shift the data distribution
        """
        assert intrin is not None
        stereo_out, _ = self.stereoNet(torch.cat((x0_stereo, x1_stereo), dim=1))
        
        flow_input = self.forward_flow(x0_flow, x0n_flow)

        # scale the disparity size
        stereo = F.interpolate(stereo_out, scale_factor=0.25, mode='bilinear', align_corners=True)

        depth_input = stereo / blxfx / float(self.stereoNormFactor * self.poseDepthNormFactor)

        # scale the disp back, because we treat it as depth
        # What is scale_w doing and where does it come from
        inputTensor = torch.cat((flow_input, depth_input, intrin), dim=1)

        pose = self.flowPoseNet(inputTensor, scale_disp=1.)
        return stereo_out, pose, flow_input, depth_input, intrin
