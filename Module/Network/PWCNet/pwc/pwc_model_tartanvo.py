"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

Adapted in TartanVO for uncertainty estimation module.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import FunctionCorrelation


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def predict_uncertainty(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, int(in_planes / 2), kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(int(in_planes / 2), int(in_planes / 4), kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.LeakyReLU(0.1),
        nn.Conv2d(int(in_planes / 4), 1, kernel_size=3, stride=1,
                  padding=1, bias=True),
        # nn.Sigmoid()
    )


def predict_flow(in_planes, uncertainty=False):
    pred = nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)
    return pred


class PredictFlow(nn.Module):
    def __init__(self, in_planes, uncertainty=False):
        super(PredictFlow, self).__init__()

        self.uncertainty = uncertainty
        self.pred = nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)
        if uncertainty:
            self.unc = predict_uncertainty(in_planes)

    def forward(self, x):
        if self.uncertainty:
            return self.pred(x), self.unc(x)
        else:
            return self.pred(x), None


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PWCDCNet_Adapted(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, md=4, flow_norm=20.0, uncertainty=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet_Adapted, self).__init__()

        self.flow_norm = flow_norm
        self.uncertainty = uncertainty

        if self.uncertainty:
            predlayer = PredictFlow
        else:
            predlayer = predict_flow

        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)

        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        featnum = 4
        if self.uncertainty:
            featnum += 1

        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predlayer(od + dd[4], self.uncertainty)
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 128 + featnum
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predlayer(od + dd[4], self.uncertainty)
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 96 + featnum
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predlayer(od + dd[4], self.uncertainty)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 64 + featnum
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predlayer(od + dd[4], self.uncertainty)
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 32 + featnum
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predlayer(od + dd[4], self.uncertainty)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predlayer(32, self.uncertainty)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size()).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def multi_scale_conv(self, conv0_func, conv1_func, conv2_func, conv3_func, conv4_func, input_feat):
        x = torch.cat((conv0_func(input_feat), input_feat), 1)
        x = torch.cat((conv1_func(x), x), 1)
        x = torch.cat((conv2_func(x), x), 1)
        x = torch.cat((conv3_func(x), x), 1)
        x = torch.cat((conv4_func(x), x), 1)
        return x

    def concate_two_layers(self, pred_func, decon_func, upfeat_func, feat_high, feat_low1, feat_low2, scale):
        if self.uncertainty:
            flow_high, flow_uncertan = pred_func(feat_high)
        else:
            flow_high, flow_uncertan = pred_func(feat_high), None
        up_flow_high = decon_func(flow_high)
        up_feat_high = upfeat_func(feat_high)

        warp_feat = self.warp(feat_low2, up_flow_high * scale)
        corr_low = FunctionCorrelation(tenFirst=feat_low1, tenSecond=warp_feat)
        corr_low = self.leakyRELU(corr_low)
        x = torch.cat((corr_low, feat_low1, up_flow_high, up_feat_high), 1)

        if flow_uncertan is not None:
            up_flow_uncertain = F.upsample(flow_uncertan, (up_feat_high.shape[2], up_feat_high.shape[3]), mode='bilinear')
            x = torch.cat((x, up_flow_uncertain), 1)

        return x, flow_high, flow_uncertan

    def forward(self, x):
        im1 = x[:, 0:3, ...]
        im2 = x[:, 3:6, ...]

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        # corr6 = self.corr(c16, c26)
        corr6 = FunctionCorrelation(tenFirst=c16, tenSecond=c26)
        corr6 = self.leakyRELU(corr6)

        x = self.multi_scale_conv(self.conv6_0, self.conv6_1, self.conv6_2, self.conv6_3, self.conv6_4, corr6)
        x, flow6, flow6_uc = self.concate_two_layers(self.predict_flow6, self.deconv6, self.upfeat6, x, c15, c25, 0.625)

        x = self.multi_scale_conv(self.conv5_0, self.conv5_1, self.conv5_2, self.conv5_3, self.conv5_4, x)
        x, flow5, flow5_uc = self.concate_two_layers(self.predict_flow5, self.deconv5, self.upfeat5, x, c14, c24, 1.25)

        x = self.multi_scale_conv(self.conv4_0, self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, x)
        x, flow4, flow4_uc = self.concate_two_layers(self.predict_flow4, self.deconv4, self.upfeat4, x, c13, c23, 2.5)

        x = self.multi_scale_conv(self.conv3_0, self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, x)
        x, flow3, flow3_uc = self.concate_two_layers(self.predict_flow3, self.deconv3, self.upfeat3, x, c12, c22, 5.0)

        x = self.multi_scale_conv(self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3, self.conv2_4, x)

        if self.uncertainty:
            flow2, flow2_uc = self.predict_flow2(x)
        else:
            flow2, flow2_uc = self.predict_flow2(x), None

        # import ipdb;ipdb.set_trace()

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        if self.uncertainty:
            refine, refine_uc = self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        else:
            refine, refine_uc = self.dc_conv7(self.dc_conv6(self.dc_conv5(x))), None
        flow2 = flow2 + refine

        if self.uncertainty:
            flow2_uc = torch.log(torch.exp(flow2_uc) + torch.exp(refine_uc))

        # if self.training:
        return (flow2, flow3, flow4, flow5, flow6), \
            (flow2_uc, flow3_uc, flow4_uc, flow5_uc, flow6_uc)
        # else:
        #     return flow2

    def scale_targetflow(self, targetflow, small_scale=False):
        """
        calculte GT flow in different scales
        """
        if small_scale:
            target4 = targetflow
        else:
            target4 = F.interpolate(targetflow, scale_factor=0.25, mode='bilinear', align_corners=False)  # /4.0
        target8 = F.interpolate(target4, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        target16 = F.interpolate(target8, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        target32 = F.interpolate(target16, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        target64 = F.interpolate(target32, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        return [target4, target8, target16, target32, target64]

    def scale_mask(self, mask, threshold=128, small_scale=False):
        """
        threshold: deperated
        in tarranair,
        mask=0:   valid_mask -  True
        mask=1:   CROSS_OCC -   False
        mask=10:  SELF_OCC -    True
        mask=100: OUT_OF_FOV -  True
        """
        if small_scale:
            mask4 = mask
        else:
            mask4 = F.interpolate(mask, scale_factor=0.25, mode='bilinear', align_corners=False)  # /4.0
        mask8 = F.interpolate(mask4, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        mask16 = F.interpolate(mask8, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        mask32 = F.interpolate(mask16, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        mask64 = F.interpolate(mask32, scale_factor=0.5, mode='bilinear', align_corners=False)  # /2.0
        mask4 = (mask4 < 0.5) | (mask4 > 1)  # only mask out cross_occ # mask4<threshold
        mask8 = (mask8 < 0.5) | (mask8 > 1)  # only mask out cross_occ # mask8<threshold
        mask16 = (mask16 < 0.5) | (mask16 > 1)  # only mask out cross_occ # mask16<threshold
        mask32 = (mask32 < 0.5) | (mask32 > 1)  # only mask out cross_occ # mask32<threshold
        mask64 = (mask64 < 0.5) | (mask64 > 1)  # only mask out cross_occ # mask64<threshold
        return [mask4, mask8, mask16, mask32, mask64]

    def calc_one_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
        if mask is not None:
            output_ = output[mask]
            target_ = target[mask]
        else:
            output_ = output
            target_ = target

        if unc is None:
            return criterion(output_, target_)
        else:  # if using uncertainty, then no mask
            diff = torch.abs(output - target)  # hard code L1 loss
            loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
            return loss_unc / (1.0 + lamb)

    def get_loss_old(self, output, target, criterion, small_scale=False):
        """
        return flow loss
        """
        if self.training:
            target4, target8, target16, target32, target64 = self.scale_targetflow(target, small_scale)
            loss1 = self.calc_one_loss(output[0], target4, criterion, mask=None, unc=output[5], lamb=1.0)  # criterion(output[0], target4)
            loss2 = self.calc_one_loss(output[1], target8, criterion, mask=None, unc=output[6], lamb=1.0)  # criterion(output[1], target8)
            loss3 = self.calc_one_loss(output[2], target16, criterion, mask=None, unc=output[7], lamb=1.0)  # criterion(output[2], target16)
            loss4 = self.calc_one_loss(output[3], target32, criterion, mask=None, unc=output[8], lamb=1.0)  # criterion(output[3], target32)
            loss5 = self.calc_one_loss(output[4], target64, criterion, mask=None, unc=output[9], lamb=1.0)  # criterion(output[4], target64)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5.0
        else:
            if small_scale:
                output4 = output[0]
                unc4 = output[5]
            else:
                output4 = F.interpolate(output[0], scale_factor=4, mode='bilinear', align_corners=False)  # /4.0
                if output[5] is not None:
                    unc4 = F.interpolate(output[5], scale_factor=4, mode='bilinear', align_corners=False)
                else:
                    unc4 = None
            loss = self.calc_one_loss(output4, target, criterion, mask=None, unc=unc4, lamb=1.0)  # criterion(output4, target)
        return loss

    def get_loss_w_mask_old(self, output, target, criterion, mask, small_scale=False):
        """
        return flow loss
        small_scale: True - the target and mask are of the same size with output
                     False - the target and mask are of 4 time size of the output
        """
        if self.training:  # PWCNet + training
            target4, target8, target16, target32, target64 = self.scale_targetflow(target, small_scale)
            mask4, mask8, mask16, mask32, mask64 = self.scale_mask(mask, small_scale=small_scale)  # only consider coss occlution which indicates moving objects
            mask4 = mask4.expand(target4.shape)
            mask8 = mask8.expand(target8.shape)
            mask16 = mask16.expand(target16.shape)
            mask32 = mask32.expand(target32.shape)
            mask64 = mask64.expand(target64.shape)
            loss1 = criterion(output[0][mask4], target4[mask4])
            loss2 = criterion(output[1][mask8], target8[mask8])
            loss3 = criterion(output[2][mask16], target16[mask16])
            loss4 = criterion(output[3][mask32], target32[mask32])
            loss5 = criterion(output[4][mask64], target64[mask64])
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5.0
        else:
            if small_scale:
                output4 = output[0]
            else:
                output4 = F.interpolate(output[0], scale_factor=4, mode='bilinear', align_corners=False)  # /4.0
            valid_mask = mask < 10
            valid_mask = valid_mask.expand(target.shape)
            loss = criterion(output4[valid_mask], target[valid_mask])
        return loss

    def calc_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
        """
        return flow loss
        small_scale: True - the target and mask are of the same size with output
                     False - the target and mask are of 4 time size of the output
        turnoff_unc: return loss without uncertainty
        """

        if target.shape == output[0].shape:
            small_scale = True
        else:
            small_scale = False
        if self.training:  # PWCNet + training
            targetlist = self.scale_targetflow(target, small_scale)  # target4, target8, target16, target32, target64
            if mask is None:
                masklist = [None, ] * 5  # mask4, mask8, mask16, mask32, mask64
            else:
                # mask4, mask8, mask16, mask32, mask64
                masklist = self.scale_mask(mask, small_scale=small_scale)  # only consider coss occlution which indicates moving objects
                masklist = [mm.expand(tt.shape) for (mm, tt) in zip(masklist, targetlist)]

            losslist = [0, ] * 5
            for k in range(5):
                unc_output = unc[k] if unc is not None else None
                losslist[k] = self.calc_one_loss(output[k], targetlist[k], criterion, mask=masklist[k],
                                                 unc=unc_output)  # criterion(output[0][mask4], target4[mask4])
            loss = (losslist[0] + losslist[1] + losslist[2] + losslist[3] + losslist[4]) / 5.0
            loss_nounc = self.calc_one_loss(output[0], targetlist[0], criterion, mask=masklist[0], unc=unc)
        else:
            if small_scale:
                output4 = output[0]
                unc4 = unc[0] if unc is not None else None
            else:
                output4 = F.interpolate(output[0], scale_factor=4, mode='bilinear', align_corners=False)  # /4.0
                if unc is not None and unc[0] is not None:
                    unc4 = F.interpolate(unc[0], scale_factor=4, mode='bilinear', align_corners=False)
                else:
                    unc4 = None
            if mask is None:
                valid_mask = None
            else:
                valid_mask = mask < 10
                valid_mask = valid_mask.expand(target.shape)
            # if turnoff_unc:
            #     unc4 = None
            loss = self.calc_one_loss(output4, target, criterion, mask=valid_mask, unc=unc4)  # criterion(output4[valid_mask], target[valid_mask])
            loss_nounc = self.calc_one_loss(output4, target, criterion, mask=valid_mask, unc=None)  # criterion(output4[valid_mask], target[valid_mask])
        return loss, loss_nounc
