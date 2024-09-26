import cv2
import numpy as np
import torch


def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5) / fx
    hh = (hh.astype(np.float32) - oy + 0.5) / fy
    intrinsicLayer = np.stack((ww, hh)).transpose((1, 2, 0))
    return intrinsicLayer

def make_device_intrinsic_layer(w, h, fx, fy, ox, oy, device: torch.device):
    ww, hh = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing="ij")
    ww = (ww.float() - ox + 0.5) / fx
    hh = (hh.float() - oy + 0.5) / fy
    intrinsicLayer = torch.stack((hh, ww), dim=-1)
    return intrinsicLayer

# Data pre-process components used in TartanVO, copied here to ensure
# consistency with original implementation

KEY2DIM = {'flow': 3, 'img0': 3, 'img1': 3, 'img0_norm': 3, 'img1_norm': 3,
           'intrinsic': 3, 'fmask': 2, 'disp0': 2, 'disp1': 2, 'depth0': 2,
           'depth1': 2, 'flow_unc': 2, 'depth0_unc': 2}


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class CropCenter(object):
    """Crops a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    if fix_ratio is False, w and h are resized separatedly
    if scale_w is given, w will be resized accordingly
    """

    def __init__(self, size, fix_ratio=True, scale_w=1.0, scale_disp=False):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fix_ratio = fix_ratio
        self.scale_w = scale_w
        self.scale_disp = scale_disp

    def __call__(self, sample):

        th, tw = self.size
        hh, ww = get_sample_dimention(sample)
        if ww == tw and hh == th:
            return sample
        # import ipdb;ipdb.set_trace()
        # resize the image if the image size is smaller than the target size
        scale_h = max(1, float(th) / hh)
        scale_w = max(1, float(tw) / ww)
        if scale_h > 1 or scale_w > 1:
            if self.fix_ratio:
                scale_h = max(scale_h, scale_w)
                scale_w = max(scale_h, scale_w)
            w = int(round(ww * scale_w))  # w after resize
            h = int(round(hh * scale_h))  # h after resize
        else:
            w, h = ww, hh

        if self.scale_w != 1.0:
            scale_w = self.scale_w
            w = int(round(ww * scale_w))

        if scale_h != 1. or scale_w != 1.:  # resize the data
            resizedata = ResizeData(size=(h, w), scale_disp=self.scale_disp)
            sample = resizedata(sample)

        x1 = int((w - tw) / 2)
        y1 = int((h - th) / 2)
        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen):
                datalist.append(sample[kk][k][y1:y1 + th, x1:x1 + tw, ...])
            sample[kk] = datalist

        return sample


def get_sample_dimention(sample):
    for kk in sample.keys():
        if kk in KEY2DIM:  # for sequencial data
            h, w = sample[kk][0].shape[0], sample[kk][0].shape[1]
            return h, w
    assert False, "No image type in {}".format(sample.keys())


class ResizeData(object):
    """Resize the data in a dict
    """

    def __init__(self, size, scale_disp=False):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_disp = scale_disp

    def resize_seq(self, dataseq, w, h):
        seqlen = dataseq.shape[0]
        datalist = []
        for k in range(seqlen):
            datalist.append(
                cv2.resize(dataseq[k], (w, h), interpolation=cv2.INTER_LINEAR))
        return np.stack(datalist, axis=0)

    def __call__(self, sample):
        th, tw = self.size
        h, w = get_sample_dimention(sample)
        if w == tw and h == th:
            return sample
        scale_w = float(tw) / w
        scale_h = float(th) / h

        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen):
                datalist.append(cv2.resize(sample[kk][k], (tw, th),
                                           interpolation=cv2.INTER_LINEAR))
            sample[kk] = datalist

        if 'flow' in sample:
            for k in range(len(sample['flow'])):
                sample['flow'][k][..., 0] = sample['flow'][k][..., 0] * scale_w
                sample['flow'][k][..., 1] = sample['flow'][k][..., 1] * scale_h

        if self.scale_disp:  # scale the depth
            if 'disp0' in sample:
                for k in range(len(sample['disp0'])):
                    sample['disp0'][k] = sample['disp0'][k] * scale_w
            if 'disp1' in sample:
                for k in range(len(sample['disp1'])):
                    sample['disp1'][k] = sample['disp1'][k] * scale_w
        else:
            sample['scale_w'] = np.array([scale_w],
                                         dtype=np.float32)  # used in e2e-stereo-vo

        return sample


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size
    This function won't resize the RGBs
    flow/disp values will NOT be changed

    """

    def __init__(self, scale=4):
        """
        size: output frame size, this should be NO LARGER than the input frame size!
        """
        self.downscale = 1.0 / scale
        # self.key2dim = {'flow':3, 'intrinsic':3, 'fmask':2, 'disp0':2, 'disp1':2} # these data have 3 dimensions

    def __call__(self, sample):
        if self.downscale == 1:
            return sample

        # import ipdb;ipdb.set_trace()
        for key in sample.keys():
            if key in {'flow', 'intrinsic', 'fmask', 'disp0', 'depth0'}:
                imgseq = []
                for k in range(len(sample[key])):
                    imgseq.append(cv2.resize(sample[key][k],
                                             (0, 0), fx=self.downscale,
                                             fy=self.downscale,
                                             interpolation=cv2.INTER_LINEAR))
                sample[key] = imgseq

        return sample


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    This option should be before the to tensor
    """

    def __init__(self, mean, std, rgbbgr=False, keep_old=False):
        """
        keep_old: keep both normalized and unnormalized data,
        normalized data will be put under new key xxx_norm
        """
        self.mean = mean
        self.std = std
        self.rgbbgr = rgbbgr
        self.keep_old = keep_old

    def __call__(self, sample):
        keys = list(sample.keys())
        for kk in keys:
            if kk.startswith('img0') or kk.startswith('img1'):  # sample[kk] is a list, sample[kk][k]: h x w x 3
                seqlen = len(sample[kk])
                datalist = []
                for s in range(seqlen):
                    sample[kk][s] = sample[kk][s]
                    if self.rgbbgr:
                        img = sample[kk][s][..., [2, 1, 0]]  # bgr2rgb
                    if self.mean is not None and self.std is not None:
                        img = np.zeros_like(sample[kk][s])
                        for k in range(3):
                            img[..., k] = (sample[kk][s][..., k] - self.mean[k]) / self.std[k]
                    else:
                        img = sample[kk][s]
                    datalist.append(img)

                if self.keep_old:
                    sample[kk + '_norm'] = datalist
                else:
                    sample[kk] = datalist
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # import ipdb;ipdb.set_trace()
        for kk in sample.keys():
            if not kk in KEY2DIM:
                continue
            if KEY2DIM[kk] == 3:  # for sequencial data
                data = np.stack(sample[kk], axis=0)
                data = data.transpose((0, 3, 1, 2))  # frame x channel x h x w
            elif KEY2DIM[kk] == 2:  # for sequencial data
                data = np.stack(sample[kk], axis=0)
                data = data[:, np.newaxis, :, :]  # frame x channel x h x w
            else:
                raise ValueError(f"Unexpected key {kk} in provided sample")

            data = data.astype(np.float32)
            sample[kk] = torch.from_numpy(data)  # copy to make memory continuous

        return sample
