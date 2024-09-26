import torch
from collections import OrderedDict

from ..FlowFormer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from ..FlowFormer.core.utils.utils import InputPadder
from .covhead import MemoryCovDecoder


class FlowFormerCov(FlowFormer):
    def __init__(self, cfg, device: str, use_inference_jit=False):
        super(FlowFormerCov, self).__init__(cfg, device, use_inference_jit)
        self.memory_decoder = MemoryCovDecoder(self.cfg)

    def forward(self, image1, image2):
        image1 = (2 * image1) - 1.0
        image2 = (2 * image2) - 1.0

        data = {}

        assert not self.cfg.context_concat, "Not supported in this mode."
        context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions, cov_predictions = self.memory_decoder(
            cost_memory, context, data, flow_init=None
        )

        return flow_predictions, cov_predictions
    
    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image1: torch.Tensor, image2: torch.Tensor):
        image1, image2 = image1.to(self.device), image2.to(self.device)
        
        # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_pre, cov_pre = self.forward(image1, image2)

        flow_pre = padder.unpad(flow_pre[0])
        cov_pre = padder.unpad(cov_pre[0])
        return flow_pre, torch.exp(cov_pre * 2)

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        cvt_ckpt = OrderedDict()
        for k in ckpt:
            if k.startswith("module."):
                cvt_ckpt[k[7:]] = ckpt[k]
            else:
                cvt_ckpt[k] = ckpt[k]
        self.load_state_dict(cvt_ckpt, strict=False)

