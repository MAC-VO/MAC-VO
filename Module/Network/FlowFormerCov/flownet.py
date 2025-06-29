import torch
from collections import OrderedDict

from ..FlowFormer.core.utils import InputPadder
from ..FlowFormer.core.transformer import FlowFormer
from .covhead import MemoryCovDecoder


class FlowFormerCov(FlowFormer):
    def __init__(self, cfg, encoder_dtype: torch.dtype=torch.float32, decoder_dtype: torch.dtype=torch.float32):
        super(FlowFormerCov, self).__init__(cfg)
        self.memory_decoder = MemoryCovDecoder(self.cfg, decoder_dtype)
        
        self.enc_dtype = encoder_dtype
        self.context_encoder = self.context_encoder.to(dtype=self.enc_dtype) 
        self.memory_encoder  = self.memory_encoder.to(dtype=self.enc_dtype)

    def forward(self, image1, image2):
        image1 = ((2 * image1) - 1.0).to(dtype=self.enc_dtype)
        image2 = ((2 * image2) - 1.0).to(dtype=self.enc_dtype)

        with torch.cuda.nvtx.range("Context Encoder"):
            context = self.context_encoder(image1)

        with torch.cuda.nvtx.range("Memory Encoder"):
            cost_memory, cost_maps = self.memory_encoder(image1, image2, context)
            cost_maps = cost_maps.float()
            context   = context.float()

        with torch.cuda.nvtx.range("Memory Decoder"):
            flow_predictions, cov_predictions = self.memory_decoder(cost_memory, context, cost_maps)

        return flow_predictions, cov_predictions
    
    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image1: torch.Tensor, image2: torch.Tensor):
        padder = InputPadder(image1.shape)
        image1, image2    = padder.pad(image1, image2)
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

