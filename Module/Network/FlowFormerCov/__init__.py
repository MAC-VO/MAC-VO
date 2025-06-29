import torch


def build_flowformer(cfg, encoder_dtype: torch.dtype, decoder_dtype: torch.dtype):
    from .flownet import FlowFormerCov
    return FlowFormerCov(cfg["latentcostformer"], encoder_dtype, decoder_dtype)
