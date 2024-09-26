def build_flowformer(cfg, device, use_inference_jit=False):
    name = cfg.transformer
    if name == "latentcostformer":
        from .flownet import FlowFormerCov
        return FlowFormerCov(cfg[name], device, use_inference_jit)
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")
