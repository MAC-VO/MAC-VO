import torch
from Utility.PrettyPrint import Logger

def SimulateOnePatch(depth_mean: torch.Tensor, depth_var: torch.Tensor, selector_dist: torch.Tensor, num_sample: int) -> tuple[float, float]:
    """
    depth_mean: (N,) torch.float
    depth_var:  (N,) torch.float
    selector_dist: (N,) sum to one probabilty density tensor
    
    @returns
    (num_sample,) torch.float tensor as the sampling result on provided patch.
    """
    bucket_idx = torch.multinomial(selector_dist, num_sample, replacement=True)
    
    mean = depth_mean[bucket_idx]
    var  = depth_var[bucket_idx]
    samples = torch.randn_like(mean) * var + mean
    
    return samples.var().item(), samples.mean().item()
    

def CalculateOnePatch(depth_mean: torch.Tensor, depth_var: torch.Tensor, selector_dist: torch.Tensor) -> tuple[float, float]:
    """
    depth_mean: (N,) torch.float
    depth_var:  (N,) torch.float
    selector_dist: (N,) sum to one probabilty density tensor
    
    @returns
    (num_sample,) torch.float tensor as the sampling result on provided patch.
    """
    # Reject sub-population with sufficiently small probability to improve robustness on result.
    # Yet another engineering trick : |
    selector_dist[selector_dist < 1e-3] = 0.
    selector_dist = selector_dist / selector_dist.sum()
    
    calc_mean = (depth_mean * selector_dist).sum()
    calc_var = ((depth_var + depth_mean.square()) * selector_dist).sum() - calc_mean.square()
    return calc_var.item() / 2, calc_mean.item()


if __name__ == "__main__":
    PATCH_NUM = 10
    for _ in range(PATCH_NUM):
        depth_mean = torch.rand((25,))
        depth_var = torch.rand((25,)) * torch.rand((25,)).clamp(max=0.5)
        
        selector_dist = torch.rand((25,))
        selector_dist = selector_dist / selector_dist.sum()
        
        sampled_var, sampled_mean = SimulateOnePatch(depth_mean.cuda(), depth_var.cuda(), selector_dist.cuda(), 10000000)
        calced_var, calced_mean = CalculateOnePatch(depth_mean.cuda(), depth_var.cuda(), selector_dist.cuda())
        
        sampled_mean = round(sampled_mean, 5)
        calced_mean = round(calced_mean, 5)
        sampled_var = round(sampled_var, 5)
        calced_var = round(calced_var, 5)
        
        Logger.write("info", f"μ_Sample: {sampled_mean}, σ_Sample: {sampled_var}\t|\t μ_Calc: {calced_mean}, σ_Calc: {calced_var}")
