import torch
from Utility.Extensions import OnCallCompiler

@OnCallCompiler()
def flow_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    flow_loss = torch.tensor(0.0, device=gt.device)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (preds[i] - gt).abs()
        flow_loss += i_weight * (mask * i_loss).nanmean()

    return flow_loss


@OnCallCompiler()
def cov_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, cov_preds: list[torch.Tensor], 
             flow_mask: torch.Tensor | None = None, max_cov: float = 10., eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    n_predictions = len(preds)
    cov_loss = torch.zeros_like(gt)
    error = torch.zeros_like(gt)
    exp_cov = torch.zeros_like(gt)
    
    error = None
    for i in range(n_predictions):
        exp_cov = cov_preds[i] + eps
        error = ((preds[i] - gt)**2).detach()
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = ((error / exp_cov ) + torch.log(exp_cov)) * i_weight
        cov_loss += i_loss
    assert error is not None
    
    return cov_loss.mean(), error

@OnCallCompiler()
def final_cov_loss(preds: torch.Tensor, gt: torch.Tensor, cov_preds: list[torch.Tensor], 
                   flow_mask: torch.Tensor | None = None, max_cov: float = 10., eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    cov = cov_preds[-1:]
    pred = preds[-1:]
    return cov_loss(1.0, pred, gt, cov, flow_mask, max_cov, eps)

@OnCallCompiler()
def depth_loss(gamma: float, preds: torch.Tensor, gt_depth: torch.Tensor, Ks: torch.Tensor, bls: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    fxs = Ks[:, 0, 0]
    gt_disparity  = (fxs * bls).unsqueeze(-1).unsqueeze(-1) / gt_depth 
    est_disparity = preds[..., 0]
    
    depth_loss = torch.tensor(0.0, device=gt_disparity.device)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss   = (est_disparity[i] + gt_disparity)
        depth_loss = i_weight * i_loss.mean()
    
    return depth_loss

def sequence_loss(cfg, preds: torch.Tensor, gt: torch.Tensor, flow_mask: torch.Tensor | None, cov_preds: list[torch.Tensor] | None):    
    gt_mag = gt.norm(dim=1, keepdim=True)
    mask = gt_mag < cfg.max_flow
    if flow_mask is not None: mask &= flow_mask.bool()
    
    metrics = dict()
    
    match cfg.training_mode:
        case "flow":
            loss = flow_loss(cfg.gamma, preds, gt, mask)
        
        case "finalcov":
            assert cov_preds is not None
            if cfg.cov_mask:
                loss, error = final_cov_loss(preds, gt, cov_preds, mask)
            else:
                loss, error = final_cov_loss(preds, gt, cov_preds)
            metrics["error"] = error.mean().item()
            metrics["cov"] = cov_preds[-1].mean().item()
            metrics["cov_ratioe"] = (error/cov_preds[-1]).mean().item()
        
        case "cov":
            assert cov_preds is not None
            if cfg.cov_mask:
                loss, error = cov_loss(cfg.gamma, preds, gt, cov_preds, mask)
            else:
                loss, error = cov_loss(cfg.gamma, preds, gt, cov_preds)
            metrics["error"] = error.mean().item()
            metrics["cov"] = cov_preds[-1].mean().item()
            cov_mag = cov_preds[-1].sum(dim=-1).sqrt()
            epe = error.sum(dim=-1).sqrt()
            metrics["cov_ratioe"] = (cov_mag / epe).mean().item()
            
        case default:
            raise ValueError(f"Unavailable training mode {default}")      
    return loss, metrics


def sequence_metric(cfg, preds: torch.Tensor, cov_preds: list[torch.Tensor] | None, gt: torch.Tensor, flow_mask: torch.Tensor | None):
    sqe = (preds[-1] - gt)**2
    epe = torch.sum(sqe, dim=1).sqrt()
    
    gt_mag = gt.norm(dim=1)
    mask = flow_mask & (gt_mag < cfg.max_flow) if flow_mask is not None else gt_mag < cfg.max_flow
    masked_epe = epe.view(-1)[mask.view(-1)]
    
    metrics = {
        'epe': masked_epe.mean().item(),
        '1px': (masked_epe < 1).float().mean().item(),
        '3px': (masked_epe < 3).float().mean().item(),
        '5px': (masked_epe < 5).float().mean().item(),
    }
    
    flow_gt_thresholds = [5, 10, 20]
    gt_mag = gt_mag.view(-1)[mask.view(-1)]
    for t in flow_gt_thresholds:
        e = masked_epe[gt_mag < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    match cfg.training_mode:
        case "flow":
            loss = flow_loss(cfg.gamma, preds, gt, mask)
            
        case "cov":
            assert cov_preds is not None
            loss, _ = cov_loss(cfg.gamma, preds, gt, cov_preds)
            
            cov_mag = cov_preds[-1].sum(dim=1).sqrt()
            cov_ratio = (cov_mag / epe)
            metrics.update({"cov_loss": loss.mean().item()})
            metrics.update({"cov_mag": cov_mag.mean().item()})
            metrics.update({"cov_ratio": cov_ratio.nanmean().item()})
            
        case default:
            raise ValueError(f"Unavailable training mode {default}")      
    
    return loss, metrics
