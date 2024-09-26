import torch

@torch.compile()
def flow_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n_predictions = len(preds)
    
    flow_loss = torch.tensor(0.0, device=gt.device)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (preds[i] - gt).abs()
        flow_loss += i_weight * (mask[:, None] * i_loss).mean()

    return flow_loss


@torch.compile()
def cov_loss(gamma: float, preds: torch.Tensor, gt: torch.Tensor, cov_preds: list[torch.Tensor]):
    n_predictions = len(preds)
    cov_loss = torch.zeros_like(gt)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = ((preds[i] - gt)**2 / (2 * torch.exp(2 * cov_preds[i])) + cov_preds[i]) * i_weight
        cov_loss += i_loss

    return cov_loss.mean()


def sequence_loss(cfg, preds: torch.Tensor, gt: torch.Tensor, flow_mask: torch.Tensor | None, cov_preds: list[torch.Tensor] | None):
    sqe = (preds[-1] - gt)**2
    epe = torch.sum(sqe, dim=1).sqrt()
    
    gt_mag = gt.norm(dim=1)
    mask = (flow_mask >= 0.5) & (gt_mag < cfg.max_flow) if flow_mask is not None else gt_mag < cfg.max_flow
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
            loss = cov_loss(cfg.gamma, preds, gt, cov_preds)
            
            cov = torch.exp(2 * cov_preds[-1])
            cov_mag = cov.sum(dim=1).sqrt()
            cov_ratio = (cov_mag / epe)
            metrics.update({"cov_loss": loss.mean().item()})
            metrics.update({"cov_mag": cov_mag.mean().item()})
            metrics.update({"cov_ratio": cov_ratio.mean().item()})
        case default:
            raise ValueError(f"Unavailable training mode {default}")        
    return loss, metrics
