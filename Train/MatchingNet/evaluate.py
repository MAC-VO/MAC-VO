#Evaluate the model on the validation set
import argparse
import torch
import yaml
import pdb

from torch import nn
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from yacs.config import CfgNode as CN
from DataLoader import TrainDataset
from Module.Network.FlowFormerCov import build_flowformer
from Module.Network.FlowFormerCov.flownet import FlowFormerCov
from Module.Network.FlowFormer.configs.submission import get_cfg
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm
from typing import List, Dict

from .train_flowformer import merge_matrices

try :
    import wandb
except ImportError:
    usewandb = False


def evaluate(model: nn.DataParallel[FlowFormerCov], loader: DataLoader, length: int, usewandb) -> List[Dict]:
    model.eval()
    with torch.no_grad():
        step = 0
        metric_list = []
        for frameData in ColoredTqdm(loader, desc="Evaluation", total=length):

            img1, img2 = frameData.cur.imageL.cuda(), frameData.nxt.imageL.cuda()
            gt_flow = frameData.cur.gtFlow.cuda()
            # flow, cov = model.module.inference(img1, img2)
            flow_pre, cov_pre = model.forward(img1, img2)
            flow, cov = flow_pre[0], torch.exp(2 * cov_pre[0])
            
            error_mask = (gt_flow.norm(dim=1) < 240)
            flow_mask = error_mask.unsqueeze(1).expand_as(gt_flow) # for masking the flow
            
            MSE = (flow - gt_flow)**2
            EPE = (flow - gt_flow).norm(dim=1)
            masked_EPE = EPE[error_mask]
            
            cov_dist = cov.sqrt().norm(dim=1)
            cov_ratio = (cov_dist / EPE)
            eval_loss = MSE / (2 * cov) + 0.5 * torch.log(cov)
            step += 1
            if step > length:
                break
            metric = {
                'mse': MSE.mean().item(),
                'epe': EPE.mean().item(),
                'masked_epe': masked_EPE.mean().item(), # 'mask for epe < 200
                'cov_dist': cov_dist.mean().item(),
                'cov_ratio': cov_ratio.mean().item(),
                'eval_loss': eval_loss.mean().item()
            }

            metric_list.append(metric)
        return metric_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on TartanAirV2")
    parser.add_argument("--ckpt", default = "Model/15001_FlowFormerCov.pth", type=str, help="Path to the model weight")
    parser.add_argument("--config", type=str, default="Config/Train/Demo.yaml")
    parser.add_argument("--length", type=int, default=2000, help="Length of the evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--wandb", action="store_true", help="Use wandb to log the results")
    args = parser.parse_args()

    cfg, _ = load_config(Path(args.config))
    modlecfg = namespace_to_cfgnode(cfg.Model)
    modlecfg.update(vars(args))
    datacfg = cfg.Evaluate

    model = nn.DataParallel(build_flowformer(namespace_to_cfgnode(cfg.Model), torch.float32, torch.float32))
    model.load_state_dict(torch.load(args.ckpt), strict = False)
    model.cuda()

    eval_loader = DataLoader(
        ConcatDataset([
            TrainDataset(from_idx=0, to_idx=-1, **vars(dcfg.args))
            for dcfg in datacfg.data
        ]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )
    metric_list = evaluate(model, eval_loader, args.length//args.batch_size, args.wandb)
    metric = merge_matrices(metric_list)
    print(metric)