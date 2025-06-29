import argparse
import os
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn

from typing import get_args
from pathlib import Path
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from DataLoader import TrainDataset, DataFramePair, StereoFrame, CenterCropFrame, CastDataType, AddImageNoise, ScaleFrame
from Train.MatchingNet.loss import sequence_loss, sequence_metric
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm, Logger

from .utils import (
    T_TrainType, 
    AssertLiteralType,
    get_scheduler, get_optimizer
)


def write_wandb(header, objs, epoch_i):
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)


def merge_matrices(matrices):
    _matric = matrices[0]
    for i, m in enumerate(matrices):
        if i == 0:
            continue
        for k, v in m.items():
            _matric[k] += v
    
    for k, v in _matric.items():
        _matric[k] /= len(matrices)
    return _matric


def train(modelcfg, cfg, loader: DataLoader[DataFramePair[StereoFrame]], eval_loader=None):
    from Module.Network.FlowFormerCov import build_flowformer
    train_mode: T_TrainType = modelcfg.training_mode
    AssertLiteralType(train_mode, T_TrainType)
    
    model = build_flowformer(modlecfg, torch.float32, torch.float32)
    if modlecfg.restore_ckpt:
        model.load_ddp_state_dict(torch.load(modlecfg.restore_ckpt, weights_only=True))

    model = nn.DataParallel(model)
    model.cuda()
    model.train()
    
    optimizer = get_optimizer(cfg.Model.optimizer.type)(
        model.parameters(),
        **vars(cfg.Model.optimizer.args)
    )
    scheduler = get_scheduler(cfg.Model.scheduler.type)(
        optimizer,
        **vars(cfg.Model.scheduler.args)
    )
    scaler = GradScaler(enabled=modlecfg.mixed_precision)
    model_ptr = model.module if isinstance(model, nn.DataParallel) else model
    match train_mode:
        case "flow":
            for param in model_ptr.memory_decoder.cov_update.parameters():
                param.requires_grad = False
        case "cov":
            for param in model_ptr.parameters():
                param.requires_grad = False
            for param in model_ptr.memory_decoder.cov_update.parameters():
                param.requires_grad = True

    if modlecfg.wandb:
        wandb.init(project=modlecfg.name, config=modlecfg)
        wandb.watch(model, log=None)
        
    total_steps = 0
    should_keep_training = True
    while should_keep_training:
        frameData: DataFramePair[StereoFrame]
        for frameData in ColoredTqdm(loader):
            assert frameData.cur.stereo.gt_flow   is not None
            assert frameData.cur.stereo.flow_mask is not None
            optimizer.zero_grad()
            img1, img2 = frameData.cur.stereo.imageL.cuda(), frameData.nxt.stereo.imageL.cuda()
            gt_flow = frameData.cur.stereo.gt_flow.cuda()
            flow_mask = frameData.cur.stereo.flow_mask.cuda()
            
            flow, cov = model(img1, img2)
            loss, _ = sequence_loss(cfg=modlecfg, preds=flow, gt=gt_flow, flow_mask=flow_mask, cov_preds=cov)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), modlecfg.clip)
            scaler.step(optimizer)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            scaler.update()
            if total_steps % int(modlecfg.log_freq) == 0:
                Logger.write("info", "Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                if modlecfg.wandb:
                    metrics = merge_matrices([sequence_metric(modelcfg, flow, cov, gt_flow, flow_mask)[1]])
                    metrics["lr"] = lr
                    wandb.log(metrics)
                
            total_steps += 1

            if total_steps > modlecfg.num_steps:
                should_keep_training = False
                break

            if modelcfg.autosave_freq and total_steps % modelcfg.autosave_freq == 0:
                PATH = "%s/%s/%d.pth" % (modelcfg.autosave_dir, modelcfg.name + modelcfg.time, total_steps)  
                Logger.write("info", f"Save model to {PATH}")
                if isinstance(model, nn.DataParallel):
                    # We don't want to have a layer of `module.` on all weights. Since we are definitely not
                    # using DDP during inference, I will just save the "real weights" of the model.
                    torch.save(model.module.state_dict(), PATH)
                else:
                    torch.save(model.state_dict(), PATH)
                
    PATH = "%s/%s/%d.pth" % (modlecfg.autosave_dir, modlecfg.name + modlecfg.time, total_steps)
    torch.save(model.state_dict(), PATH)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Config/Train/Demo.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--autosave_dir", type=str, default="Model")
    parser.add_argument("--training_mode", type=str, choices=get_args(T_TrainType),
                        default="cov", help=f"Training mode: {get_args(T_TrainType)}")
    args = parser.parse_args()
    cfg, _ = load_config(Path(args.config))
    modlecfg = namespace_to_cfgnode(cfg.Model)
    modlecfg.update(vars(args))
    datacfg = cfg.Train
    modlecfg.time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    
    os.makedirs("%s/%s" % (args.autosave_dir, modlecfg.name + modlecfg.time), exist_ok=True)
    torch.manual_seed(modlecfg.seed)
    np.random.seed(modlecfg.seed)
    transforms = [CenterCropFrame(dict(width=640, height=480)),
                  CastDataType(dict(dtype=cfg.Model.datatype)),
                  AddImageNoise(dict(stdv=5.0)),
                  ScaleFrame(dict(scale_u=cfg.Model.image_scale, scale_v=cfg.Model.image_scale, interp='nearest'))]
    
    traindatasets = TrainDataset[StereoFrame].mp_instantiation(datacfg.data, 0, -1, lambda cfg: cfg.type in {"TartanAir_NoIMU", "TartanAirv2_NoIMU"})
    trainloader = DataLoader[DataFramePair[StereoFrame]](
        ConcatDataset([
            ds.transform_source(transforms)
            for ds in traindatasets
            if ds is not None
        ]),
        batch_size=modlecfg.batch_size,
        shuffle=True,
        collate_fn=DataFramePair.collate,
        drop_last=True,
        num_workers=4,
    )
    
    if args.wandb:
        try:
            import wandb
            wandb.init(project="FlowFormerCov", name = modlecfg.name,  config=modlecfg)
        except ImportError:
            Logger.write("warn", "Wandb is not installed, disabling it.")
            modlecfg.wandb = False

    train(modlecfg, cfg, trainloader)
