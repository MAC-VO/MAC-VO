import argparse
import os
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn

from typing import get_args, Callable
from pathlib import Path
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from DataLoader import TrainDataset, FramePair, SourceDataFrame
from Train.MatchingNet.loss import sequence_loss
from Utility.Config import load_config, namespace_to_cfgnode
from Utility.PrettyPrint import ColoredTqdm

from .utils import (
    T_DataType, T_TrainType, 
    AssertLiteralType,
    get_datatype, get_scheduler, get_optimizer
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


def CropTv2(width = 640, height = 480) -> Callable[[FramePair,], FramePair]:
    def _crop_frame(frame: FramePair) -> FramePair:
        _, _, h, w = frame.cur.imageL.shape
        if h < height or w < width:
            raise ValueError(f"Image size ({h}, {w}) is smaller than the target size ({height}, {width})")

        top = (h - height) // 2
        left = (w - width) // 2

        def _transformFrame(frame: SourceDataFrame) -> None :
            if frame.gtFlow is not None: frame.gtFlow=frame.gtFlow[:,:, top:top + height, left:left + width]
            if frame.flowMask is not None: frame.flowMask=frame.flowMask[..., top:top + height, left:left + width]
            if frame.imageL is not None: frame.imageL=frame.imageL[..., top:top + height, left:left + width]
            if frame.imageR is not None: frame.imageR=frame.imageR[..., top:top + height, left:left + width]
            if frame.gtDepth is not None: frame.gtDepth=frame.gtDepth[..., top:top + height, left:left + width]

        _transformFrame(frame.cur); _transformFrame(frame.nxt)
        return frame
    
    return _crop_frame


def CastDatatype(data_type: T_DataType) -> Callable[[FramePair,], FramePair]:
    target_dtype = get_datatype(data_type)
    def _cast_datatype(frame: FramePair) -> FramePair:
        frame.cur.imageL = frame.cur.imageL.to(dtype=target_dtype)
        frame.cur.imageR = frame.cur.imageR.to(dtype=target_dtype)
        if frame.cur.gtFlow is not None: frame.cur.gtFlow = frame.cur.gtFlow.to(dtype=target_dtype)
        if frame.cur.gtDepth is not None: frame.cur.gtDepth = frame.cur.gtDepth.to(dtype=target_dtype)
        if frame.cur.flowMask is not None: frame.cur.flowMask = frame.cur.flowMask.to(dtype=target_dtype)
        
        frame.nxt.imageL = frame.nxt.imageL.to(dtype=target_dtype)
        frame.nxt.imageR = frame.nxt.imageR.to(dtype=target_dtype)
        if frame.nxt.gtFlow is not None  : frame.nxt.gtFlow   = frame.nxt.gtFlow.to(dtype=target_dtype)
        if frame.nxt.gtDepth is not None : frame.nxt.gtDepth  = frame.nxt.gtDepth.to(dtype=target_dtype)
        if frame.nxt.flowMask is not None: frame.nxt.flowMask = frame.nxt.flowMask.to(dtype=target_dtype)
        return frame
        
    return _cast_datatype


def AddNoise(stdv: float = 5.0) -> Callable[[FramePair,], FramePair]:
    stdv = np.random.uniform(0.0, stdv) / 255.0
    
    def _apply_noise(img: torch.Tensor) -> torch.Tensor:
        return (img + stdv * torch.randn(*img.shape)).clamp(0.0, 1.0)

    def _add_noise(frame: FramePair) -> FramePair:
        frame.cur.imageL = _apply_noise(frame.cur.imageL)
        frame.cur.imageR = _apply_noise(frame.cur.imageR)
        frame.nxt.imageL = _apply_noise(frame.nxt.imageL)
        frame.nxt.imageR = _apply_noise(frame.nxt.imageR)
        return frame

    return _add_noise


def train(modelcfg, cfg, loader, eval_loader=None):
    from Module.Network.FlowFormerCov import build_flowformer
    train_mode: T_TrainType = modelcfg.training_mode
    AssertLiteralType(train_mode, T_TrainType)
    
    model = build_flowformer(modlecfg, "cuda")
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
        wandb.watch(model, log=None)    #type: ignore
        
    total_steps = 0
    should_keep_training = True
    while should_keep_training:
        for frameData in ColoredTqdm(loader):
            metric_list = []
            assert frameData.cur.gtFlow is not None
            optimizer.zero_grad()
            img1, img2 = frameData.cur.imageL.cuda(), frameData.nxt.imageL.cuda()
            gt_flow = frameData.cur.gtFlow.cuda()
            flow_mask = frameData.cur.flowMask.cuda()
            
            flow, cov = model(img1, img2)
            loss, metrics = sequence_loss(
                                cfg=modlecfg,
                                preds=flow,
                                gt=gt_flow,
                                flow_mask=flow_mask,
                                cov_preds=cov)
            metric_list.append(metrics)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), modlecfg.clip)
            scaler.step(optimizer)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metrics["lr"] = lr
            scaler.update()
            if total_steps % int(modlecfg.log_freq) == 0:
                print("Iter: %d, Loss: %.4f" % (total_steps, loss.item()))
                if modlecfg.wandb:
                    wandb.log(merge_matrices(metric_list))
                metric_list = []
                
            total_steps += 1

            if total_steps > modlecfg.num_steps:
                should_keep_training = False
                break

            if modelcfg.autosave_freq and total_steps % modelcfg.autosave_freq == 0:
                PATH = "%s/%s/%d.pth" % (modelcfg.autosave_dir, modelcfg.name + modelcfg.time, total_steps)  
                
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
    
    traindatasets = TrainDataset.mp_instantiation(datacfg.data, 0, -1, lambda cfg: cfg.args.type in {"TartanAirv2", "TartanAir"})
    trainloader = DataLoader(
        ConcatDataset([
            ds.transform(CropTv2()).transform(CastDatatype(cfg.Model.datatype)).transform(AddNoise())
            for ds in traindatasets
            if ds is not None
        ]),
        batch_size=modlecfg.batch_size,
        shuffle=True,
        collate_fn=FramePair.collate,
        drop_last=True,
        num_workers=4,
    )
    
    if args.wandb:
        try:
            import wandb
            wandb.init(project="FlowFormerCov", name = modlecfg.name,  config=modlecfg)
        except ImportError:
            print("Wandb is not installed, disabling it.")
            modlecfg.wandb = False

    train(modlecfg, cfg, trainloader)
