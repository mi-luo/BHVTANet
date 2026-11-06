import copy
import math
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR

DEFAULT_TRAIN_CFG = {
    'weight_decay': 1e-4,
    'monitor': 'val_score',
    'monitor_mode': 'max',
    'early_stop_patience': 15,
    'early_stop_min_delta': 5e-4,
    'gradient_clip_val': 1.0,
    'val_top1_weight': 0.7,
    'val_top5_weight': 0.3,
    'scheduler': {
        'enabled': True,
        'name': 'warmup_cosine',
        'warmup_ratio': 0.1,
        'min_lr_ratio': 0.05,
    },
    'swa': {
        'enabled': False,
        'lr': None,
        'epoch_start': 0.75,
        'annealing_epochs': 10,
    },
    'domain_alignment': {
        'enabled': False,
        'weight': 0.0,
        'target': 'residual',
        'normalize': True,
        'start_epoch': 0,
        'stop_epoch': None,
        'stop_ratio': None,
        'min_count': 8,
        'variance_weight': 0.0,
    },
    'subject_balanced_loss': {
        'enabled': False,
        'weight': 0.0,
        'min_count': 16,
        'start_epoch': 0,
        'stop_epoch': None,
        'stop_ratio': None,
    },
    'top1_margin_loss': {
        'enabled': False,
        'weight': 0.0,
        'margin': 0.02,
        'temperature': 0.05,
        'hard_topk': 1,
        'symmetric': True,
        'start_epoch': 0,
        'stop_epoch': None,
        'stop_ratio': None,
    },
    'rank_calibration_loss': {
        'enabled': False,
        'weight': 0.0,
        'topk': 5,
        'rank_temperature': 1.0,
        'beta': 5.0,
        'margin': 0.0,
        'symmetric': True,
        'topk_weight': 1.0,
        'listwise_weight': 0.10,
        'reliability_enabled': True,
        'reliability_temperature': 1.0,
        'reliability_min_weight': 0.35,
        'reliability_entropy_weight': 0.35,
        'start_epoch': 0,
        'stop_epoch': None,
        'stop_ratio': None,
        'warmup_epochs': 0,
    },
    'consistency_loss': {
        'enabled': False,
        'weight': 0.0,
        'retrieval_weight': 1.0,
        'embedding_weight': 0.0,
        'p': 1.0,
        'detach_target': True,
        'noise_std': 0.0,
        'scale_std': 0.0,
        'time_shift': 0,
        'time_mask_width': 0,
        'time_mask_num': 0,
        'channel_mask_num': 0,
        'start_epoch': 0,
        'stop_epoch': None,
        'stop_ratio': None,
        'warmup_epochs': 0,
    },
}


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, 'items'):
        return {k: v for k, v in obj.items()}
    return {}


def resolve_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = _to_plain_dict(config.get('train', {}))
    resolved = copy.deepcopy(DEFAULT_TRAIN_CFG)
    resolved.update({k: v for k, v in raw.items() if k not in {'scheduler', 'swa'} and v is not None})

    scheduler = copy.deepcopy(DEFAULT_TRAIN_CFG['scheduler'])
    scheduler.update(_to_plain_dict(raw.get('scheduler', {})))
    scheduler['enabled'] = bool(scheduler.get('enabled', False))
    scheduler['warmup_ratio'] = float(scheduler.get('warmup_ratio', 0.1))
    scheduler['min_lr_ratio'] = float(scheduler.get('min_lr_ratio', 0.05))
    resolved['scheduler'] = scheduler

    swa = copy.deepcopy(DEFAULT_TRAIN_CFG['swa'])
    swa.update(_to_plain_dict(raw.get('swa', {})))
    swa['enabled'] = bool(swa.get('enabled', False))
    resolved['swa'] = swa

    domain_alignment = copy.deepcopy(DEFAULT_TRAIN_CFG['domain_alignment'])
    domain_alignment.update(_to_plain_dict(raw.get('domain_alignment', {})))
    domain_alignment['enabled'] = bool(domain_alignment.get('enabled', False))
    domain_alignment['weight'] = float(domain_alignment.get('weight', 0.0))
    domain_alignment['target'] = str(domain_alignment.get('target', 'residual'))
    domain_alignment['normalize'] = bool(domain_alignment.get('normalize', True))
    domain_alignment['start_epoch'] = max(0, int(domain_alignment.get('start_epoch', 0)))
    if domain_alignment.get('stop_epoch', None) is not None:
        domain_alignment['stop_epoch'] = int(domain_alignment['stop_epoch'])
    if domain_alignment.get('stop_ratio', None) is not None:
        domain_alignment['stop_ratio'] = float(domain_alignment['stop_ratio'])
    domain_alignment['min_count'] = max(1, int(domain_alignment.get('min_count', 8)))
    domain_alignment['variance_weight'] = float(domain_alignment.get('variance_weight', 0.0))
    resolved['domain_alignment'] = domain_alignment

    subject_balanced_loss = copy.deepcopy(DEFAULT_TRAIN_CFG['subject_balanced_loss'])
    subject_balanced_loss.update(_to_plain_dict(raw.get('subject_balanced_loss', {})))
    subject_balanced_loss['enabled'] = bool(subject_balanced_loss.get('enabled', False))
    subject_balanced_loss['weight'] = float(subject_balanced_loss.get('weight', 0.0))
    subject_balanced_loss['min_count'] = max(2, int(subject_balanced_loss.get('min_count', 16)))
    subject_balanced_loss['start_epoch'] = max(0, int(subject_balanced_loss.get('start_epoch', 0)))
    if subject_balanced_loss.get('stop_epoch', None) is not None:
        subject_balanced_loss['stop_epoch'] = int(subject_balanced_loss['stop_epoch'])
    if subject_balanced_loss.get('stop_ratio', None) is not None:
        subject_balanced_loss['stop_ratio'] = float(subject_balanced_loss['stop_ratio'])
    resolved['subject_balanced_loss'] = subject_balanced_loss

    top1_margin_loss = copy.deepcopy(DEFAULT_TRAIN_CFG['top1_margin_loss'])
    top1_margin_loss.update(_to_plain_dict(raw.get('top1_margin_loss', {})))
    top1_margin_loss['enabled'] = bool(top1_margin_loss.get('enabled', False))
    top1_margin_loss['weight'] = float(top1_margin_loss.get('weight', 0.0))
    top1_margin_loss['margin'] = float(top1_margin_loss.get('margin', 0.02))
    top1_margin_loss['temperature'] = max(1e-6, float(top1_margin_loss.get('temperature', 0.05)))
    top1_margin_loss['hard_topk'] = max(1, int(top1_margin_loss.get('hard_topk', 1)))
    top1_margin_loss['symmetric'] = bool(top1_margin_loss.get('symmetric', True))
    top1_margin_loss['start_epoch'] = max(0, int(top1_margin_loss.get('start_epoch', 0)))
    if top1_margin_loss.get('stop_epoch', None) is not None:
        top1_margin_loss['stop_epoch'] = int(top1_margin_loss['stop_epoch'])
    if top1_margin_loss.get('stop_ratio', None) is not None:
        top1_margin_loss['stop_ratio'] = float(top1_margin_loss['stop_ratio'])
    resolved['top1_margin_loss'] = top1_margin_loss

    rank_calibration_loss = copy.deepcopy(DEFAULT_TRAIN_CFG['rank_calibration_loss'])
    rank_calibration_loss.update(_to_plain_dict(raw.get('rank_calibration_loss', {})))
    rank_calibration_loss['enabled'] = bool(rank_calibration_loss.get('enabled', False))
    rank_calibration_loss['weight'] = float(rank_calibration_loss.get('weight', 0.0))
    rank_calibration_loss['topk'] = max(1, int(rank_calibration_loss.get('topk', 5)))
    rank_calibration_loss['rank_temperature'] = max(1e-6, float(rank_calibration_loss.get('rank_temperature', 1.0)))
    rank_calibration_loss['beta'] = max(1e-6, float(rank_calibration_loss.get('beta', 5.0)))
    rank_calibration_loss['margin'] = float(rank_calibration_loss.get('margin', 0.0))
    rank_calibration_loss['symmetric'] = bool(rank_calibration_loss.get('symmetric', True))
    rank_calibration_loss['topk_weight'] = float(rank_calibration_loss.get('topk_weight', 1.0))
    rank_calibration_loss['listwise_weight'] = float(rank_calibration_loss.get('listwise_weight', 0.10))
    rank_calibration_loss['reliability_enabled'] = bool(rank_calibration_loss.get('reliability_enabled', True))
    rank_calibration_loss['reliability_temperature'] = max(1e-6, float(rank_calibration_loss.get('reliability_temperature', 1.0)))
    rank_calibration_loss['reliability_min_weight'] = float(rank_calibration_loss.get('reliability_min_weight', 0.35))
    rank_calibration_loss['reliability_entropy_weight'] = float(rank_calibration_loss.get('reliability_entropy_weight', 0.35))
    rank_calibration_loss['start_epoch'] = max(0, int(rank_calibration_loss.get('start_epoch', 0)))
    rank_calibration_loss['warmup_epochs'] = max(0, int(rank_calibration_loss.get('warmup_epochs', 0)))
    if rank_calibration_loss.get('stop_epoch', None) is not None:
        rank_calibration_loss['stop_epoch'] = int(rank_calibration_loss['stop_epoch'])
    if rank_calibration_loss.get('stop_ratio', None) is not None:
        rank_calibration_loss['stop_ratio'] = float(rank_calibration_loss['stop_ratio'])
    resolved['rank_calibration_loss'] = rank_calibration_loss

    consistency_loss = copy.deepcopy(DEFAULT_TRAIN_CFG['consistency_loss'])
    consistency_loss.update(_to_plain_dict(raw.get('consistency_loss', {})))
    consistency_loss['enabled'] = bool(consistency_loss.get('enabled', False))
    consistency_loss['weight'] = float(consistency_loss.get('weight', 0.0))
    consistency_loss['retrieval_weight'] = float(consistency_loss.get('retrieval_weight', 1.0))
    consistency_loss['embedding_weight'] = float(consistency_loss.get('embedding_weight', 0.0))
    consistency_loss['p'] = min(1.0, max(0.0, float(consistency_loss.get('p', 1.0))))
    consistency_loss['detach_target'] = bool(consistency_loss.get('detach_target', True))
    consistency_loss['noise_std'] = max(0.0, float(consistency_loss.get('noise_std', 0.0)))
    consistency_loss['scale_std'] = max(0.0, float(consistency_loss.get('scale_std', 0.0)))
    consistency_loss['time_shift'] = max(0, int(consistency_loss.get('time_shift', 0)))
    consistency_loss['time_mask_width'] = max(0, int(consistency_loss.get('time_mask_width', 0)))
    consistency_loss['time_mask_num'] = max(0, int(consistency_loss.get('time_mask_num', 0)))
    consistency_loss['channel_mask_num'] = max(0, int(consistency_loss.get('channel_mask_num', 0)))
    consistency_loss['start_epoch'] = max(0, int(consistency_loss.get('start_epoch', 0)))
    consistency_loss['warmup_epochs'] = max(0, int(consistency_loss.get('warmup_epochs', 0)))
    if consistency_loss.get('stop_epoch', None) is not None:
        consistency_loss['stop_epoch'] = int(consistency_loss['stop_epoch'])
    if consistency_loss.get('stop_ratio', None) is not None:
        consistency_loss['stop_ratio'] = float(consistency_loss['stop_ratio'])
    resolved['consistency_loss'] = consistency_loss

    resolved['weight_decay'] = float(resolved.get('weight_decay', 1e-4))
    resolved['early_stop_patience'] = int(resolved.get('early_stop_patience', 15))
    resolved['early_stop_min_delta'] = float(resolved.get('early_stop_min_delta', 5e-4))
    resolved['gradient_clip_val'] = float(resolved.get('gradient_clip_val', 0.0))
    resolved['val_top1_weight'] = float(resolved.get('val_top1_weight', 0.7))
    resolved['val_top5_weight'] = float(resolved.get('val_top5_weight', 0.3))
    return resolved


def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.1, min_lr_ratio: float = 0.05):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(min_lr_ratio, float(current_step + 1) / float(warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def to_jsonable(obj: Any):
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj
