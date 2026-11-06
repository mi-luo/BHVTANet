import copy
from typing import Any, Dict

import torch

DEFAULT_AUGMENTATION = {
    'enabled': False,
    'noise_std': 0.0,
    'noise_p': 0.0,
    'time_mask_width': 0,
    'time_mask_num': 0,
    'time_mask_p': 0.0,
    'channel_mask_num': 0,
    'channel_mask_p': 0.0,
    'scale_std': 0.0,
    'scale_p': 0.0,
    'time_shift': 0,
    'time_shift_p': 0.0,
}


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, 'items'):
        return {k: v for k, v in obj.items()}
    return {}


def resolve_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = _to_plain_dict(config.get('augmentation', {}))
    resolved = copy.deepcopy(DEFAULT_AUGMENTATION)
    resolved.update({k: v for k, v in raw.items() if v is not None})
    resolved['enabled'] = bool(resolved.get('enabled', False))
    for key in ['noise_std', 'noise_p', 'time_mask_p', 'channel_mask_p', 'scale_std', 'scale_p', 'time_shift_p']:
        resolved[key] = float(resolved.get(key, 0.0))
    for key in ['time_mask_width', 'time_mask_num', 'channel_mask_num', 'time_shift']:
        resolved[key] = int(resolved.get(key, 0))
    return resolved


class BrainAugmentor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = resolve_augmentation_config({'augmentation': cfg})

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        if not self.cfg['enabled']:
            return eeg
        x = eeg.clone()
        device = x.device
        c_num, t_num = x.shape[-2], x.shape[-1]

        if self.cfg['scale_std'] > 0 and torch.rand(1, device=device).item() < self.cfg['scale_p']:
            scale = 1.0 + torch.randn(1, device=device) * self.cfg['scale_std']
            x = x * scale

        if self.cfg['noise_std'] > 0 and torch.rand(1, device=device).item() < self.cfg['noise_p']:
            x = x + torch.randn_like(x) * self.cfg['noise_std']

        if self.cfg['time_shift'] > 0 and torch.rand(1, device=device).item() < self.cfg['time_shift_p']:
            max_shift = min(self.cfg['time_shift'], max(1, t_num - 1))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        if self.cfg['channel_mask_num'] > 0 and torch.rand(1, device=device).item() < self.cfg['channel_mask_p']:
            n_mask = min(self.cfg['channel_mask_num'], c_num)
            idx = torch.randperm(c_num, device=device)[:n_mask]
            x[idx] = 0.0

        if self.cfg['time_mask_width'] > 0 and self.cfg['time_mask_num'] > 0 and torch.rand(1, device=device).item() < self.cfg['time_mask_p']:
            width = min(self.cfg['time_mask_width'], t_num)
            for _ in range(self.cfg['time_mask_num']):
                start = int(torch.randint(0, max(1, t_num - width + 1), (1,), device=device).item())
                x[:, start:start + width] = 0.0

        return x
