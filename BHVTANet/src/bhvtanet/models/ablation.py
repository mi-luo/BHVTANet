import copy
from typing import Any, Dict, List, Optional

VALID_FEATURE_KINDS = {'clean', 'semantic', 'perceptual'}

PRESET_LIBRARY: Dict[str, Dict[str, Any]] = {
    'baseline': {
        'primary_feature': 'clean',
        'secondary_feature': None,
        'use_feature_interpolation': False,
        'learnable_mix_score': False,
        'fixed_mix_weight': 0.0,
    },
    'semantic_only': {
        'primary_feature': 'semantic',
        'secondary_feature': None,
        'use_feature_interpolation': False,
        'learnable_mix_score': False,
        'fixed_mix_weight': 0.0,
    },
    'full': {
        'primary_feature': 'semantic',
        'secondary_feature': 'perceptual',
        'use_feature_interpolation': True,
        'learnable_mix_score': True,
        'fixed_mix_weight': 0.5,
    },
}

DEFAULT_ABLATION: Dict[str, Any] = {
    'preset': 'full',
    'primary_feature': 'semantic',
    'secondary_feature': 'perceptual',
    'use_feature_interpolation': True,
    'learnable_mix_score': True,
    'fixed_mix_weight': 0.5,
    'normalize_primary_feature': True,
    'normalize_secondary_feature': False,
}


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, 'items'):
        return {k: v for k, v in obj.items()}
    return {}


def resolve_ablation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = _to_plain_dict(config.get('ablation', {}))
    preset = str(raw.get('preset', DEFAULT_ABLATION['preset'])).lower()

    if preset == 'innov1':
        preset = 'semantic_only'

    resolved = copy.deepcopy(DEFAULT_ABLATION)
    if preset in PRESET_LIBRARY:
        resolved.update(copy.deepcopy(PRESET_LIBRARY[preset]))
    resolved.update({k: v for k, v in raw.items() if v is not None})
    resolved['preset'] = preset

    primary = resolved.get('primary_feature')
    secondary = resolved.get('secondary_feature')
    if primary not in VALID_FEATURE_KINDS:
        raise ValueError(f'Unsupported primary_feature: {primary}')
    if secondary is not None and secondary not in VALID_FEATURE_KINDS:
        raise ValueError(f'Unsupported secondary_feature: {secondary}')

    resolved['use_feature_interpolation'] = bool(resolved.get('use_feature_interpolation', False))
    resolved['learnable_mix_score'] = bool(resolved.get('learnable_mix_score', False))
    resolved['normalize_primary_feature'] = bool(resolved.get('normalize_primary_feature', True))
    resolved['normalize_secondary_feature'] = bool(resolved.get('normalize_secondary_feature', False))
    resolved['fixed_mix_weight'] = float(resolved.get('fixed_mix_weight', 0.5))

    if not resolved['use_feature_interpolation']:
        resolved['secondary_feature'] = None
        resolved['learnable_mix_score'] = False
    else:
        if resolved['secondary_feature'] is None:
            raise ValueError('secondary_feature must be set when use_feature_interpolation=True')
        if resolved['secondary_feature'] == resolved['primary_feature']:
            raise ValueError('primary_feature and secondary_feature must differ when interpolation is enabled')
        resolved['fixed_mix_weight'] = min(1.0, max(0.0, resolved['fixed_mix_weight']))

    return resolved


def required_feature_kinds(config: Dict[str, Any]) -> List[str]:
    ablation = resolve_ablation_config(config)
    kinds = {ablation['primary_feature']}
    if ablation['secondary_feature'] is not None:
        kinds.add(ablation['secondary_feature'])
    return sorted(kinds)
