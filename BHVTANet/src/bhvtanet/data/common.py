import copy
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union
from pathlib import Path

import cv2
import numpy as np
import open_clip
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
import logging
from bhvtanet.models.ablation import required_feature_kinds
from bhvtanet.models.mctf import resolve_mctf_config

EEG_CHANNELS: List[str] = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2'
]

CHANNEL_GROUPS: Dict[str, List[str]] = {
    'ALL': EEG_CHANNELS,
    'O': ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'],
    'P': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'C': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    'T': ['FT9', 'FT7', 'FT8', 'FT10', 'TP9', 'TP7', 'TP8', 'TP10', 'T7', 'T8'],
    'F': ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'],
    'OP': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'],
    'OPC': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    'OPT': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'FT9', 'FT7', 'FT8', 'FT10', 'TP9', 'TP7', 'TP8', 'TP10', 'T7', 'T8'],
    'OPCT': ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'FT9', 'FT7', 'FT8', 'FT10', 'TP9', 'TP7', 'TP8', 'TP10', 'T7', 'T8'],
}


def deep_copy_config(config: Any) -> Any:
    return copy.deepcopy(config)


def normalize_selected_channels(selected_ch: Union[str, Sequence[str], bool, None]) -> Optional[List[str]]:
    if selected_ch in (False, None):
        return None
    if isinstance(selected_ch, str):
        key = selected_ch.upper()
        if key in ('NONE', 'FALSE'):
            return None
        if key in CHANNEL_GROUPS:
            return CHANNEL_GROUPS[key]
        return [selected_ch]
    return list(selected_ch)


def canonical_vision_backbone_name(model_type: str) -> str:
    mapping = {
        'ViT-B-16': 'ViT_B_16',
        'ViT-B-32': 'ViT_B_32',
        'OpenAI-ViT-L-14': 'OpenAI_ViT_L_14',
        'OpenAI-ViT-L-14-336': 'OpenAI_ViT_L_14_336',
        'ViT-L-14': 'ViT_L_14',
        'DFN-ViT-L-14': 'DFN_ViT_L_14',
        'DataComp-ViT-L-14': 'DataComp_ViT_L_14',
        'SigLIP-B-16-256': 'SigLIP_B_16_256',
        'SigLIP2-B-16-256': 'SigLIP2_B_16_256',
        'SigLIP2-L-16-256': 'SigLIP2_L_16_256',
        'Timm-SigLIP2-L-16-256': 'Timm_SigLIP2_L_16_256',
        'SigLIP-SO400M-14-384': 'SigLIP_SO400M_14_384',
        'Timm-DINOv2-L-14': 'Timm_DINOv2_L_14',
        'SigLIP-L-16-256': 'SigLIP_L_16_256',
        'PE-Core-B-16': 'PE_Core_B_16',
        'PE-Core-L-14-336': 'PE_Core_L_14_336',
        'PE-Core-bigG-14-448': 'PE_Core_bigG_14_448',
        'DFN5B-ViT-H-14-378': 'DFN5B_ViT_H_14_378',
        'ViT-H-14': 'ViT_H_14',
        'MetaCLIP-H-14': 'MetaCLIP_H_14',
        'ViT-H-14-CLIPA-336': 'ViT_H_14_CLIPA_336',
        'ViT-g-14': 'ViT_g_14',
        'ViT-bigG-14': 'ViT_bigG_14',
        'EVA02-B-16': 'EVA02_B_16',
        'EVA02-L-14': 'EVA02_L_14',
        'ConvNeXt-BW': 'ConvNeXt_BW',
        'InternViT-L28': 'InternViT_L28',
        'MobileCLIP2-B': 'MobileCLIP2_B',
        'MetaCLIP-B-16': 'MetaCLIP_B_16',
    }
    return mapping.get(model_type, model_type)


def infer_channel_count(selected_ch: Union[str, Sequence[str], bool, None], dataset: str) -> int:
    normalized = normalize_selected_channels(selected_ch)
    if normalized is None:
        return 271 if dataset == 'meg' else len(EEG_CHANNELS)
    return len(normalized)


def feature_candidates(feature_dir: str, model_type: str, mode: str, soft_k: int, soft_g: int, hard_k: int) -> Dict[str, List[str]]:
    model_name = canonical_vision_backbone_name(model_type)
    if model_name == 'InternViT_L28':
        internvit_path = os.path.join(feature_dir, f'internvit_6b_layers_28_{mode}.pt')
        return {
            'clean': [internvit_path],
            'semantic': [internvit_path],
            'perceptual': [internvit_path],
        }
    return {
        'clean': [
            os.path.join(feature_dir, f'{model_name}_clean_{mode}.pt'),
        ],
        'semantic': [
            os.path.join(feature_dir, f'{model_name}_semantic_fovea_k{soft_k}_g{soft_g}_{mode}.pt'),
        ],
        'perceptual': [
            os.path.join(feature_dir, f'{model_name}_perceptual_gaussian_k{hard_k}_{mode}.pt'),
        ],
    }


def _standardize_feature_mapping(mapping: Dict[Any, Any]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for key, value in mapping.items():
        value_tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        result[str(key)] = value_tensor.float().cpu().clone()
    return result


def _standardize_multi_level_mapping(mapping: Dict[Any, Any]) -> Dict[str, List[torch.Tensor]]:
    result: Dict[str, List[torch.Tensor]] = {}
    for key, value in mapping.items():
        if not isinstance(value, (list, tuple)):
            raise TypeError(f'Multi-level feature for {key} must be a list/tuple, got {type(value)}')
        result[str(key)] = [v if torch.is_tensor(v) else torch.as_tensor(v) for v in value]
        result[str(key)] = [v.float().cpu().clone() for v in result[str(key)]]
    return result


def load_feature_store(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if torch.is_tensor(obj):
        return {'kind': 'tensor', 'data': obj.float().cpu(), 'path': path}
    if not isinstance(obj, dict):
        raise TypeError(f'Unsupported feature cache type at {path}: {type(obj)}')

    if obj.get('format') == 'multi_level_path_dict' and isinstance(obj.get('features_by_path'), dict):
        return {
            'kind': 'multi_level_path_dict',
            'data': _standardize_multi_level_mapping(obj['features_by_path']),
            'path': path,
            'level_names': list(obj.get('level_names', [])),
        }
    if isinstance(obj.get('features'), dict):
        layers = list(obj.get('layers', []))
        if len(layers) == 1 and isinstance(obj['features'].get(layers[0]), dict):
            return {
                'kind': 'path_dict',
                'data': _standardize_feature_mapping(obj['features'][layers[0]]),
                'path': path,
            }
    if 'features_by_path' in obj and isinstance(obj['features_by_path'], dict):
        return {'kind': 'path_dict', 'data': _standardize_feature_mapping(obj['features_by_path']), 'path': path}
    if 'img_features' in obj and isinstance(obj['img_features'], dict):
        return {'kind': 'path_dict', 'data': _standardize_feature_mapping(obj['img_features']), 'path': path}
    if 'tensor' in obj and torch.is_tensor(obj['tensor']):
        return {'kind': 'tensor', 'data': obj['tensor'].float().cpu(), 'path': path}
    if all(torch.is_tensor(v) or hasattr(v, 'shape') for v in obj.values()):
        return {'kind': 'path_dict', 'data': _standardize_feature_mapping(obj), 'path': path}
    raise TypeError(f'Cannot parse feature cache at {path}.')


def existing_paths(paths: Iterable[str]) -> List[str]:
    return [path for path in paths if os.path.exists(path)]


def pick_existing_feature_store(paths: Iterable[str]) -> Dict[str, Any]:
    checked = list(paths)
    for path in checked:
        if os.path.exists(path):
            return load_feature_store(path)
    raise FileNotFoundError('No feature cache found. Tried: ' + ', '.join(checked))


def _normalize_feature_key(path: Any) -> str:
    key = os.path.normpath(str(path)).replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def _feature_key_aliases(path: Any) -> List[str]:
    key = _normalize_feature_key(path)
    aliases = [key]

    for prefix in ("train_images/", "test_images/"):
        if key.startswith(prefix):
            aliases.append(key[len(prefix):])
    if key.startswith("train_images/"):
        aliases.append("training_images/" + key[len("train_images/"):])
    if key.startswith("training_images/"):
        aliases.append("train_images/" + key[len("training_images/"):])

    return list(dict.fromkeys(aliases))


def lookup_feature(store: Dict[str, Any], img_path: Any, index: int):
    if store['kind'] == 'tensor':
        raise RuntimeError(
            f'Legacy tensor cache detected at {store.get("path", "")}. '
            'Delete old feature caches and regenerate path-based caches first.'
        )

    data = store['data']
    aliases = _feature_key_aliases(img_path)

    for key in aliases:
        if key in data:
            return data[key]

    raise KeyError(
        f'Image path {img_path} not found in feature cache {store.get("path", "")}. '
        f'Tried aliases: {aliases}'
    )


def feature_store_dims(store: Dict[str, Any]) -> List[int]:
    if store['kind'] == 'tensor':
        if store['data'].ndim < 2:
            raise ValueError('Legacy tensor cache must have at least 2 dims.')
        return [int(store['data'].shape[-1])]
    sample = next(iter(store['data'].values()))
    if store['kind'] == 'multi_level_path_dict':
        return [int(v.shape[-1]) for v in sample]
    return [int(sample.shape[-1])]


def ensure_mctf_multilevel_cache(config: Dict[str, Any], mode: str, image_paths: Optional[Sequence[str]] = None) -> List[str]:
    mctf_cfg = resolve_mctf_config(config)
    if not mctf_cfg['enabled'] or mctf_cfg['feature_mode'] != 'multilevel_cache':
        return existing_paths(mctf_cfg.get('level_feature_paths', []))

    output_paths = list(mctf_cfg.get('level_feature_paths', []))
    auto_cfg = mctf_cfg.get('auto_extract', {})
    existing = existing_paths(output_paths)

    if existing:
        # Add missing image descriptors when a cache only covers part of the dataset.
        if image_paths:
            store = load_feature_store(existing[0])
            if store['kind'] in ('path_dict', 'multi_level_path_dict'):
                existing_keys = set(store['data'].keys())
                missing = []
                for p in image_paths:
                    aliases = _feature_key_aliases(p)
                    if not any(alias in existing_keys for alias in aliases):
                        missing.append(str(p))
                if missing:
                    logging.warning(
                        'Existing multilevel cache is incomplete for mode=%s (%d missing paths). '
                        'Adding missing paths to cache: %s. Example missing path: %s',
                        mode, len(missing), existing[0], missing[0]
                    )
                    existing = []
                else:
                    return existing
            else:
                return existing
        else:
            return existing

    from preprocess.extract_multilevel_teacher_features import (
        ensure_multilevel_feature_cache,
        infer_default_multilevel_levels,
    )

    if not auto_cfg.get('enabled', False):
        return existing

    data_cfg = config.get('data', {})
    model_type = data_cfg.get('model_type') or config.get('vision_backbone')
    if not model_type:
        raise ValueError('Cannot auto-extract multilevel cache because model_type/vision_backbone is missing in config.')

    image_root = auto_cfg.get('image_root') or infer_image_root(config)
    levels = auto_cfg.get('levels') or infer_default_multilevel_levels(model_type)
    batch_size = int(auto_cfg.get('batch_size', data_cfg.get('feature_batch_size', 256)))
    num_workers = int(
        auto_cfg.get(
            'num_workers',
            data_cfg.get('feature_num_workers', data_cfg.get('eval_num_workers', data_cfg.get('num_workers', 4)))
        )
    )
    glob_pattern = str(auto_cfg.get('glob_pattern', '**/*.*'))
    use_dataset_image_paths = bool(auto_cfg.get('use_dataset_image_paths', False))
    requested_paths = list(image_paths) if (use_dataset_image_paths and image_paths is not None) else None

    primary_output = output_paths[0]
    logging.info(
        'MCTF strict multilevel cache missing for mode=%s; auto-extracting backbone=%s levels=%s -> %s',
        mode, model_type, levels, primary_output,
    )

    result = ensure_multilevel_feature_cache(
        model_name=model_type,
        image_root=image_root,
        output_path=primary_output,
        levels=levels,
        image_paths=requested_paths,
        glob_pattern=glob_pattern,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    logging.info('MCTF strict multilevel cache ready: %s', result)
    return existing_paths(output_paths)


def resolve_mctf_level_sources(config: Dict[str, Any], mode: str, candidates: Dict[str, List[str]], image_paths: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    mctf_cfg = resolve_mctf_config(config)
    if not mctf_cfg['enabled']:
        return []

    sources: List[Dict[str, Any]] = []

    def _feature_kind_sources(kinds: Sequence[str], level_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        names = list(level_names or [])
        resolved_sources: List[Dict[str, Any]] = []
        for idx, kind in enumerate(kinds):
            if kind not in candidates:
                raise KeyError(f'Unknown MCTF fallback feature kind: {kind}. Available: {list(candidates.keys())}')
            level_name = names[idx] if idx < len(names) else kind
            resolved_sources.append({'name': level_name, 'kind': kind, 'store': pick_existing_feature_store(candidates[kind])})
        return resolved_sources

    if mctf_cfg['feature_mode'] == 'feature_kinds':
        return _feature_kind_sources(mctf_cfg['level_feature_kinds'], mctf_cfg['level_names'])

    existing_multilevel = ensure_mctf_multilevel_cache(config, mode, image_paths=image_paths)
    if not existing_multilevel:
        if mctf_cfg.get('allow_missing_multilevel_fallback', True):
            fallback_kinds = mctf_cfg.get('fallback_feature_kinds') or ['perceptual', 'clean', 'semantic']
            logging.warning(
                'MCTF multilevel cache not found for mode=%s. Falling back from feature_mode=multilevel_cache to feature_kinds=%s. Missing paths: %s',
                mode, fallback_kinds, mctf_cfg['level_feature_paths']
            )
            fallback_names = list(mctf_cfg.get('level_names') or [])[:len(fallback_kinds)]
            return _feature_kind_sources(fallback_kinds, fallback_names)
        raise FileNotFoundError('No MCTF multilevel feature cache found. Tried: ' + ', '.join(mctf_cfg['level_feature_paths']))

    configured_names = list(mctf_cfg.get('level_names') or [])
    name_offset = 0
    for idx, path in enumerate(existing_multilevel):
        store = pick_existing_feature_store([path])
        if store['kind'] == 'multi_level_path_dict':
            # A single cache may contain multiple ordered internal levels from one frozen teacher.
            store_level_names = list(store.get('level_names') or [])
            num_levels = len(feature_store_dims(store))
            if configured_names and len(configured_names) >= name_offset + num_levels:
                level_names = configured_names[name_offset:name_offset + num_levels]
            elif store_level_names:
                level_names = store_level_names
            else:
                level_names = [f'level_{i}' for i in range(num_levels)]
            for level_index, level_name in enumerate(level_names):
                sources.append({'name': level_name, 'kind': f'multilevel_{level_index}', 'store': store, 'level_index': level_index})
            name_offset += len(level_names)
            continue
        level_name = configured_names[name_offset] if name_offset < len(configured_names) else f'level_{name_offset}'
        sources.append({'name': level_name, 'kind': f'path_{idx}', 'store': store})
        name_offset += 1
    return sources


def infer_default_image_root(data_dir: str) -> str:
    return os.path.normpath(os.path.join(data_dir, '..', 'Image_set_Resize'))


def train_eval_workers(config: Dict[str, Any]) -> Dict[str, int]:
    data_cfg = config.get('data', {})
    train_workers = int(data_cfg.get('train_num_workers', data_cfg.get('num_workers', 4)))
    eval_workers = int(data_cfg.get('eval_num_workers', data_cfg.get('num_workers', 4)))
    return {'train': train_workers, 'eval': eval_workers}


OPENCLIP_PRETRAIN: Dict[str, str] = {
    'RN50': 'openai',
    'RN101': 'openai',
    'ViT-B-16': 'openai',
    'ViT-B-32': 'openai',
    'OpenAI-ViT-L-14': 'openai',
    'OpenAI-ViT-L-14-336': 'openai',
    'ViT-L-14': 'laion2b_s32b_b82k',
    'DFN-ViT-L-14': 'dfn2b',
    'DataComp-ViT-L-14': 'datacomp_xl_s13b_b90k',
    'SigLIP-B-16-256': 'webli',
    'SigLIP2-B-16-256': 'webli',
    'SigLIP2-L-16-256': 'webli',
    'SigLIP-SO400M-14-384': 'webli',
    'SigLIP-L-16-256': 'webli',
    'PE-Core-B-16': 'meta',
    'PE-Core-L-14-336': 'meta',
    'PE-Core-bigG-14-448': 'meta',
    'ViT-H-14': 'laion2b_s32b_b79k',
    'MetaCLIP-H-14': 'metaclip_fullcc',
    'ViT-H-14-CLIPA-336': 'datacomp1b',
    'DFN5B-ViT-H-14-378': 'dfn5b',
    'ViT-g-14': 'laion2b_s34b_b88k',
    'ViT-bigG-14': 'laion2b_s39b_b160k',
    'EVA02-B-16': 'merged2b_s8b_b131k',
    'EVA02-L-14': 'merged2b_s4b_b131k',
    'ConvNeXt-BW': 'laion2b_s13b_b82k_augreg',
    'MobileCLIP2-B': 'dfndr2b',
    'MetaCLIP-B-16': 'metaclip_fullcc',
}

OPENCLIP_LOCAL_CHECKPOINTS: Dict[str, str] = {
    'ViT-L-14': 'openclip_weights/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin',
    'DFN-ViT-L-14': 'openclip_weights/DFN2B-CLIP-ViT-L-14-39B/open_clip_pytorch_model.bin',
    'DataComp-ViT-L-14': 'openclip_weights/CLIP-ViT-L-14-DataComp-XL-s13B-b90K/open_clip_pytorch_model.bin',
    'SigLIP-B-16-256': 'openclip_weights/ViT-B-16-SigLIP-256-webli/open_clip_pytorch_model.bin',
    'SigLIP2-B-16-256': 'openclip_weights/ViT-B-16-SigLIP2-256-webli/open_clip_pytorch_model.bin',
    'SigLIP2-L-16-256': 'openclip_weights/ViT-L-16-SigLIP2-256-webli/open_clip_pytorch_model.bin',
    'SigLIP-SO400M-14-384': 'openclip_weights/ViT-SO400M-14-SigLIP-384-webli/open_clip_model.safetensors',
    'SigLIP-L-16-256': 'openclip_weights/ViT-L-16-SigLIP-256-webli/open_clip_pytorch_model.bin',
    'PE-Core-B-16': 'openclip_weights/PE-Core-B-16-meta/open_clip_model.safetensors',
    'PE-Core-L-14-336': 'openclip_weights/PE-Core-L-14-336-meta/open_clip_model.safetensors',
    'PE-Core-bigG-14-448': 'openclip_weights/PE-Core-bigG-14-448-meta/open_clip_model.safetensors',
    'ViT-H-14': 'openclip_weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
    'ViT-H-14-CLIPA-336': 'openclip_weights/ViT-H-14-CLIPA-336-datacomp1B/open_clip_model.safetensors',
    'DFN5B-ViT-H-14-378': 'openclip_weights/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin',
    'EVA02-B-16': 'openclip_weights/EVA02-B-16-merged2b-s8b-b131k/open_clip_pytorch_model.bin',
    'EVA02-L-14': 'openclip_weights/EVA02-L-14-merged2b-s4b-b131k/open_clip_pytorch_model.bin',
    'ConvNeXt-BW': 'openclip_weights/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/open_clip_pytorch_model.bin',
    'MobileCLIP2-B': 'openclip_weights/MobileCLIP2-B-OpenCLIP/open_clip_pytorch_model.bin',
    'MetaCLIP-B-16': 'openclip_weights/MetaCLIP-B-16-fullcc/open_clip_pytorch_model.bin',
}

OPENCLIP_PREPROCESS_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'PE-Core-B-16': {
        'image_mean': (0.5, 0.5, 0.5),
        'image_std': (0.5, 0.5, 0.5),
        'image_interpolation': 'bilinear',
        'image_resize_mode': 'squash',
    },
    'PE-Core-L-14-336': {
        'image_mean': (0.5, 0.5, 0.5),
        'image_std': (0.5, 0.5, 0.5),
        'image_interpolation': 'bilinear',
        'image_resize_mode': 'squash',
    },
    'PE-Core-bigG-14-448': {
        'image_mean': (0.5, 0.5, 0.5),
        'image_std': (0.5, 0.5, 0.5),
        'image_interpolation': 'bilinear',
        'image_resize_mode': 'squash',
    },
    'MetaCLIP-B-16': {
        'image_mean': (0.48145466, 0.4578275, 0.40821073),
        'image_std': (0.26862954, 0.26130258, 0.27577711),
        'image_interpolation': 'bicubic',
        'image_resize_mode': 'shortest',
    },
}

TIMM_PRETRAIN: Dict[str, Dict[str, str]] = {
    'Timm-SigLIP2-L-16-256': {
        'model_name': 'vit_large_patch16_siglip_256.v2_webli',
        'checkpoint': 'timm_weights/vit_large_patch16_siglip_256_v2_webli/model.safetensors',
        'format': 'safetensors',
    },
    'Timm-DINOv2-L-14': {
        'model_name': 'vit_large_patch14_dinov2.lvd142m',
        'checkpoint': 'timm_weights/dinov2_vitl14_lvd142m/dinov2_vitl14_pretrain.pth',
        'format': 'pth',
    },
}


class _PathDataset(Dataset):
    def __init__(self, image_root: str, image_paths: List[str], preprocess: Callable, transform: Callable):
        self.image_root = image_root
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        full_path = _resolve_image_file(self.image_root, rel_path)
        image = Image.open(full_path).convert('RGB')
        image = self.transform(image)
        image = self.preprocess(image)
        return rel_path, image


def _collate_paths(batch):
    paths = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch], dim=0)
    return paths, images


def _image_path_candidates(image_root: str, rel_path: str) -> List[str]:
    normalized = _normalize_feature_key(rel_path)
    candidates = [os.path.join(image_root, normalized)]
    if normalized.startswith("train_images/"):
        candidates.append(os.path.join(image_root, "training_images", normalized[len("train_images/"):]))
    if normalized.startswith("training_images/"):
        candidates.append(os.path.join(image_root, "train_images", normalized[len("training_images/"):]))
    return list(dict.fromkeys(candidates))


def _resolve_image_file(image_root: str, rel_path: str) -> str:
    for candidate in _image_path_candidates(image_root, rel_path):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'Image path {rel_path} not found under {image_root}. Tried: {_image_path_candidates(image_root, rel_path)}')


def resolve_feature_dir(config: Dict[str, Any]) -> str:
    data_cfg = config.get('data', {})
    feature_dir = data_cfg.get('feature_dir')
    if feature_dir:
        return os.path.normpath(feature_dir)
    # Preserve the original cache directory for configurations without an explicit path.
    feature_subdir = data_cfg.get('feature_subdir', 'HyFI')
    data_dir = data_cfg['data_dir']
    return os.path.normpath(os.path.join(data_dir, '..', 'Image_feature', feature_subdir))


def infer_image_root(config: Dict[str, Any]) -> str:
    data_cfg = config.get('data', {})
    image_root = data_cfg.get('image_root')
    if image_root:
        return os.path.normpath(image_root)
    return infer_default_image_root(data_cfg['data_dir'])


def _gaussian_transform(kernel_size: int) -> Callable:
    if kernel_size % 2 == 0:
        raise ValueError('Gaussian kernel size must be odd.')

    def _transform(image: Image.Image) -> Image.Image:
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        blurred_np = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
        return Image.fromarray(cv2.cvtColor(blurred_np, cv2.COLOR_BGR2RGB))

    return _transform


def _weights_root() -> Path:
    """Portable root for optional local visual encoder checkpoints."""
    default = Path(__file__).resolve().parents[3] / 'weights'
    return Path(os.path.expandvars(os.environ.get('BHVTANET_WEIGHTS_ROOT', str(default)))).expanduser()


def _configured_pretrained(backbone: str, vision_pretrained: Any) -> Optional[str]:
    if isinstance(vision_pretrained, dict) or hasattr(vision_pretrained, 'items'):
        value = vision_pretrained.get(backbone)
    else:
        value = vision_pretrained
    if value is None or str(value).strip() == '':
        return None
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if Path(value).is_file():
        return value
    if Path(value).suffix.lower() in {'.bin', '.safetensors', '.pth', '.pt'} or '/' in value or '\\' in value:
        raise FileNotFoundError(f'Visual encoder checkpoint not found: {value}')
    return value


def _local_checkpoint(relative_path: str) -> Optional[str]:
    path = _weights_root() / relative_path
    return str(path) if path.is_file() else None


def _build_openclip_encoder(backbone: str, device: str, vision_pretrained: Any = None):
    model_backbone = {
        'OpenAI-ViT-L-14': 'ViT-L-14',
        'OpenAI-ViT-L-14-336': 'ViT-L-14-336',
        'DFN-ViT-L-14': 'ViT-L-14',
        'DataComp-ViT-L-14': 'ViT-L-14',
        'SigLIP-B-16-256': 'ViT-B-16-SigLIP-256',
        'SigLIP2-B-16-256': 'ViT-B-16-SigLIP2-256',
        'SigLIP2-L-16-256': 'ViT-L-16-SigLIP2-256',
        'SigLIP-SO400M-14-384': 'ViT-SO400M-14-SigLIP-384',
        'SigLIP-L-16-256': 'ViT-L-16-SigLIP-256',
        'PE-Core-B-16': 'PE-Core-B-16',
        'PE-Core-L-14-336': 'PE-Core-L-14-336',
        'PE-Core-bigG-14-448': 'PE-Core-bigG-14-448',
        'DFN5B-ViT-H-14-378': 'ViT-H-14-378',
        'MetaCLIP-H-14': 'ViT-H-14',
        'MetaCLIP-B-16': 'ViT-B-16-quickgelu',
        'ConvNeXt-BW': 'convnext_base_w',
    }.get(backbone, backbone)
    clip_name_map = {
        'RN50': 'RN50',
        'RN101': 'RN101',
        'ViT-B-16': 'ViT-B/16',
        'ViT-B-32': 'ViT-B/32',
        'ViT-L-14': 'ViT-L/14',
        'ViT-L-14-336': 'ViT-L/14@336px',
        'OpenAI-ViT-L-14': 'ViT-L/14',
        'OpenAI-ViT-L-14-336': 'ViT-L/14@336px',
    }

    pretrained = _configured_pretrained(backbone, vision_pretrained)
    if pretrained is None:
        relative_path = OPENCLIP_LOCAL_CHECKPOINTS.get(backbone)
        pretrained = _local_checkpoint(relative_path) if relative_path else None
    if pretrained is None:
        pretrained = OPENCLIP_PRETRAIN[backbone]
    if model_backbone in clip_name_map and pretrained == 'openai':
        model, preprocess = clip.load(clip_name_map[model_backbone], device=device)
        model = model.eval()

        def encode_batch(images: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model.encode_image(images)

        return model, encode_batch, preprocess

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_backbone,
        pretrained=pretrained,
        device=device,
        **OPENCLIP_PREPROCESS_OVERRIDES.get(backbone, {}),
    )
    model = model.eval()

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.encode_image(images)

    return model, encode_batch, preprocess







def _build_timm_encoder(backbone: str, device: str, vision_pretrained: Any = None):
    import timm
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_data_config

    spec = TIMM_PRETRAIN[backbone]
    ckpt_path = _configured_pretrained(backbone, vision_pretrained)
    if ckpt_path is None:
        ckpt_path = _local_checkpoint(spec['checkpoint'])
    if ckpt_path is not None and not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'TIMM vision_pretrained must be a checkpoint file: {ckpt_path}')

    model = timm.create_model(spec['model_name'], pretrained=ckpt_path is None, num_classes=0)
    if ckpt_path is None:
        pass
    elif spec.get('format') == 'pth':
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [key for key in unexpected if key != 'mask_token']
        if missing or unexpected:
            raise RuntimeError(f'TIMM checkpoint mismatch for {backbone}: missing={missing[:8]}, unexpected={unexpected[:8]}')
    else:
        state = load_file(ckpt_path)
        model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    preprocess = create_transform(**resolve_data_config({}, model=model), is_training=False)

    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(images.to(device))

    return model, encode_batch, preprocess


def _build_encoder(backbone: str, device: str, vision_pretrained: Any = None):
    if backbone in OPENCLIP_PRETRAIN:
        return _build_openclip_encoder(backbone, device, vision_pretrained=vision_pretrained)
    if backbone in TIMM_PRETRAIN:
        return _build_timm_encoder(backbone, device, vision_pretrained=vision_pretrained)
    raise ValueError(f'Unsupported vision backbone: {backbone}')


def _save_feature_cache(output_path: str, feature_type: str, backbone: str, transform_name: str, features_by_path: Dict[str, torch.Tensor]):
    torch.save(
        {
            'format': 'path_dict',
            'feature_type': feature_type,
            'vision_backbone': backbone,
            'transform': transform_name,
            'features_by_path': features_by_path,
        },
        output_path,
    )


def _encode_feature_dict(encode_batch: Callable, preprocess: Callable, image_root: str, image_paths: List[str], transform: Callable, batch_size: int, num_workers: int, device: str) -> Dict[str, torch.Tensor]:
    dataset = _PathDataset(image_root=image_root, image_paths=image_paths, preprocess=preprocess, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.startswith('cuda'), collate_fn=_collate_paths)

    feature_dict: Dict[str, torch.Tensor] = {}
    for rel_paths, images in tqdm(loader, total=len(loader), desc='encoding image features', leave=False):
        images = images.to(device)
        feats = encode_batch(images).detach().float().cpu()
        for rel_path, feat in zip(rel_paths, feats):
            feature_dict[rel_path] = feat
    return feature_dict


def _identity_transform(image: Image.Image) -> Image.Image:
    return image


def ensure_visual_feature_caches(config: Dict[str, Any], mode: str, image_paths: Sequence[str]) -> str:
    from bhvtanet.data.image_transforms import FoveaBlur

    data_cfg = config.get('data', {})
    model_type = data_cfg['model_type']
    feature_dir = resolve_feature_dir(config)
    os.makedirs(feature_dir, exist_ok=True)

    soft_k = int(config['Blur']['soft_k'])
    soft_g = int(config['Blur']['soft_g'])
    hard_k = int(config['Blur']['hard_k'])
    candidates = feature_candidates(feature_dir, model_type, mode, soft_k, soft_g, hard_k)
    required_kinds = set(required_feature_kinds(config))
    mctf_cfg = resolve_mctf_config(config)
    if mctf_cfg.get('enabled', False) and mctf_cfg.get('feature_mode') == 'feature_kinds':
        required_kinds.update(mctf_cfg.get('level_feature_kinds', []))
    required_kinds = sorted(required_kinds)

    existing = {kind: any(os.path.exists(path) for path in candidates[kind]) for kind in required_kinds}
    if all(existing.values()):
        return feature_dir

    unique_paths = sorted({str(path) for path in image_paths})
    if not unique_paths:
        raise ValueError(f'No image paths found while preparing {mode} features.')

    image_root = infer_image_root(config)
    missing_images = []
    for rel in unique_paths[:10]:
        try:
            _resolve_image_file(image_root, rel)
        except FileNotFoundError:
            missing_images.append(rel)
    if missing_images:
        sample = ', '.join(missing_images[:3])
        raise FileNotFoundError(f'Image root {image_root} is missing files such as: {sample}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, encode_batch, preprocess = _build_encoder(
        model_type, device, vision_pretrained=data_cfg.get('vision_pretrained', config.get('vision_pretrained'))
    )
    batch_size = int(data_cfg.get('feature_batch_size', 256))
    num_workers = int(data_cfg.get('feature_num_workers', data_cfg.get('eval_num_workers', data_cfg.get('num_workers', 4))))

    transforms_by_kind = {
        'clean': (_identity_transform, 'identity'),
        'semantic': (FoveaBlur(h=224, w=224, blur_kernel_size=soft_k, curve_type='exp', system_g=soft_g), 'fovea_blur'),
        'perceptual': (_gaussian_transform(hard_k), 'gaussian_blur'),
    }
    backbone_name = canonical_vision_backbone_name(model_type)
    output_paths = {
        'clean': os.path.join(feature_dir, f'{backbone_name}_clean_{mode}.pt'),
        'semantic': os.path.join(feature_dir, f'{backbone_name}_semantic_fovea_k{soft_k}_g{soft_g}_{mode}.pt'),
        'perceptual': os.path.join(feature_dir, f'{backbone_name}_perceptual_gaussian_k{hard_k}_{mode}.pt'),
    }

    for kind in required_kinds:
        if existing[kind]:
            continue
        transform_fn, transform_name = transforms_by_kind[kind]
        features = _encode_feature_dict(encode_batch, preprocess, image_root, unique_paths, transform_fn, batch_size, num_workers, device)
        _save_feature_cache(output_paths[kind], kind, model_type, transform_name, features)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feature_dir


