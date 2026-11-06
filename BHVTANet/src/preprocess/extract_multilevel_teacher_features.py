import argparse
import glob
import json
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bhvtanet.data.common import (
    _build_encoder,
    _identity_transform,
    _resolve_image_file,
    canonical_vision_backbone_name,
)

DEFAULT_LEVELS = {
    'RN50': ['layer1', 'layer2', 'layer3', 'attnpool'],
    'RN101': ['layer1', 'layer2', 'layer3', 'attnpool'],
    'ViT-B-16': ['2', '5', '8', '11'],
    'ViT-B-32': ['2', '5', '8', '11'],
    'OpenAI-ViT-L-14': ['5', '11', '17', '23'],
    'OpenAI_ViT_L_14': ['5', '11', '17', '23'],
    'OpenAI-ViT-L-14-336': ['5', '11', '17', '23'],
    'OpenAI_ViT_L_14_336': ['5', '11', '17', '23'],
    'ViT-L-14': ['5', '11', '17', '23'],
    'DFN-ViT-L-14': ['5', '11', '17', '23'],
    'DFN_ViT_L_14': ['5', '11', '17', '23'],
    'DataComp-ViT-L-14': ['5', '11', '17', '23'],
    'DataComp_ViT_L_14': ['5', '11', '17', '23'],
    'SigLIP-B-16-256': ['2', '5', '8', '11'],
    'SigLIP_B_16_256': ['2', '5', '8', '11'],
    'SigLIP2-B-16-256': ['2', '5', '8', '11'],
    'SigLIP2_B_16_256': ['2', '5', '8', '11'],
    'SigLIP2-L-16-256': ['5', '11', '17', '23'],
    'SigLIP2_L_16_256': ['5', '11', '17', '23'],
    'Timm-SigLIP2-L-16-256': ['5', '11', '17', '23'],
    'Timm_SigLIP2_L_16_256': ['5', '11', '17', '23'],
    'SigLIP-SO400M-14-384': ['6', '13', '20', '26'],
    'SigLIP_SO400M_14_384': ['6', '13', '20', '26'],
    'Timm-DINOv2-L-14': ['5', '11', '17', '23'],
    'Timm_DINOv2_L_14': ['5', '11', '17', '23'],
    'SigLIP-L-16-256': ['5', '11', '17', '23'],
    'SigLIP_L_16_256': ['5', '11', '17', '23'],
    'PE-Core-B-16': ['2', '5', '8', '11'],
    'PE_Core_B_16': ['2', '5', '8', '11'],
    'PE-Core-L-14-336': ['5', '11', '17', '23'],
    'PE_Core_L_14_336': ['5', '11', '17', '23'],
    'ViT-H-14': ['7', '15', '23', '31'],
    'MetaCLIP-H-14': ['7', '15', '23', '31'],
    'MetaCLIP_H_14': ['7', '15', '23', '31'],
    'ViT-H-14-CLIPA-336': ['7', '15', '23', '31'],
    'ViT_H_14_CLIPA_336': ['7', '15', '23', '31'],
    'ViT-g-14': ['9', '19', '29', '39'],
    'ViT-bigG-14': ['11', '23', '35', '47'],
    'EVA02-B-16': ['2', '5', '8', '11'],
    'EVA02_B_16': ['2', '5', '8', '11'],
    'EVA02-L-14': ['5', '11', '17', '23'],
    'EVA02_L_14': ['5', '11', '17', '23'],
    'ConvNeXt-BW': ['0', '1', '2', '3'],
    'ConvNeXt_BW': ['0', '1', '2', '3'],
    'MobileCLIP2-B': ['2', '5', '8', '11'],
    'MobileCLIP2_B': ['2', '5', '8', '11'],
    'MetaCLIP-B-16': ['2', '5', '8', '11'],
    'MetaCLIP_B_16': ['2', '5', '8', '11'],
}


def infer_default_multilevel_levels(model_name: str) -> List[str]:
    canonical = canonical_vision_backbone_name(model_name)
    levels = DEFAULT_LEVELS.get(canonical) or DEFAULT_LEVELS.get(model_name)
    if levels is None:
        raise ValueError(
            f'No default multilevel levels configured for backbone={model_name}. '
            'Please specify mctf.auto_extract.levels explicitly.'
        )
    return list(levels)


def _pool_feature(output) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim == 4:
        return output.mean(dim=(-1, -2))
    if output.ndim == 3:
        if output.shape[0] < output.shape[1]:
            return output[0]
        return output[:, 0]
    if output.ndim == 2:
        return output
    raise ValueError(f'Unsupported hooked output shape: {tuple(output.shape)}')


class _ImageDataset(Dataset):
    def __init__(self, image_root: str, image_paths: List[str], preprocess: Callable, transform: Callable):
        self.image_root = image_root
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        image = Image.open(_resolve_image_file(self.image_root, rel_path)).convert('RGB')
        image = self.transform(image)
        image = self.preprocess(image)
        return rel_path, image


def _collate(batch):
    paths = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch], dim=0)
    return paths, images


def _resolve_hooks(model, backbone: str, levels: Sequence[str]):
    visual = getattr(model, 'visual', model)
    requested = [str(token).strip() for token in levels if str(token).strip()]

    hooks: OrderedDict[str, torch.nn.Module] = OrderedDict()

    if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
        blocks = visual.transformer.resblocks
        total = len(blocks)
        for token in requested:
            idx = int(token)
            if idx < 0:
                idx = total + idx
            hooks[f'block_{idx}'] = blocks[idx]
        return hooks

    if hasattr(visual, 'layer1') and hasattr(visual, 'layer4'):
        mapping = {
            'layer1': visual.layer1,
            'layer2': visual.layer2,
            'layer3': visual.layer3,
            'layer4': visual.layer4,
            'attnpool': getattr(visual, 'attnpool', visual.layer4),
        }
        for token in requested:
            if token not in mapping:
                raise ValueError(f'Unsupported RN level token: {token}')
            hooks[token] = mapping[token]
        return hooks

    if hasattr(visual, 'trunk') and hasattr(visual.trunk, 'blocks'):
        blocks = visual.trunk.blocks
        total = len(blocks)
        for token in requested:
            idx = int(token)
            if idx < 0:
                idx = total + idx
            hooks[f'block_{idx}'] = blocks[idx]
        return hooks

    if hasattr(visual, 'trunk') and hasattr(visual.trunk, 'stages'):
        stages = visual.trunk.stages
        total = len(stages)
        for token in requested:
            idx = int(token)
            if idx < 0:
                idx = total + idx
            hooks[f'stage_{idx}'] = stages[idx]
        return hooks

    if hasattr(visual, 'blocks'):
        blocks = visual.blocks
        total = len(blocks)
        for token in requested:
            idx = int(token)
            if idx < 0:
                idx = total + idx
            hooks[f'block_{idx}'] = blocks[idx]
        return hooks

    raise ValueError(f'Unsupported model for multi-level extraction: {backbone}')


def _resolve_image_paths(image_root: str, image_paths: Optional[Sequence[str]] = None, glob_pattern: str = '**/*.*') -> List[str]:
    if image_paths is not None:
        unique = []
        seen = set()
        for item in image_paths:
            rel = os.path.normpath(str(item))
            if rel not in seen:
                seen.add(rel)
                unique.append(rel)
        return sorted(unique)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_files = sorted([
        p for p in glob.glob(os.path.join(image_root, glob_pattern), recursive=True)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in extensions
    ])
    return [os.path.relpath(p, image_root) for p in image_files]


def extract_multilevel_feature_cache(
    model_name: str,
    image_root: str,
    output_path: str,
    levels: Optional[Sequence[str]] = None,
    image_paths: Optional[Sequence[str]] = None,
    glob_pattern: str = '**/*.*',
    batch_size: int = 256,
    num_workers: int = 4,
    force: bool = False,
) -> Dict[str, object]:
    existing_payload = None
    existing_features: Dict[str, List[torch.Tensor]] = {}
    if (not force) and os.path.exists(output_path):
        payload = torch.load(output_path, map_location='cpu', weights_only=False)
        existing_payload = payload if isinstance(payload, dict) else None
        if existing_payload is not None and isinstance(existing_payload.get('features_by_path'), dict):
            existing_features = dict(existing_payload.get('features_by_path', {}))
        if image_paths is None:
            return {
                'output': output_path,
                'num_images': len(existing_features),
                'levels': list(payload.get('level_names', [])) if isinstance(payload, dict) else [],
                'skipped': True,
            }

        requested_paths = _resolve_image_paths(image_root=image_root, image_paths=image_paths, glob_pattern=glob_pattern)
        missing_paths = [rel for rel in requested_paths if rel not in existing_features]
        if not missing_paths:
            return {
                'output': output_path,
                'num_images': len(existing_features),
                'levels': list(payload.get('level_names', [])) if isinstance(payload, dict) else [],
                'skipped': True,
            }
        image_paths = missing_paths

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    levels = list(levels or infer_default_multilevel_levels(model_name))
    model, _encode_batch_unused, preprocess = _build_encoder(model_name, device)
    model = model.eval()

    hooks = _resolve_hooks(model, model_name, levels)
    handles = []
    hook_buffers: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, module in hooks.items():
        def _make_hook(key):
            def _hook(_module, _inputs, output):
                hook_buffers[key] = _pool_feature(output).detach().float().cpu()
            return _hook
        handles.append(module.register_forward_hook(_make_hook(name)))

    rel_paths = _resolve_image_paths(image_root=image_root, image_paths=image_paths, glob_pattern=glob_pattern)
    if not rel_paths:
        raise ValueError(f'No images found under image_root={image_root} with glob_pattern={glob_pattern}')
    missing = [rel for rel in rel_paths[:32] if not os.path.exists(os.path.join(image_root, rel))]
    if missing:
        raise FileNotFoundError(f'Image root {image_root} is missing files such as: {missing[:3]}')

    dataset = _ImageDataset(image_root, rel_paths, preprocess, _identity_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith('cuda'),
        collate_fn=_collate,
    )

    features_by_path: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for batch_paths, images in tqdm(loader, total=len(loader), desc='extract multilevel features'):
            hook_buffers.clear()
            images = images.to(device)
            _ = model.encode_image(images)
            batch_level_features = [hook_buffers[name] for name in hooks.keys()]
            for sample_idx, rel_path in enumerate(batch_paths):
                features_by_path[rel_path] = [feat[sample_idx].clone() for feat in batch_level_features]

    for handle in handles:
        handle.remove()

    payload = {
        'format': 'multi_level_path_dict',
        'vision_backbone': canonical_vision_backbone_name(model_name),
        'level_names': list(hooks.keys()),
        'features_by_path': {**existing_features, **features_by_path},
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'output': output_path, 'num_images': len(features_by_path), 'levels': list(hooks.keys()), 'skipped': False}


def ensure_multilevel_feature_cache(
    model_name: str,
    image_root: str,
    output_path: str,
    levels: Optional[Sequence[str]] = None,
    image_paths: Optional[Sequence[str]] = None,
    glob_pattern: str = '**/*.*',
    batch_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, object]:
    return extract_multilevel_feature_cache(
        model_name=model_name,
        image_root=image_root,
        output_path=output_path,
        levels=levels,
        image_paths=image_paths,
        glob_pattern=glob_pattern,
        batch_size=batch_size,
        num_workers=num_workers,
        force=False,
    )


def main():
    parser = argparse.ArgumentParser(description='Extract multi-level teacher features into a path-based cache.')
    parser.add_argument('--model', required=True, help='vision backbone, e.g. RN50 or ViT-B-16')
    parser.add_argument('--image-root', required=True, help='root directory containing relative image paths')
    parser.add_argument('--glob-pattern', default='**/*.*', help='glob pattern under image-root')
    parser.add_argument('--output', required=True, help='output .pt path')
    parser.add_argument('--levels', default=None, help='comma separated hook spec, e.g. 3,7,11 for ViT or layer1,layer3,attnpool for RN50')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    levels = None if args.levels in (None, '', 'auto') else [token.strip() for token in args.levels.split(',') if token.strip()]
    result = ensure_multilevel_feature_cache(
        model_name=args.model,
        image_root=args.image_root,
        output_path=args.output,
        levels=levels,
        image_paths=None,
        glob_pattern=args.glob_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
