"""Training, checkpoint selection and evaluation orchestration."""

import argparse
import glob
import json
import os
import shutil
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam, AdamW, SGD

from bhvtanet.models.ablation import resolve_ablation_config
from bhvtanet.models.mctf import MCTF, resolve_mctf_config
from bhvtanet.data.common import infer_channel_count
from bhvtanet.geometry import lorentz as L
from bhvtanet.models.retrieval import RetrievalObjective, resolve_retrieval_config
from bhvtanet.training.config import build_warmup_cosine_scheduler, resolve_train_config, to_jsonable
from bhvtanet.training.utils import instantiate_from_config, update_config
from bhvtanet.training.paths import PROJECT_ROOT, register_path_resolvers, resolve_project_paths
from bhvtanet.training.checkpoint_io import LocalAtomicCheckpointIO



PRETRAIN_MAP = {
    'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
    'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
    'ViT-B-16': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
    'ViT-B-32': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
    'OpenAI-ViT-L-14': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 768},
    'OpenAI-ViT-L-14-336': {'pretrained': 'openai', 'resize': (336, 336), 'z_dim': 768},
    'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
    'DFN-ViT-L-14': {'pretrained': 'dfn2b_s39b', 'resize': (224, 224), 'z_dim': 768},
    'DataComp-ViT-L-14': {'pretrained': 'datacomp_xl_s13b_b90k', 'resize': (224, 224), 'z_dim': 768},
    'SigLIP-B-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 768},
    'SigLIP2-B-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 768},
    'SigLIP2-L-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 1024},
    'Timm-SigLIP2-L-16-256': {'pretrained': 'webli-image-tower', 'resize': (256, 256), 'z_dim': 1024},
    'SigLIP-SO400M-14-384': {'pretrained': 'webli', 'resize': (384, 384), 'z_dim': 1152},
    'Timm-DINOv2-L-14': {'pretrained': 'lvd142m', 'resize': (518, 518), 'z_dim': 1024},
    'SigLIP-L-16-256': {'pretrained': 'webli', 'resize': (256, 256), 'z_dim': 1024},
    'PE-Core-B-16': {'pretrained': 'meta', 'resize': (224, 224), 'z_dim': 1024},
    'PE-Core-L-14-336': {'pretrained': 'meta', 'resize': (336, 336), 'z_dim': 1024},
    'PE-Core-bigG-14-448': {'pretrained': 'meta', 'resize': (448, 448), 'z_dim': 1280},
    'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
    'MetaCLIP-H-14': {'pretrained': 'metaclip_fullcc', 'resize': (224, 224), 'z_dim': 1024},
    'ViT-H-14-CLIPA-336': {'pretrained': 'datacomp1b', 'resize': (336, 336), 'z_dim': 1024},
    'DFN5B-ViT-H-14-378': {'pretrained': 'dfn5b', 'resize': (378, 378), 'z_dim': 1024},
    'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
    'EVA02-B-16': {'pretrained': 'merged2b_s8b_b131k', 'resize': (224, 224), 'z_dim': 512},
    'EVA02-L-14': {'pretrained': 'merged2b_s4b_b131k', 'resize': (224, 224), 'z_dim': 768},
    'ConvNeXt-BW': {'pretrained': 'laion2b_s13b_b82k_augreg', 'resize': (224, 224), 'z_dim': 640},
    'MobileCLIP2-B': {'pretrained': 'dfndr2b', 'resize': (224, 224), 'z_dim': 512},
    'MetaCLIP-B-16': {'pretrained': 'metaclip_fullcc', 'resize': (224, 224), 'z_dim': 512},
    'InternViT-L28': {'pretrained': 'cache', 'resize': (448, 448), 'z_dim': 512, 'feature_dim': 3200},
}


HYPERBOLIC_LOGIT_SCALE = 100.0029525756836


def _unwrap_dataset(dataset):
    base = dataset
    while hasattr(base, 'dataset'):
        base = base.dataset
    return base


def load_model(config, train_loader, test_loader):
    model = {}
    for key, value in config['models'].items():
        print(f'init {key}')
        model[key] = instantiate_from_config(value)
    return BHVTANet(model, config, train_loader, test_loader)



from bhvtanet.models.bhvtanet import BHVTANet


def parse_subjects_arg(subjects_arg, dataset):
    if subjects_arg is None:
        subjects_arg = 'all'
    token = str(subjects_arg).strip()
    if token == '' or token.lower() == 'all':
        total = 10 if dataset == 'eeg' else 4
        return [f'sub-{i:02d}' for i in range(1, total + 1)]

    subjects = []
    for part in token.split(','):
        part = part.strip()
        if not part:
            continue
        if part.startswith('sub-'):
            subjects.append(part)
        elif part.isdigit():
            subjects.append(f'sub-{int(part):02d}')
        else:
            raise ValueError(f'Unsupported subject spec: {part}')
    return subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/eeg_intra.yaml', help='configuration path, relative to the project root')
    parser.add_argument('--data-root', type=str, default=None, help='directory containing things-eeg and things-meg')
    parser.add_argument('--anchor-root', type=str, default=None, help='directory containing EEG anchor feature folders')
    parser.add_argument('--feature-subdir', type=str, default=None, help='visual cache folder name below Image_feature')
    parser.add_argument('--output-root', type=str, default=None, help='override the result directory')
    parser.add_argument('--dataset', type=str, default=None, choices=['eeg', 'meg'], help='dataset override')
    parser.add_argument('--subjects', type=str, default='all', help='single subject like sub-01, comma-separated list, or all')
    parser.add_argument('--seed', type=int, default=200, help='random seed')
    parser.add_argument('--exp_setting', type=str, default=None, choices=['intra-subject', 'inter-subject'], help='protocol override')
    parser.add_argument('--epoch', type=int, default=None, help='maximum training epochs')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--brain_backbone', type=str, default=None, help='brain encoder override')
    parser.add_argument('--vision_backbone', type=str, default=None, help='visual encoder override')
    parser.add_argument('--preset', type=str, default=None, choices=['baseline', 'semantic_only', 'full', 'innov1'], help='ablation preset override')
    parser.add_argument('--retrieval_preset', type=str, default=None, help='retrieval preset override: original | xbm | siglip | vsepp | xbm_siglip | xbm_vsepp')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='evaluate an existing checkpoint without training; use {subject} as a placeholder for multi-subject evaluation',
    )
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    opt = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if opt.gpu == 'cpu' else opt.gpu
    if opt.data_root:
        os.environ['BHVTANET_DATA_ROOT'] = str(Path(opt.data_root).expanduser().resolve())
    if opt.anchor_root:
        os.environ['BHVTANET_ANCHOR_ROOT'] = str(Path(opt.anchor_root).expanduser().resolve())
    if opt.feature_subdir:
        os.environ['BHVTANET_FEATURE_SUBDIR'] = opt.feature_subdir
    register_path_resolvers()
    config_path = Path(opt.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    opt.config = str(config_path.resolve())
    seed_everything(opt.seed, workers=True)
    config = OmegaConf.load(opt.config)
    config = update_config(opt, config)
    for key in ('exp_setting', 'epoch', 'lr', 'brain_backbone', 'vision_backbone'):
        if getattr(opt, key) is None:
            setattr(opt, key, config[key])

    dataset = str(opt.dataset or config.get('dataset', 'eeg')).lower()
    if dataset not in {'eeg', 'meg'}:
        raise ValueError(f'Unsupported dataset: {dataset}')
    config['dataset'] = dataset

    if 'ablation' not in config or config['ablation'] is None:
        config['ablation'] = {}
    if opt.preset is not None:
        config['ablation']['preset'] = opt.preset
    config['ablation'] = resolve_ablation_config(config)

    if 'retrieval' not in config or config['retrieval'] is None:
        config['retrieval'] = {}
    if opt.retrieval_preset is not None:
        config['retrieval']['preset'] = opt.retrieval_preset
    config['retrieval'] = resolve_retrieval_config(config)

    config['mctf'] = resolve_mctf_config(config)

    config['c_num'] = infer_channel_count(config['data'].get('selected_ch'), dataset)

    if opt.vision_backbone not in PRETRAIN_MAP:
        raise ValueError(f"Unsupported vision backbone: {opt.vision_backbone}")

    vision_meta = PRETRAIN_MAP[opt.vision_backbone]
    config['visual_feature_dim'] = int(vision_meta.get('feature_dim', vision_meta['z_dim']))
    config['z_dim'] = int(config.get('align_dim', vision_meta['z_dim']))
    if 'models' in config and 'brain' in config['models']:
        config['models']['brain'].setdefault('params', {})
        config['models']['brain']['params'].setdefault('image_dim', config['visual_feature_dim'])

    config = resolve_project_paths(config)
    if opt.output_root:
        config['save_dir'] = str(Path(opt.output_root).expanduser().resolve())

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'

    subjects = parse_subjects_arg(opt.subjects, dataset)
    top1_acc_list = []
    top5_acc_list = []
    mac_list = []
    score_list = []

    if dataset == 'eeg':
        from bhvtanet.data.eeg import load_data
    else:
        from bhvtanet.data.meg import load_data

    for subject_idx in subjects:
        print('subject_idx: ', subject_idx)
        config['data']['subjects'] = [subject_idx]

        os.makedirs(config['save_dir'], exist_ok=True)
        logger = TensorBoardLogger(
            config['save_dir'],
            name=config['name'],
            version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}",
        )
        os.makedirs(logger.log_dir, exist_ok=True)
        shutil.copyfile(opt.config, os.path.join(logger.log_dir, os.path.basename(opt.config)))
        OmegaConf.save(config, os.path.join(logger.log_dir, 'resolved_config.yaml'), resolve=True)

        train_loader, val_loader, test_loader = load_data(config)
        train_subjects = list(getattr(_unwrap_dataset(train_loader.dataset), 'subjects', []))
        val_subjects = list(getattr(_unwrap_dataset(val_loader.dataset), 'subjects', []))
        test_subjects = list(getattr(_unwrap_dataset(test_loader.dataset), 'subjects', []))
        print(f'train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}')
        print(f'[Data split] train_subjects={train_subjects}, val_subjects={val_subjects}, test_subjects={test_subjects}')

        with open(os.path.join(logger.log_dir, 'run_meta.json'), 'w', encoding='utf-8') as f:
            meta = {
                'dataset': str(dataset),
                'subject': str(subject_idx),
                'subjects': list(config['data']['subjects']),
                'config': str(opt.config),
                'config_name': str(os.path.basename(opt.config)),
                'preset': str(config['ablation']['preset']),
                'ablation': dict(config['ablation']),
                'retrieval_preset': str(config['retrieval']['preset']),
                'retrieval': dict(config['retrieval']),
                'mctf': dict(config['mctf']),
                'aux_teacher': dict(config.get('aux_teacher', {})) if hasattr(config.get('aux_teacher', {}), 'items') else config.get('aux_teacher', {}),
                'target_adapter': dict(config.get('target_adapter', {})) if hasattr(config.get('target_adapter', {}), 'items') else config.get('target_adapter', {}),
                'eval_calibration': dict(config.get('eval_calibration', {})) if hasattr(config.get('eval_calibration', {}), 'items') else config.get('eval_calibration', {}),
                'exp_setting': str(config['exp_setting']),
                'brain_backbone': str(opt.brain_backbone),
                'vision_backbone': str(opt.vision_backbone),
                'epoch': int(config['train']['epoch']),
                'lr': float(config['train']['lr']),
                'seed': int(config['seed']),
                'save_dir': str(config['save_dir']),
                'experiment_name': str(config['name']),
                'environment': {
                    'python': platform.python_version(), 'torch': torch.__version__,
                    'lightning': pl.__version__, 'cuda': torch.version.cuda,
                    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                    'platform': platform.platform(),
                },
                'uses_test_as_validation': val_loader.dataset is test_loader.dataset,
                'dg': dict(config.get('dg', {})) if hasattr(config.get('dg', {}), 'items') else config.get('dg', {}),
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'test_subjects': test_subjects,
            }
            json.dump(to_jsonable(meta), f, indent=2, ensure_ascii=False)

        pl_model = load_model(config, train_loader, test_loader)

        checkpoint_cfg = config['train'].get('checkpoint', {})
        checkpoint_enabled = bool(checkpoint_cfg.get('enabled', True))
        checkpoint_callback = None
        if checkpoint_enabled:
            checkpoint_callback = ModelCheckpoint(
                monitor=str(config['train'].get('monitor', 'val_score')),
                mode=str(config['train'].get('monitor_mode', 'max')),
                save_top_k=1,
                save_last=True,
                filename='{epoch:02d}',
            )
        early_stop_callback = EarlyStopping(
            monitor=str(config['train'].get('monitor', 'val_score')),
            min_delta=float(config['train'].get('early_stop_min_delta', 5e-4)),
            patience=int(config['train'].get('early_stop_patience', 15)),
            verbose=False,
            mode=str(config['train'].get('monitor_mode', 'max')),
        )

        callbacks = [early_stop_callback]
        if checkpoint_callback is not None:
            callbacks.append(checkpoint_callback)
        swa_cfg = config['train'].get('swa', {})
        if bool(swa_cfg.get('enabled', False)):
            swa_lrs = swa_cfg.get('lr', None)
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=float(swa_lrs) if swa_lrs is not None else float(config['train']['lr']) * 0.5,
                    swa_epoch_start=swa_cfg.get('epoch_start', 0.75),
                    annealing_epochs=int(swa_cfg.get('annealing_epochs', 10)),
                )
            )

        trainer = Trainer(
            log_every_n_steps=10,
            callbacks=callbacks,
            max_epochs=config['train']['epoch'],
            min_epochs=int(config['train'].get('min_epoch', 0)),
            devices=1,
            accelerator=accelerator,
            logger=logger,
            gradient_clip_val=float(config['train'].get('gradient_clip_val', 0.0)),
            enable_checkpointing=checkpoint_enabled,
            plugins=[LocalAtomicCheckpointIO()],
        )
        print(trainer.logger.log_dir)

        if opt.checkpoint:
            checkpoint_path = opt.checkpoint.format(subject=subject_idx)
            if os.path.isdir(checkpoint_path):
                candidates = sorted(glob.glob(os.path.join(checkpoint_path, 'epoch=*.ckpt')))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f'Expected exactly one best epoch checkpoint in {checkpoint_path}, found: {candidates}'
                    )
                checkpoint_path = candidates[0]
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = payload.get('state_dict', payload) if isinstance(payload, dict) else payload
            pl_model.load_state_dict(state_dict, strict=True)
            # MDTP's progressive depth prior is active during evaluation too.
            # Lightning's standalone test loop resets current_epoch, so restore
            # the epoch recorded in the trusted local checkpoint explicitly.
            if isinstance(payload, dict) and 'epoch' in payload:
                trainer.fit_loop.epoch_progress.current.completed = int(payload['epoch'])
            test_results = trainer.test(model=pl_model, dataloaders=test_loader, ckpt_path=None)
            with open(os.path.join(logger.log_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=4)

            result = test_results[0]
            top1_acc_list.append(result.get('test_top1_acc'))
            top5_acc_list.append(result.get('test_top5_acc'))
            mac_list.append(result.get('mAP'))
            score_list.append(result.get('score'))
            continue

        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
        meta_path = os.path.join(logger.log_dir, 'run_meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            best_model_path = getattr(checkpoint_callback, 'best_model_path', '') if checkpoint_callback is not None else ''
            best_model_score = getattr(checkpoint_callback, 'best_model_score', None) if checkpoint_callback is not None else None
            meta.update({
                'fast_early_stop': bool(config.get('fast_early_stop', {}).get('enabled', False)),
                'max_epoch': int(config['train']['epoch']),
                'min_epoch': int(config['train'].get('min_epoch', 0)),
                'patience': int(config['train'].get('early_stop_patience', 0)),
                'monitor_metric': str(config['train'].get('monitor', 'val_score')),
                'best_checkpoint': str(best_model_path),
                'best_model_score': float(best_model_score.detach().cpu().item()) if best_model_score is not None else None,
                'best_epoch': os.path.basename(str(best_model_path)).replace('epoch=', '').replace('.ckpt', '') if best_model_path else '',
            })
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(to_jsonable(meta), f, indent=2, ensure_ascii=False)
        test_ckpt = str(config['train'].get('test_ckpt', 'best'))
        if not checkpoint_enabled and test_ckpt in {'best', 'last'}:
            test_ckpt = 'none'
        if test_ckpt == 'last':
            # Checkpoints are created by this local run and include Lightning callback state.
            # PyTorch 2.6 defaults to weights_only=True, which cannot restore that state.
            test_results = trainer.test(ckpt_path='last', dataloaders=test_loader, weights_only=False)
        elif test_ckpt == 'none':
            test_results = trainer.test(model=pl_model, dataloaders=test_loader, ckpt_path=None)
        else:
            test_results = trainer.test(ckpt_path='best', dataloaders=test_loader, weights_only=False)

        with open(os.path.join(logger.log_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=4)

        result = test_results[0]
        top1_acc_list.append(result.get('test_top1_acc'))
        top5_acc_list.append(result.get('test_top5_acc'))
        mac_list.append(result.get('mAP'))
        score_list.append(result.get('score'))

    print('top1_acc_list:', np.array(top1_acc_list).mean() if top1_acc_list else 0)
    print(top1_acc_list)
    print('top5_acc_list:', np.array(top5_acc_list).mean() if top5_acc_list else 0)
    print(top5_acc_list)
    print('mac_list:', np.array(mac_list).mean() if mac_list else 0)
    print(mac_list)
    print('score_list:', np.array(score_list).mean() if score_list else 0)
    print(score_list)


if __name__ == '__main__':
    main()
