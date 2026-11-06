import logging
import os
from typing import Dict

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, random_split

from bhvtanet.models.ablation import resolve_ablation_config
from bhvtanet.data.augment import BrainAugmentor, resolve_augmentation_config
from bhvtanet.data.common import (
    EEG_CHANNELS,
    deep_copy_config,
    feature_candidates,
    lookup_feature,
    normalize_selected_channels,
    ensure_visual_feature_caches,
    pick_existing_feature_store,
    resolve_mctf_level_sources,
    feature_store_dims,
    resolve_feature_dir,
    train_eval_workers,
)


def resolve_inter_subject_split(all_subjects, test_subjects, val_subjects='auto'):
    source_subjects = [s for s in all_subjects if s not in test_subjects]
    if len(source_subjects) < 2:
        raise ValueError(
            f'Need at least 2 source subjects for inter-subject split, got source_subjects={source_subjects}, test_subjects={test_subjects}'
        )

    if val_subjects in (None, 'auto'):
        val_subjects = [source_subjects[-1]]
    elif isinstance(val_subjects, str):
        val_subjects = [s.strip() for s in val_subjects.split(',') if s.strip()]
    else:
        val_subjects = list(val_subjects)

    invalid_val = [s for s in val_subjects if s not in source_subjects]
    if invalid_val:
        raise ValueError(
            f'Validation subjects must come from source subjects. invalid_val={invalid_val}, source_subjects={source_subjects}'
        )

    train_subjects = [s for s in source_subjects if s not in val_subjects]
    if not train_subjects:
        raise ValueError(
            f'Empty train_subjects after split. source_subjects={source_subjects}, val_subjects={val_subjects}'
        )

    return train_subjects, val_subjects



def _canonical_img_array(img_raw, avg: bool):
    arr = np.asarray(img_raw, dtype=object)
    if avg:
        if arr.ndim == 0:
            return np.asarray([str(arr.item())])
        if arr.ndim == 1:
            return np.asarray([str(x) for x in arr.tolist()])
        # repeated-trial paths per stimulus -> use the first path as canonical key
        return np.asarray([str(np.asarray(x).reshape(-1)[0]) for x in arr])
    return np.asarray([str(x) for x in arr.reshape(-1).tolist()])


def _resolve_eval_batch_size(cfg_value, dataset_len: int) -> int:
    if int(cfg_value) <= 0:
        return max(1, int(dataset_len))
    return int(cfg_value)


class MEGDataset(Dataset):
    def __init__(self, config: Dict, mode: str):
        self.config = config
        self.mode = mode
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        self.model_type = config['data']['model_type']
        self.selected_ch = normalize_selected_channels(config['data']['selected_ch'])
        self.channels = EEG_CHANNELS
        self.avg = config['data'][f'{mode}_avg']
        self.timesteps = config['data']['timesteps']
        self.per_trials = 1 if self.mode == 'train' else 12
        self.partial_avg_k = int(config['data'].get('train_partial_avg_k', 1))
        self.partial_avg_repeats = int(config['data'].get('train_partial_avg_repeats', 1))
        self.partial_avg_active = bool(
            self.mode == 'train' and (not self.avg) and self.partial_avg_k > 1 and self.partial_avg_repeats > 0
        )
        self.augment_cfg = resolve_augmentation_config(config)
        self.augment = BrainAugmentor(self.augment_cfg) if (mode == 'train' and self.augment_cfg.get('enabled', False)) else None

        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self._load_subject(path) for path in self.data_paths]
        self.trial_subject = int(self.loaded_data[0]['eeg'].shape[0])
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        self.feature_dir = resolve_feature_dir(config)
        self.ablation = resolve_ablation_config(config)
        soft_k = int(config['Blur']['soft_k'])
        soft_g = int(config['Blur']['soft_g'])
        hard_k = int(config['Blur']['hard_k'])
        image_paths = []
        for loaded in self.loaded_data:
            image_paths.extend([str(path) for path in np.asarray(loaded['img']).reshape(-1).tolist()])
        ensure_visual_feature_caches(config, mode, image_paths)
        candidates = feature_candidates(self.feature_dir, self.model_type, mode, soft_k, soft_g, hard_k)
        self.primary_kind = self.ablation['primary_feature']
        self.secondary_kind = self.ablation.get('secondary_feature') or self.primary_kind
        self.primary_store = pick_existing_feature_store(candidates[self.primary_kind])
        self.secondary_store = pick_existing_feature_store(candidates[self.secondary_kind])
        self.aux_teacher_cfg = dict(config.get('aux_teacher', {}) or {})
        self.aux_teacher_enabled = bool(self.aux_teacher_cfg.get('enabled', False))
        self.aux_teacher_store = None
        self.aux_teacher_dim = 0
        if self.aux_teacher_enabled:
            aux_dir = self.aux_teacher_cfg.get('feature_dir')
            aux_model = self.aux_teacher_cfg.get('model_type')
            aux_kind = str(self.aux_teacher_cfg.get('feature_kind', 'clean'))
            if not aux_dir or not aux_model:
                raise ValueError('aux_teacher.enabled requires aux_teacher.feature_dir and aux_teacher.model_type.')
            aux_candidates = feature_candidates(aux_dir, aux_model, mode, soft_k, soft_g, hard_k)
            if aux_kind not in aux_candidates:
                raise ValueError(f'Unsupported aux_teacher.feature_kind={aux_kind}; expected one of {list(aux_candidates)}')
            self.aux_teacher_store = pick_existing_feature_store(aux_candidates[aux_kind])
            self.aux_teacher_dim = int(feature_store_dims(self.aux_teacher_store)[0])
        self.target_adapter_cfg = dict(config.get('target_adapter', {}) or {})
        self.target_adapter_enabled = bool(self.target_adapter_cfg.get('enabled', False))
        self.target_anchor_store = None
        self.target_anchor_stores = []
        self.target_anchor_dim = 0
        self.target_anchor_dims = []
        if self.target_adapter_enabled:
            anchor_specs = self.target_adapter_cfg.get('anchors', None)
            if anchor_specs is None:
                anchor_specs = [{
                    'anchor_feature_dir': self.target_adapter_cfg.get('anchor_feature_dir'),
                    'anchor_model_type': self.target_adapter_cfg.get('anchor_model_type'),
                    'anchor_feature_kind': self.target_adapter_cfg.get('anchor_feature_kind', 'semantic'),
                }]
            for anchor_spec in list(anchor_specs):
                anchor_dir = anchor_spec.get('anchor_feature_dir') or anchor_spec.get('feature_dir')
                anchor_model = anchor_spec.get('anchor_model_type') or anchor_spec.get('model_type')
                anchor_kind = str(anchor_spec.get('anchor_feature_kind', anchor_spec.get('feature_kind', 'semantic')))
                if not anchor_dir or not anchor_model:
                    raise ValueError('target_adapter.enabled requires anchor_feature_dir and anchor_model_type.')
                anchor_candidates = feature_candidates(anchor_dir, anchor_model, mode, soft_k, soft_g, hard_k)
                if anchor_kind not in anchor_candidates:
                    raise ValueError(f'Unsupported target_adapter.anchor_feature_kind={anchor_kind}; expected one of {list(anchor_candidates)}')
                anchor_store = pick_existing_feature_store(anchor_candidates[anchor_kind])
                self.target_anchor_stores.append(anchor_store)
                self.target_anchor_dims.append(int(feature_store_dims(anchor_store)[0]))
            if self.target_anchor_stores:
                self.target_anchor_store = self.target_anchor_stores[0]
                self.target_anchor_dim = self.target_anchor_dims[0]
        self.mctf_level_sources = resolve_mctf_level_sources(config, mode, candidates, image_paths=image_paths)
        self.mctf_level_names = [src['name'] for src in self.mctf_level_sources]
        self.mctf_level_dims = []
        for src in self.mctf_level_sources:
            dims = feature_store_dims(src['store'])
            if 'level_index' in src:
                self.mctf_level_dims.append(int(dims[src['level_index']]))
            else:
                self.mctf_level_dims.append(int(dims[0]))
        # Feature stores are immutable path dictionaries. Caching their tensor
        # references removes repeated Python dictionary lookups across epochs.
        self._features_by_path = {}

    def _load_subject(self, data_path: str) -> Dict:
        logging.info('----load %s----', data_path)
        loaded = torch.load(data_path, weights_only=False)
        loaded['eeg'] = torch.from_numpy(loaded['eeg'])

        if self.selected_ch is not None:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded['eeg'] = loaded['eeg'][:, :, selected_idx]

        loaded['eeg'] = self._maybe_zscore(loaded['eeg'].float(), data_path)
        eeg_trials = loaded['eeg']
        img_array = _canonical_img_array(loaded['img'], avg=self.avg)

        if self.avg:
            out = {
                'eeg': eeg_trials.mean(axis=1),
                'img': img_array,
            }
        elif self.partial_avg_active:
            n_stimuli, n_trials = int(eeg_trials.shape[0]), int(eeg_trials.shape[1])
            k = max(1, min(self.partial_avg_k, n_trials))
            repeats = max(1, self.partial_avg_repeats)
            seed_base = int(self.config.get('seed', 42))

            if img_array.ndim == 1:
                img_group = img_array.reshape(n_stimuli, -1)
            else:
                img_group = img_array

            partial_eeg = []
            partial_eeg_avg = []
            partial_img = []
            for stim_idx in range(n_stimuli):
                stim_trials = eeg_trials[stim_idx]
                stim_avg = stim_trials.mean(axis=0)
                stim_img = str(np.asarray(img_group[stim_idx]).reshape(-1)[0])
                for rep_idx in range(repeats):
                    rng = np.random.default_rng(seed_base + stim_idx * 1009 + rep_idx * 9176)
                    select_idx = rng.choice(n_trials, size=k, replace=False)
                    eeg_partial = stim_trials[select_idx].mean(axis=0)
                    partial_eeg.append(eeg_partial)
                    partial_eeg_avg.append(stim_avg)
                    partial_img.append(stim_img)

            out = {
                'eeg': torch.stack(partial_eeg, dim=0),
                'eeg_avg': torch.stack(partial_eeg_avg, dim=0),
                'img': np.asarray(partial_img),
            }
        else:
            out = {
                'eeg': eeg_trials.reshape(-1, *eeg_trials.shape[2:]),
                'eeg_avg': eeg_trials.mean(axis=1),
                'img': img_array,
            }

        for key, value in out.items():
            if key in {'eeg', 'img'}:
                if hasattr(value, 'shape'):
                    logging.info('%s: %s', key, value.shape)
                else:
                    logging.info('%s: type=%s', key, type(value))
        return out

    def _maybe_zscore(self, eeg: torch.Tensor, data_path: str) -> torch.Tensor:
        if not bool(self.config['data'].get('zscore', False)):
            return eeg

        stat_source = eeg
        if self.mode != 'train' and bool(self.config['data'].get('zscore_train_stats', True)):
            train_path = os.path.join(os.path.dirname(data_path), 'train.pt')
            train_payload = torch.load(train_path, weights_only=False)
            stat_source = torch.from_numpy(train_payload['eeg']).float()
            if self.selected_ch is not None:
                selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
                stat_source = stat_source[:, :, selected_idx]

        channel_axis = stat_source.ndim - 2
        reduce_axes = tuple(idx for idx in range(stat_source.ndim) if idx != channel_axis)
        mean = stat_source.mean(dim=reduce_axes, keepdim=True)
        std = stat_source.std(dim=reduce_axes, keepdim=True).clamp_min(1e-6)
        return (eeg - mean) / std

    def _get_visual_features(self, img_path: str, trial_index: int):
        cached = self._features_by_path.get(img_path)
        if cached is not None:
            return cached

        primary = lookup_feature(self.primary_store, img_path, trial_index)
        secondary = lookup_feature(self.secondary_store, img_path, trial_index)
        levels = []
        for source in self.mctf_level_sources:
            feature = lookup_feature(source['store'], img_path, trial_index)
            if isinstance(feature, list):
                feature = feature[source.get('level_index', 0)]
            levels.append(feature)

        if self.target_adapter_enabled:
            anchors = [lookup_feature(store, img_path, trial_index) for store in self.target_anchor_stores]
        else:
            anchors = None
        cached = (primary, secondary, levels, anchors)
        self._features_by_path[img_path] = cached
        return cached

    def __getitem__(self, index: int) -> Dict:
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.augment is not None:
            eeg = self.augment(eeg)


        if self.avg:
            eeg_mean = eeg
        elif self.partial_avg_active:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index].float()
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        img_entries = self.loaded_data[subject]['img']
        img_count = int(len(img_entries))
        if img_count == self.trial_subject:
            img_lookup_index = trial_index
        else:
            avg_count = int(self.loaded_data[subject]['eeg_avg'].shape[0]) if 'eeg_avg' in self.loaded_data[subject] else 0
            if img_count == avg_count:
                img_lookup_index = trial_index // self.per_trials
            elif img_count == 1:
                img_lookup_index = 0
            else:
                raise IndexError(
                    f'Unable to map MEG image index: trial_index={trial_index}, img_count={img_count}, '
                    f'trial_subject={self.trial_subject}, avg_count={avg_count}, per_trials={self.per_trials}, '
                    f'mode={self.mode}, subject={self.subjects[subject]}'
                )
        img_path = str(img_entries[img_lookup_index])


        primary_feature, secondary_feature, level_features, anchor_features = self._get_visual_features(img_path, trial_index)

        subject_name = self.subjects[subject]
        try:
            subject_global = int(str(subject_name).split('-')[-1]) - 1
        except Exception:
            subject_global = int(subject)

        sample = {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'img_path': img_path,
            'visual_feature_primary': primary_feature,
            'visual_feature_secondary': secondary_feature,
            'low_blur_feature': primary_feature,
            'hard_blur_feature': secondary_feature,
            'subject': subject_global,
            'subject_local': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
            'visual_feature_levels': level_features,
        }
        if self.aux_teacher_enabled:
            sample['public_teacher_feature'] = lookup_feature(self.aux_teacher_store, img_path, trial_index)
        if self.target_adapter_enabled:
            if len(self.target_anchor_stores) > 1:
                sample['target_anchor_features'] = anchor_features
            else:
                sample['target_anchor_feature'] = anchor_features[0]
        return sample

    def __len__(self) -> int:
        return self.trial_all_subjects


def load_data(config: Dict):
    exp_setting = config.get('exp_setting', 'intra-subject')
    workers = train_eval_workers(config)

    if exp_setting == 'intra-subject':
        full_train_dataset = MEGDataset(config, mode='train')
        test_dataset = MEGDataset(config, mode='test')

        use_test_as_val = bool(config['data'].get('use_test_as_val', False))
        generator = torch.Generator().manual_seed(int(config.get('seed', 42)))
        if use_test_as_val:
            train_dataset = full_train_dataset
            val_dataset = test_dataset
        else:
            val_ratio = float(config['data'].get('val_ratio', 0.1))
            total_len = len(full_train_dataset)
            val_len = max(1, int(total_len * val_ratio))
            train_len = total_len - val_len
            train_dataset, val_dataset = random_split(
                full_train_dataset,
                [train_len, val_len],
                generator=generator
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config['data']['train_batch_size']),
            shuffle=True,
            drop_last=True,
            num_workers=workers['train'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=_resolve_eval_batch_size(config['data']['val_batch_size'], len(val_dataset)),
            shuffle=False,
            drop_last=False,
            num_workers=workers['eval'],
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=_resolve_eval_batch_size(config['data']['test_batch_size'], len(test_dataset)),
            shuffle=False,
            drop_last=False,
            num_workers=workers['eval'],
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    if exp_setting == 'inter-subject':
        test_subjects = list(config['data']['subjects'])
        all_subjects = [f'sub-{i:02d}' for i in range(1, 5)]

        dg_cfg = config.get('dg', {})
        protocol_cfg = dg_cfg.get('protocol', {}) if dg_cfg is not None else {}
        protocol_enabled = bool(protocol_cfg.get('enabled', True))

        test_dataset = MEGDataset(config, mode='test')

        if protocol_enabled:
            val_subjects_cfg = protocol_cfg.get('val_subjects', 'auto')
            source_test_val = bool(protocol_cfg.get('source_test_val', False))
            if source_test_val:
                train_subjects = [s for s in all_subjects if s not in test_subjects]
                val_subjects = list(train_subjects)
            else:
                train_subjects, val_subjects = resolve_inter_subject_split(
                    all_subjects=all_subjects,
                    test_subjects=test_subjects,
                    val_subjects=val_subjects_cfg,
                )
            train_config = deep_copy_config(config)
            train_config['data']['subjects'] = train_subjects
            val_config = deep_copy_config(config)
            val_config['data']['subjects'] = val_subjects
            train_dataset = MEGDataset(train_config, mode='train')
            val_dataset = MEGDataset(val_config, mode='test')
        else:
            train_subjects = list(sorted(set(all_subjects) - set(test_subjects)))
            train_config = deep_copy_config(config)
            train_config['data']['subjects'] = train_subjects
            full_train_dataset = MEGDataset(train_config, mode='train')
            val_ratio = float(config['data'].get('val_ratio', 0.05))
            total_len = len(full_train_dataset)
            val_len = max(1, int(total_len * val_ratio))
            train_len = max(1, total_len - val_len)
            generator = torch.Generator().manual_seed(int(config.get('seed', 42)))
            train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len], generator=generator)

        test_loader = DataLoader(
            test_dataset,
            batch_size=_resolve_eval_batch_size(config['data']['test_batch_size'], len(test_dataset)),
            shuffle=False,
            drop_last=False,
            num_workers=workers['eval'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=_resolve_eval_batch_size(config['data']['val_batch_size'], len(val_dataset)),
            shuffle=False,
            drop_last=False,
            num_workers=workers['eval'],
            pin_memory=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config['data']['train_batch_size']),
            shuffle=True,
            drop_last=True,
            num_workers=workers['train'],
            pin_memory=True
        )
        return train_loader, val_loader, test_loader

    raise ValueError(f'Unsupported exp_setting: {exp_setting}')
