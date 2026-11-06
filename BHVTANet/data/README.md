# Data and visual feature caches

The data loaders are `src/bhvtanet/data/eeg.py` and `src/bhvtanet/data/meg.py`. Obtain THINGS-EEG and THINGS-MEG/THINGS-data through the official releases cited in the paper. Record the dataset release and preprocessing settings used for an experiment.

The loaders consume preprocessed subject files. Downloading the raw public datasets does not create these files: sampling, channel selection, whitening/normalization, stimulus splits and repetition grouping must match the supplied configuration. This repository does not provide a complete raw brain-signal preprocessing pipeline. Brain recordings, images, visual-model weights, feature caches and trained brain-network checkpoints are not included in the source release.

## Directory layout

Use `--data-root "DATA_ROOT"` or `BHVTANET_DATA_ROOT` for the directory containing:

```text
DATA_ROOT/
├── things-eeg/
│   ├── Preprocessed_data_250Hz_whiten/
│   │   └── sub-01/{train,test}.pt
│   ├── Image_set_Resize/
│   └── Image_feature/BHVTANet/
└── things-meg/
    ├── Preprocessed_data/
    │   └── sub-01/{train,test}.pt
    ├── Image_set_Resize/
    └── Image_feature/BHVTANet/
```

EEG requires `sub-01` through `sub-10`; MEG requires `sub-01` through `sub-04`. Inter-subject training needs all source subjects even when evaluating one held-out subject. The default data root is the project's `data/` directory. For an existing visual-cache directory with another name, pass `--feature-subdir "CACHE_DIRECTORY_NAME"` or set `BHVTANET_FEATURE_SUBDIR`.

## Subject files

Each `train.pt` and `test.pt` is a dictionary read with `torch.load(..., weights_only=False)`. Load files from a trusted preparation process.

| Field | Expected content |
| --- | --- |
| `eeg` | NumPy array ordered as stimulus/image, repetition, channel, time; this key is also used for MEG signals |
| `img` | Image identifiers aligned with the stimulus and repetition order |
| `label`, `text` | EEG stimulus metadata carrying the stimulus/repetition axes |
| `session` | EEG session metadata; with the supplied averaging configuration, one entry per averaged stimulus is required |
| `times` | EEG temporal metadata in the representation expected by the loader |

The EEG loader assumes four training repetitions and 80 test repetitions when expanding trials. Supplied configurations average repetitions. The MEG loader also supports partial averaging. Temporal ranges, channels and averaging options are specified in each YAML file; retain those settings for comparable experiments.

## RN50 features

Each dataset's `Image_feature/BHVTANet/` contains the following caches:

```text
RN50_semantic_fovea_k51_g3_train.pt
RN50_semantic_fovea_k51_g3_test.pt
RN50_perceptual_gaussian_k31_train.pt
RN50_perceptual_gaussian_k31_test.pt
multilevel/RN50_multilevel_all.pt
```

The multi-level cache contains the ordered `early`, `middle`, `late`, `final` levels. A `features_by_path` dictionary associates image identifiers with tensors, or a tensor list for multiple levels. Tensor-only formats rely on row ordering. Preserve the original image identifiers and feature alignment when copying resources. Accepted formats and identifier lookup are implemented in `src/bhvtanet/data/common.py`.

## EEG anchor features

Use `--anchor-root "ANCHOR_ROOT"` or `BHVTANET_ANCHOR_ROOT` for the directory containing:

```text
ANCHOR_ROOT/
├── eeg_siglipb16/
│   ├── SigLIP_B_16_256_clean_train.pt
│   └── SigLIP_B_16_256_clean_test.pt
└── eeg_siglip2b16/
    ├── SigLIP2_B_16_256_clean_train.pt
    └── SigLIP2_B_16_256_clean_test.pt
```

The default anchor root is `data/anchors/`. The selected model and feature kind are specified in `target_adapter` in the EEG configurations. These cache files are not trained BHVTANet checkpoints.

## Prepare and check resources

Check the input files before starting a full experiment:

```bash
python -s scripts/verify_setup.py --data-root "DATA_ROOT" --anchor-root "ANCHOR_ROOT"
```

`--files-only` skips dependency checks other than OmegaConf. Passing this check establishes file presence, not tensor correctness, data-split validity or reproduced accuracy.

For missing visual features, provide the preprocessed subject image identifiers and source image tree, then run:

```bash
python -s scripts/prepare_visual_features.py --config configs/eeg_intra.yaml --data-root "DATA_ROOT" --anchor-root "ANCHOR_ROOT"
```

Repeat for the other configurations as needed. The script prepares RN50 features, multi-level features and enabled EEG anchors. It requires the corresponding visual-model weights, which may be downloaded if absent. `BHVTANET_WEIGHTS_ROOT` selects the local weight directory. Preserve the model version and extraction settings with regenerated caches. Source images are needed for extraction; complete existing caches can support training without extraction.

The configured multi-level extraction scans the image tree and filters common image extensions, so README and metadata files are skipped. Full resource regeneration from a newly downloaded release has not been validated.
