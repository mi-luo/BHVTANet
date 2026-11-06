# BHVTANet

Code for **Brain-Compatible Hierarchical Visual Target Adaptation Network for Zero-Shot Brain-to-Image Retrieval**. BHVTANet constructs hierarchical visual targets for EEG/MEG image retrieval through MCTF, MDTP, DACAM, and DSWR.

## Install

The reference environment is Linux/WSL, Python 3.12, PyTorch 2.6.0, and CUDA 12.4.

```bash
cd BHVTANet
conda create -n bhvtanet python=3.12 -y
conda activate bhvtanet
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Datasets

- **THINGS-EEG:** [Official dataset](https://osf.io/3jk45/) · [Prepared data from UBP](https://huggingface.co/datasets/Haitao999/things-eeg).
- **THINGS-MEG:** [Official dataset](https://openneuro.org/datasets/ds004212) · [Prepared data from UBP](https://huggingface.co/datasets/Haitao999/things-meg).

## Preprocessing

EEG preprocessing follows the common pipeline used by NICE, UBP, and HyFI; MEG data preparation follows UBP. Please refer to their papers and public implementations:

- **NICE:** [Decoding Natural Images from EEG for Object Recognition](https://openreview.net/forum?id=dhLIno8FmH), ICLR 2024. [GitHub](https://github.com/eeyhsong/NICE-EEG).
- **UBP:** [Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://doi.org/10.1109/CVPR52734.2025.00215
        
        ), CVPR 2025. [GitHub](https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior).
- **HyFI:** [HyFI: Hyperbolic Feature Interpolation for Brain-Vision Alignment](https://doi.org/10.1609/aaai.v40i7.37476
        
        ), AAAI 2026. [GitHub](https://github.com/SangMin316/HyFI).

## Train

Set the local data and feature paths as described in [data/README.md](data/README.md), then run from the project root:

```bash
# EEG: intra-subject
python -s main.py --config configs/eeg_intra.yaml --subjects all --seed 200 --gpu 0

# EEG: inter-subject
python -s main.py --config configs/eeg_inter.yaml --subjects all --seed 200 --gpu 0

# MEG: intra-subject
python -s main.py --config configs/meg_intra.yaml --subjects all --seed 200 --gpu 0

# MEG: inter-subject
python -s main.py --config configs/meg_inter.yaml --subjects all --seed 200 --gpu 0
```

## Test

Evaluate a saved checkpoint with its matching configuration and training settings:

```bash
python -s main.py --config configs/eeg_intra.yaml \
  --subjects sub-01 --seed 200 --gpu 0 \
  --checkpoint "/absolute/path/to/model.ckpt" \
  --output-root checkpoint_evaluation
```

Trained checkpoints are not included. Summarize mean Top-1/Top-5 accuracy from training outputs with:

```bash
python -s scripts/summarize_results.py --results results --seed 200
```

## Acknowledgements

We thank the authors of THINGS-EEG and THINGS-data for making their datasets publicly available, and the authors of NICE, UBP, and HyFI for sharing their preprocessing resources and implementations. We also acknowledge the baseline networks used in our study:

- **THINGS-EEG:** Gifford et al., [A large and rich EEG dataset for modeling human visual object recognition](https://doi.org/10.1016/j.neuroimage.2022.119754
        
        
        
        ), NeuroImage, 2022.
- **THINGS-data / THINGS-MEG:** Hebart et al., [THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior](https://doi.org/10.7554/eLife.82580
        
        
        
        ), eLife, 2023.
- **EEGProject / TSConv:** Song et al., [Decoding Natural Images from EEG for Object Recognition](https://openreview.net/forum?id=dhLIno8FmH), ICLR, 2024.
- **ShallowNet / DeepNet:** Schirrmeister et al., [Deep learning with convolutional neural networks for EEG decoding and visualization](https://doi.org/10.1002/hbm.23730
        
        
        
        ), Human Brain Mapping, 2017.
- **EEGNet:** Lawhern et al., [EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces](https://doi.org/10.1088/1741-2552/aace8c
        
        
        
        ), Journal of Neural Engineering, 2018.

## Contact

For bug reports, feature requests, or questions about BHVTANet, please open an issue in this repository. Discussions and suggestions are welcome!
