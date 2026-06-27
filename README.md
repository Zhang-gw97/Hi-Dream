# Hi-DREAM: Brain-Inspired Hierarchical Diffusion for fMRI-to-Image Reconstruction via ROI Encoder and VisuAl Mapping

**Hi-DREAM** is a research project for hierarchical fMRI-to-image reconstruction. The project explores how signals from different visual cortical regions can be mapped to different levels of a generative image model, enabling more structured and semantically meaningful image reconstruction from human brain activity.

> This repository is currently being prepared for public release. Full training and inference code will be added progressively.

## Overview

Hi-DREAM aims to reconstruct visual images from fMRI responses by combining:

- **Hierarchical visual ROI conditioning**: early, middle, and high-level visual areas are used as separate condition streams.
- **Depth-aligned generative injection**: different ROI groups are injected into different depths of a diffusion-based image generator.
- **Diffusion-based reconstruction**: a latent diffusion backbone is used to generate natural images conditioned on brain activity.

The main idea is that the human visual hierarchy and the generative model hierarchy can be aligned: lower-level visual areas provide structural and spatial cues, while higher-level regions contribute semantic information.

## Project Structure

```text
Hi-Dream/
├── configs/                 # Example configuration files
├── docs/                    # Project notes and documentation
├── scripts/                 # Training and inference entry points
├── src/hi_dream/            # Core package code
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Zhang-gw97/Hi-Dream.git
cd Hi-Dream
```

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate hi-dream
```

Alternatively, install the Python dependencies with pip:

```bash
pip install -r requirements.txt
```

## Example Usage

The current scripts are lightweight placeholders for the final release. After the full implementation is added, the expected workflow will be:

```bash
python scripts/train.py --config configs/hidream_example.yaml
python scripts/inference.py --config configs/hidream_example.yaml --checkpoint path/to/checkpoint.pt
```

## Data

This project is designed for fMRI-based visual reconstruction experiments. Dataset preparation scripts and instructions will be added in a later release.

For public datasets such as NSD, users should follow the original dataset access requirements and licensing terms.

## Citation

If you find this project useful, please consider citing the corresponding paper once it is available.

```bibtex
@inproceedings{zhang2026hidream,
  title     = {Hi-DREAM: Hierarchical fMRI-to-Image Reconstruction with Depth-Aligned Diffusion Conditioning},
  author    = {Zhang, Guowei and others},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

## License

The license will be added before the full code release.

## Contact

For questions, please open an issue or contact the project maintainer.
