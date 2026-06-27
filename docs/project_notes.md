# Project Notes

## Method Summary

Hi-DREAM is based on the idea that visual cortical hierarchy can be aligned with the depth hierarchy of a diffusion-based image generator.

The project organizes fMRI features into three broad streams:

- **Early visual stream**: mainly supports low-level spatial structure and edge-like information.
- **Middle visual stream**: contributes intermediate shape and part-level cues.
- **Late visual stream**: contributes higher-level semantic information.

These streams are then injected into different depths of the generative model rather than being treated as a single flat condition vector.

## Planned Release

The repository is currently structured for the public release. Planned components include:

1. fMRI feature loading and ROI grouping utilities.
2. ROI adapter and depth-aligned conditioning modules.
3. Training scripts for diffusion-based reconstruction.
4. Inference scripts for image reconstruction.
5. Evaluation scripts for low-level and high-level reconstruction metrics.

## Notes for Users

Dataset access and preprocessing should follow the original dataset guidelines. Large datasets, pretrained checkpoints, generated images, and experiment outputs should not be committed directly to this repository.
