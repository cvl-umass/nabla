# NABLA: Identity-Preserving Generation for Birds (CVPR26 Highlight)

This repository contains the official code for **"Not All Birds Look The Same: Identity-Preserving Generation For Birds"**. 

NABLA is a unified framework combining the capabilities of bounding-box conditional generation (Insert Anything) and global spatial conditional generation (OminiControl) to evaluate and train identity-preserving diffusion models.

## Repository Structure

The codebase is organized into a shared pipeline that dynamically routes to the appropriate model architecture based on your configuration.

- `src/shared/`: Contains unified training loops, evaluation scripts, dataset loaders, and metric calculators.
- `src/insert/`: Model definitions and data wrappers specific to the [Insert Anything](https://github.com/song-wensong/insert-anything) architecture (expansion, probabilistic masking).
- `src/omini/`: Model definitions and data wrappers specific to the [OminiControl](https://github.com/Yuanshi9815/OminiControl) architecture (tight crops, exact masking).
- `configs/`: YAML configuration files defining hyperparameters, conditioning modes, and model types.
- `datasets/`: Curated evaluation datasets (`nabla_pairs`, `inat_seen`, `inat_unseen`).
- `scripts/`: Example SLURM submission scripts for training and evaluation.

## Installation

Ensure you have a working PyTorch environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Models

Checkpoints can be downloaded at the following links: [Flux](https://github.com/black-forest-labs/flux), [Insert Anything](https://github.com/song-wensong/insert-anything), [OminiControl](https://github.com/Yuanshi9815/OminiControl).

## Training

Training is centralized through `src/shared/main.py`. The script relies on YAML configuration files to determine whether to train an `insert` or `omini` model, and what conditioning modes to use.

To launch a training job:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python src/shared/main.py \
    --job_dir out/my_experiment \
    --config configs/insert_augment.yaml
```

## Evaluation

Evaluation is handled by `src/shared/generate_eval.py`. This script is designed for multi-GPU inference using `torch.distributed.run` and will automatically partition the dataset across available GPUs. It handles dynamic image preprocessing (padding, masking, unpadding) depending on the `model_type` in the config.

### Running Generation and Metrics

To evaluate a model on the NABirds test split:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
NUM_GPUS=2

python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} \
    src/shared/generate_eval.py \
    --job_dir out/my_experiment \
    --dataset nabirds \
    --split test \
    --guidance_scale 2.5
```

**Key Flags:**
- `--dataset`: Choose between `nabirds`, `inat`, or `video`.
- `--split`: Use `test` for NABirds, or `na` / `unseen` for iNaturalist.
- `--baseline`: Skips loading LoRA checkpoints to evaluate the base model.
- `--skip_existing`: Skips generating images that already exist in the output directory.
- `--recompute_metrics`: Loads existing images to recalculate inline metrics (LPIPS, L2, SigLIP, DINO) without regenerating the images.

*Outputs (generated images, stitched comparison grids, and `.csv` metric logs) will be saved in `out/my_experiment/eval_<dataset>_<split>_<ckpt>`.*

## Distribution Metrics (FID & CMMD)

While `generate_eval.py` calculates pairwise metrics (like LPIPS and DINO similarity), dataset-wide distribution metrics are calculated using `src/shared/metrics/calc_metrics.py`.

```bash
python src/shared/metrics/calc_metrics.py \
    --gt path/to/ground_truth_images \
    --pred path/to/generated_images \
    --metrics fid,cmmd \
    --batch_size 32
```

*Note: You must have the required metric repositories (e.g., `pytorch-fid`) installed and configured as defined in the script.*

## Configuration Guide

The YAML files in `configs/` control the model behavior. Key parameters include:

- `model_type`: Set to `"insert"` or `"omini"`.
- `train.conditioning_mode`: Defines the spatial control (e.g., `"fill"`, `"depth"`, `"keypoints"`).
- `train.dataset.augment`: Toggles training augmentations.

For a full list of hyperparameters, please refer to the provided baseline YAMLs.