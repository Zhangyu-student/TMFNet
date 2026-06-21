# TMFNet

Official implementation of **TMFNet (TemporalMambaFusionNet)** for three-temporal optical remote sensing cloud removal.

TMFNet performs pixel-wise temporal fusion over multi-temporal observations and reconstructs a cloud-free target image with a lightweight Mamba-style selective state update design.

## Paper

This work has been published in **IEEE Geoscience and Remote Sensing Letters (GRSL)**.

**Title:** TemporalMambaFusionNet: Cloud-Aware Selective Temporal Fusion for Multi-Temporal Remote Sensing Cloud Removal

**Authors:** Yu Zhang, Hairong Tang, Lijia Huang, and Peng Zhang

**Journal:** IEEE Geoscience and Remote Sensing Letters, Volume 23

**Article Sequence Number:** 6009105

**Date of Publication:** 28 April 2026

**DOI:** [10.1109/LGRS.2026.3687583](https://doi.org/10.1109/LGRS.2026.3687583)

## Highlights

- Three-temporal cloud removal framework for optical remote sensing imagery
- Pixel-wise temporal fusion with a lightweight selective state update module
- Support for training, inference, quantitative evaluation, and visualization
- Includes the core code for model training, inference, and evaluation

## Repository Structure

```text
.
|-- models/                  # TMFNet and ablation model variants
|-- pretrained/              # pretrained checkpoints
|-- main_tmp.py              # training entry
|-- test_png.py              # inference entry
|-- eval.py                  # folder-based evaluation script
|-- dataset.py               # dataset definitions
|-- loss.py                  # training losses
|-- metrics.py               # image quality metrics
|-- training_utils.py        # optimizer / scheduler / logging helpers
`-- visualize.py, visualize2.py
```

## Requirements

- Python 3.8+
- PyTorch
- numpy
- pillow
- tqdm
- scikit-image
- opencv-python
- tifffile

For evaluation:

- lpips
- cleanfid

Example:

```bash
pip install numpy pillow tqdm scikit-image opencv-python tifffile
pip install lpips cleanfid
```

Install PyTorch according to your CUDA environment from the official guide:

https://pytorch.org/get-started/locally/

## Data Preparation

The current codebase contains two dataset modes:

- `new_multi`: split-file based dataset using `train.txt`, `val.txt`, and `test.txt`
- `old_multi`: legacy dataset layout under `multipleImage/`

### `new_multi` expected layout

```text
your_data_root/
|-- train.txt
|-- val.txt
|-- test.txt
`-- Sen2_MTC/
    |-- tile_xxx/
    |   |-- cloud/
    |   |   |-- image_0.tif
    |   |   |-- image_1.tif
    |   |   `-- image_2.tif
    |   `-- cloudless/
    |       `-- image.tif
    `-- ...
```

Each sample uses three cloudy observations and one cloud-free target image.

## Training

Training is configured through the `config` dictionary inside `main_tmp.py`.

Before running training, update at least:

- `data_root`
- `dataset_type`
- `batch_size`
- `num_epochs`
- `save_dir`
- `vis_dir`
- `log_file`
- `pretrained_path`

Run:

```bash
python main_tmp.py
```

The training script writes outputs such as:

- checkpoints to `save_dir`
- visualizations to `vis_dir`
- CSV logs to `log_file`
- TensorBoard logs to `log_dir`

## Inference

Inference is provided in `test_png.py`.

Before running, update the `config` dictionary at the bottom of the file:

- `model_path`
- `data_root`
- `save_dir`
- `batch_size`
- `device`
- `visualize_attention`

Run:

```bash
python test_png.py
```

The script saves:

- restored PNG images
- ground-truth PNG images
- per-image metric records
- FID folders for generated and real samples

## Evaluation

Folder-based quantitative evaluation is provided in `eval.py`.

Before running, update:

- `gt_dir`
- `pred_dir`
- `output_dir`
- `lpips_net`
- `workers`
- `use_cpu`

Run:

```bash
python eval.py
```

Reported metrics:

- PSNR
- SSIM
- SAM
- LPIPS
- FID

Evaluation outputs are written to `output_dir`, including:

- `detailed_metrics.csv`
- `summary_metrics.txt`
- `config.json`

## Pretrained Weights

Place released checkpoints in `pretrained/`.

The repository currently includes a local checkpoint path reference:

- `pretrained/TMFNet_new.pth`

## Citation

If you use this repository, please cite:

```bibtex
@article{tmfnet_grsl_2026,
  title   = {TemporalMambaFusionNet: Cloud-Aware Selective Temporal Fusion for Multi-Temporal Remote Sensing Cloud Removal},
  author  = {Zhang, Yu and Tang, Hairong and Huang, Lijia and Zhang, Peng},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  volume  = {23},
  pages   = {6009105},
  doi     = {10.1109/LGRS.2026.3687583}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
