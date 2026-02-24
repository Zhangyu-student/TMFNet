# TMFNet (TemporalMambaFusionNet)

TMFNet is a cloud-quality-guided framework for **three-temporal** optical remote sensing cloud removal.  
It performs **pixel-wise selective temporal fusion** via a lightweight Mamba-inspired selective scan with an intuitive *write/forget* behavior, and provides interpretable suppression/confidence maps.

Repo:
https://github.com/Zhangyu-student/TMFNet

---

## Requirements

- Python 3.8+
- PyTorch
- numpy, pillow, tqdm, scikit-image, opencv-python

**For the evaluation script (LPIPS/FID):**
- lpips
- cleanfid

Example install:
pip install numpy pillow tqdm scikit-image opencv-python
pip install lpips cleanfid

Install PyTorch following your CUDA / CPU setup:
https://pytorch.org/get-started/locally/

---

## Training

This repo currently uses a **hard-coded `config` dict** in the training entry script.

1) Open the training script (e.g., `main_tmp.py`) and edit:
- `data_root` (your dataset path)
- `dataset_type` (e.g., `s2asiawest` / your dataset id)
- `batch_size`, `num_epochs`, etc.
- `pretrained_path` (optional)

2) Run:
python main_tmp.py

Outputs (default, configurable in `config`):
- checkpoints: ./checkpoints
- tensorboard logs: ./runs
- visualizations: ./visualizations
- csv log: ./train_logs_landsat.csv

---

## Evaluation (folder-to-folder metrics)

The evaluation script computes metrics between two folders of paired images:
- `gt_dir`: ground-truth (real) images
- `pred_dir`: predicted images

Metrics: **PSNR / SSIM / SAM / LPIPS / FID**  
It will write results to `output_dir` (default ./evaluation_results) including:
- detailed_metrics.csv
- summary_metrics.txt
- config.json

1) Open the evaluation script (e.g., `evaluate.py`) and edit the config at the bottom:
- `gt_dir`
- `pred_dir`
- `output_dir`
- `lpips_net` (alex/vgg/squeeze)
- `workers`
- `use_cpu`

2) Run:
python evaluate.py

---

## Pretrained Weights

If you want to share small pretrained weights, put them under:
pretrained/

Then set `pretrained_path` in the training `config` to the file you want to load.

Tip: avoid committing large checkpoints into git history. For large files, consider Git LFS or external hosting.

---

## Notes

Recommended .gitignore entries for runtime artifacts:
checkpoints/
logs/
*_results/
inference_results*/
.idea/

Paper link: TBD (not submitted yet).

---

## License

TBD
