# 16-825 Assignment 2: Single View to 3D

**Name:** Yi-Ning Huang
**Andrew ID:** yiningh3

Report webpage: 
http://www.andrew.cmu.edu/course/16-825/projects/yiningh3/proj2/ or 
https://yining213.github.io/16825-assignment2/

The original assignment instructions are in [`handout.md`](handout.md).

Completed: Q1.1–1.3, Q2.1–2.5, Q3.1 (implicit decoder) and Q3.2 (parametric decoder). Q3.3 was not submitted.

---

## Setup

Use the same environment as Assignment 1 (PyTorch + PyTorch3D), then install the remaining packages:

```bash
conda create -n learning3d python=3.10
conda activate learning3d
pip install torch torchvision                # or the CUDA build for your machine
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

The results were produced with Python 3.10, PyTorch 2.14 (CUDA 13.0), torchvision 0.29 and PyTorch3D 0.7.8.

### Dataset

Download the single-class (chair) R2N2 dataset into this folder (git-lfs is not needed):

```bash
wget -c https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset/resolve/main/r2n2_shapenet_dataset.zip
unzip r2n2_shapenet_dataset.zip              # creates r2n2_shapenet_dataset/ (7.7 GB)
```

`dataset_location.py` looks for `r2n2_shapenet_dataset/` next to the code. To keep the dataset elsewhere, point
`L3D_DATA_ROOT` at the folder that contains it:

```bash
export L3D_DATA_ROOT=/path/to/folder/containing/r2n2_shapenet_dataset
```

---

## Running the code

`main.py` runs every question and collects all outputs in `results/`.

```bash
python main.py                  # run everything
python main.py --list           # list the available questions
python main.py -q 1 2.4         # run only the listed questions ("1" expands to 1.1, 1.2, 1.3)
python main.py --dry_run        # print the commands main.py would run, without running them
python main.py --retrain        # retrain models even if their checkpoint exists
python main.py --device cpu     # force CPU (training on CPU is very slow)
python main.py --num_workers 16 # more data-loader workers for training (default 8)
```

Model checkpoints are not included (they are 140–350 MB each). Any question that needs a model trains it first and
saves `checkpoint_<type><exp>.pth` in this folder; later runs reuse it. Training took 40 min to 2.5 h per model on one
GPU (voxel ~40 min, point and mesh ~1 h each, implicit ~2 h, parametric ~2.3 h). Data loading is CPU-bound, so with
many workers set `OMP_NUM_THREADS=2` to avoid oversubscribing the CPU.

---

## Outputs

| Question | Output in `results/` |
|---|---|
| 1.1 Fit a voxel grid | `q1/q1_1_vox.gif` (optimized left, ground truth right) |
| 1.2 Fit a point cloud | `q1/q1_2_point.gif` |
| 1.3 Fit a mesh | `q1/q1_3_mesh.gif` |
| 2.1 Image to voxel grid | `q2_1/eval_vox.png` (F1 plot), `q2_1/examples/` (report examples), `q2_1/eval/` (every 200th test chair) |
| 2.2 Image to point cloud | `q2_2/eval_point.png`, `q2_2/examples/`, `q2_2/eval/` |
| 2.3 Image to mesh | `q2_3/eval_mesh.png`, `q2_3/examples/`, `q2_3/eval/` |
| 2.4 Occupancy threshold | `q2_4/eval_vox_t<0.1–0.4>_seed0.png`, `q2_4/eval_vox_seed0.png` (0.5), `q2_4/q2_4_vox_thresh.png` |
| 2.5 Interpretation | `q2_5/q2_5_view_f1.png`, `q2_5/q2_5_vox_orientation.png`, `q2_5/q2_5_interpolation.png` |
| 3.1 Implicit decoder | `q3_1/eval_vox_implicit_seed0.png`, `q3_1/eval_vox_implicit_t0.3_seed0.png`, `q3_1/examples/` |
| 3.2 Parametric decoder | `q3_2/eval_point_param_seed0.png`, `q3_2/eval_point_seed0.png` (2.2 decoder, same views), `q3_2/examples/`, `q3_2/q3_2_param_patches.png` |

Each example folder holds, per test chair, a still (input image | prediction | ground-truth mesh, rendered from the
input camera) and a 360° GIF (prediction | ground truth). The average F1@0.05 is printed at the end of each
evaluation (`Overall: … Avg F1@0.05`).

---

## Settings used for the report

- **Q1:** Adam, lr 1e-2, 3,000 iterations (`fit_data.py --lr 1e-2 --max_iter 3000`).
- **Q2 / Q3 training:** ImageNet-pretrained ResNet18 encoder fine-tuned with the decoder; Adam (lr 4e-4, batch 32),
  10,000 iterations, on a GPU. `--load_feat` was not used.
- **Examples:** test chairs 240, 660 and 150, each from its most front-facing input view (`vis_examples.py --front`).
- **Same input views:** R2N2 picks a random input view each time a chair is loaded, so the average F1 can differ by up to
  about one point between evaluation runs. Comparisons across models or thresholds (Q2.4, Q2.5, Q3) use `--seed 0`, so every run sees the same
  views. This is why the 2.1 / 2.2 models score 53.0 / 79.3 in those comparisons and 53.4 / 80.0 in the main evaluation.
- **Voxel threshold:** voxel predictions are probabilities in eval mode; `--vox_thresh` (default 0.5) sets the
  occupancy threshold used for marching cubes and the renders.

---

## Source files

| File | Contents |
|---|---|
| `main.py` | Entry point. Runs any or all questions with the report's settings. |
| `losses.py` | Q1 losses: `voxel_loss` (BCE with logits), `chamfer_loss` (from scratch), `smoothness_loss` (Laplacian). |
| `fit_data.py` | Q1 fitting loop (course starter) plus 360° visualization of the result. |
| `model.py` | `SingleViewto3D` and the decoders: `VoxelDecoder` (2.1), `PointDecoder` (2.2), `MeshDecoder` (2.3), `ImplicitDecoder` (3.1), `ParametricDecoder` (3.2). |
| `train_model.py` | Training loop (course starter). Added `--exp`, `--vox_dec`, `--point_dec`, `--n_patches`; resumed checkpoints load onto `--device`. |
| `eval_model.py` | Evaluation and F1 plot (course starter). Added visualization, `--seed`, `--vox_thresh`, per-class F1, a guard for empty voxel predictions, and checkpoint loading onto `--device` (so GPU-trained checkpoints evaluate on CPU). |
| `vis_examples.py` | Renders chosen test chairs (`--indices`, `--front`) without running the full evaluation. |
| `interpret.py` | Report figures for Q2.4, Q2.5 and Q3.2 (`--only thresholds / view / orientation / interp / patches`). |
| `visualize.py` | Rendering helpers: voxel / point / mesh 360° renders, two-sided lighting, GIF and comparison images. |
| `dataset_location.py` | Dataset paths (defaults to this folder, or `L3D_DATA_ROOT`). |
| `r2n2_custom.py`, `utils_vox.py` | Course-provided dataset loader and voxel utilities (unchanged). |

Each script can also be run on its own, for example:

```bash
python train_model.py --type point --max_iter 10001 --save_freq 1000
python eval_model.py --type point --load_checkpoint
python vis_examples.py --type mesh --front --indices 240 660 150
python interpret.py --only view --out_dir results/q2_5
```
