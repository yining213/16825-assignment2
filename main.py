"""
16-825 Assignment 2: Single View to 3D
Yi-Ning Huang (yiningh3)

Runs every question in the assignment and collects all results in results/.
Each question calls the per-task scripts (fit_data.py, train_model.py, eval_model.py, vis_examples.py,
interpret.py) with the settings used for the report.

Usage:
    python main.py                 # run everything (trains any model whose checkpoint is missing)
    python main.py -q 1 2.4        # run only the listed questions ("1" expands to 1.1, 1.2, 1.3)
    python main.py --list          # show available questions
    python main.py --dry_run       # print the commands without running them
    python main.py --retrain       # retrain models even if their checkpoint exists
    python main.py --device cpu    # force CPU (training on CPU is very slow)

Outputs (all under results/):
    1.1-1.3  q1/q1_1_vox.gif, q1/q1_2_point.gif, q1/q1_3_mesh.gif
    2.1-2.3  q2_N/eval_<type>.png (F1 plot), q2_N/eval/ (every 200th test chair), q2_N/examples/ (report examples)
    2.4      q2_4/eval_vox_t<thr>_seed0.png for thresholds 0.1-0.5, q2_4/q2_4_vox_thresh.png
    2.5      q2_5/q2_5_view_f1.png, q2_5/q2_5_vox_orientation.png, q2_5/q2_5_interpolation.png
    3.1      q3_1/eval_vox_implicit_seed0.png, q3_1/eval_vox_implicit_t0.3_seed0.png, q3_1/examples/
    3.2      q3_2/eval_point_param_seed0.png, q3_2/eval_point_seed0.png, q3_2/examples/, q3_2/q3_2_param_patches.png
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

RESULTS_DIR = "results"
EXAMPLES = ["240", "660", "150"]  # test chairs shown in the report, each from its most front-facing view
TRAIN_ITERS = 10001
SAVE_FREQ = 1000


def run(cmd, dry_run):
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run([sys.executable] + cmd[1:] if cmd[0] == "python" else cmd, check=True)


def collect(pattern, dest, dry_run):
    """eval_model.py writes its F1 plot to the working directory; move it next to the question's results."""
    if dry_run:
        return
    os.makedirs(dest, exist_ok=True)
    for path in glob.glob(pattern):
        shutil.move(path, os.path.join(dest, os.path.basename(path)))


def train(model_args, exp, opts):
    """Train unless the checkpoint already exists (or --retrain)."""
    checkpoint = f"checkpoint_{model_args[1]}{exp}.pth"
    if os.path.exists(checkpoint) and not opts.retrain:
        print(f"[skip training] {checkpoint} exists (use --retrain to train again)")
        return
    cmd = ["python", "train_model.py", *model_args, "--max_iter", str(TRAIN_ITERS), "--save_freq", str(SAVE_FREQ),
           "--num_workers", str(opts.num_workers), "--device", opts.device]
    if exp:
        cmd += ["--exp", exp]
    run(cmd, opts.dry_run)


def evaluate(model_args, exp, out, opts, extra=()):
    cmd = ["python", "eval_model.py", *model_args, "--load_checkpoint", "--device", opts.device,
           "--vis_dir", os.path.join(out, "eval"), "--vis_freq", "200", *extra]
    if exp:
        cmd += ["--exp", exp]
    run(cmd, opts.dry_run)
    collect(f"eval_{model_args[1]}*.png", out, opts.dry_run)


def examples(model_args, exp, out, opts):
    cmd = ["python", "vis_examples.py", *model_args, "--front", "--indices", *EXAMPLES, "--device", opts.device,
           "--vis_dir", os.path.join(out, "examples")]
    if exp:
        cmd += ["--exp", exp]
    run(cmd, opts.dry_run)


def figures(only, out, opts):
    run(["python", "interpret.py", "--only", only, "--device", opts.device, "--out_dir", out], opts.dry_run)


# --------------------------------------------------------------------------
# Question 1: Exploring loss functions
# --------------------------------------------------------------------------
def fit(t, opts):
    run(["python", "fit_data.py", "--type", t, "--lr", "1e-2", "--max_iter", "3000", "--device", opts.device,
         "--vis_dir", os.path.join(RESULTS_DIR, "q1")], opts.dry_run)


def q1_1(opts):
    """1.1 Fit a voxel grid with BCE loss."""
    fit("vox", opts)


def q1_2(opts):
    """1.2 Fit a point cloud with chamfer loss."""
    fit("point", opts)


def q1_3(opts):
    """1.3 Fit a mesh with chamfer + Laplacian smoothing loss."""
    fit("mesh", opts)


# --------------------------------------------------------------------------
# Question 2: Reconstructing 3D from a single view
# --------------------------------------------------------------------------
def single_view(t, q, opts):
    model_args = ["--type", t]
    out = os.path.join(RESULTS_DIR, q)
    train(model_args, "", opts)
    evaluate(model_args, "", out, opts)
    examples(model_args, "", out, opts)


def q2_1(opts):
    """2.1 Image to voxel grid."""
    single_view("vox", "q2_1", opts)


def q2_2(opts):
    """2.2 Image to point cloud."""
    single_view("point", "q2_2", opts)


def q2_3(opts):
    """2.3 Image to mesh."""
    single_view("mesh", "q2_3", opts)


def q2_4(opts):
    """2.4 Hyperparameter: voxel occupancy threshold (same input views for every setting)."""
    model_args = ["--type", "vox"]
    out = os.path.join(RESULTS_DIR, "q2_4")
    train(model_args, "", opts)
    for thr in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
        evaluate(model_args, "", out, opts, extra=["--seed", "0", "--vox_thresh", thr, "--vis_freq", "100000"])
    figures("thresholds", out, opts)


def q2_5(opts):
    """2.5 Interpretation: view consistency, rotated GT voxels, latent interpolation."""
    for t in ["vox", "point", "mesh"]:
        train(["--type", t], "", opts)
    out = os.path.join(RESULTS_DIR, "q2_5")
    for only in ["view", "orientation", "interp"]:
        figures(only, out, opts)


# --------------------------------------------------------------------------
# Question 3: Other architectures
# --------------------------------------------------------------------------
def q3_1(opts):
    """3.1 Implicit occupancy decoder (compared with 2.1 on the same views via --seed 0)."""
    model_args = ["--type", "vox", "--vox_dec", "implicit"]
    out = os.path.join(RESULTS_DIR, "q3_1")
    train(model_args, "_implicit", opts)
    for thr in ["0.5", "0.3"]:
        evaluate(model_args, "_implicit", out, opts, extra=["--seed", "0", "--vox_thresh", thr])
    examples(model_args, "_implicit", out, opts)


def q3_2(opts):
    """3.2 Parametric (AtlasNet-style) point decoder, plus the 2.2 decoder on the same views for comparison."""
    model_args = ["--type", "point", "--point_dec", "parametric", "--n_patches", "4"]
    out = os.path.join(RESULTS_DIR, "q3_2")
    train(model_args, "_param", opts)
    evaluate(model_args, "_param", out, opts, extra=["--seed", "0"])
    train(["--type", "point"], "", opts)
    evaluate(["--type", "point"], "", out, opts, extra=["--seed", "0", "--vis_freq", "100000"])
    examples(model_args, "_param", out, opts)
    figures("patches", out, opts)


QUESTIONS = {
    "1.1": q1_1, "1.2": q1_2, "1.3": q1_3,
    "2.1": q2_1, "2.2": q2_2, "2.3": q2_3, "2.4": q2_4, "2.5": q2_5,
    "3.1": q3_1, "3.2": q3_2,
}


def expand(selected):
    keys = []
    for q in selected:
        matches = [k for k in QUESTIONS if k == q or k.startswith(q + ".")]
        if not matches:
            sys.exit(f"Unknown question '{q}'. Use --list to see the options.")
        keys += [k for k in matches if k not in keys]
    return keys


def main():
    parser = argparse.ArgumentParser(description="Run all of Assignment 2 (16-825).")
    parser.add_argument("-q", "--questions", nargs="+", default=list(QUESTIONS), help="questions to run, e.g. 1 2.4 3.1")
    parser.add_argument("--list", action="store_true", help="list the questions and exit")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--num_workers", default=8, type=int, help="data-loader workers for training")
    parser.add_argument("--retrain", action="store_true", help="train even if the checkpoint already exists")
    parser.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    opts = parser.parse_args()

    if opts.list:
        for k, fn in QUESTIONS.items():
            print(fn.__doc__)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for k in expand(opts.questions):
        start = time.time()
        print(f"\n=== Q{k}: {QUESTIONS[k].__doc__}")
        QUESTIONS[k](opts)
        if not opts.dry_run:
            print(f"=== Q{k} done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
