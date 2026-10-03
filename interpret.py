"""
Figures for the report that go beyond eval_model.py (all rendered from trained checkpoints).

    python interpret.py --only view --out_dir results/q2_5

    thresholds   Q2.4  voxel predictions of three chairs at occupancy thresholds 0.1 / 0.3 / 0.5
    view         Q2.5  F1@0.05 of the vox / point / mesh models vs. the input view's azimuth
    orientation  Q2.5  dataset GT voxel grids that are rotated relative to their meshes
    interp       Q2.5  latent interpolation between two chairs (point and mesh models)
    patches      Q3.2  parametric-decoder predictions colored by patch
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes

import dataset_location
import visualize
from eval_model import evaluate, preprocess
from model import SingleViewto3D
from r2n2_custom import R2N2
from vis_examples import view_angles

TYPES = ["vox", "point", "mesh"]
COLORS = {"vox": "#2a78d6", "point": "#eb6834", "mesh": "#1baf7a"}  # categorical slots 1-3
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]
# fixed three-quarter front camera for renders that compare shapes across inputs
FIXED_VIEW = dict(azims=[150], elev=20)


def type_args(t, device):
    return argparse.Namespace(type=t, device=device, load_feat=False, arch="resnet18", n_points=1000,
                              batch_size=1, vox_thresh=0.5)


def load_model(t, device):
    args = type_args(t, device)
    model = SingleViewto3D(args)
    model.load_state_dict(torch.load(f"checkpoint_{t}.pth", map_location=device)["model_state_dict"])
    return model.to(device).eval(), args


def load_view(dataset, idx, view, device):
    feed_dict = collate_batched_R2N2([dataset[(idx, [view])]])
    images, mesh_gt = preprocess(feed_dict, type_args("point", device))
    return feed_dict, images, mesh_gt


def front_view(angles):
    return min(angles, key=lambda v: min(angles[v][0], 360 - angles[v][0]))


def render_prediction(t, pred, device, **view):
    if t == "vox":
        return visualize.render_mesh_360(visualize.voxels_to_mesh_eval_frame(pred[0, 0]), device=device, **view)[0]
    if t == "point":
        return visualize.render_points_360(pred[0], radius=0.02, device=device, **view)[0]
    mesh = visualize.make_mesh(pred.verts_list()[0], pred.faces_list()[0])
    return visualize.render_mesh_360(mesh, device=device, **view)[0]


def crop(im, pad=24):
    return im[pad:-pad, pad:-pad]


def style_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def view_consistency(dataset, models, device, out_dir, chair_step):
    records = []
    for idx in range(0, len(dataset), chair_step):
        angles = view_angles(dataset, idx)
        for view, (azim, _) in angles.items():
            feed_dict, images, mesh_gt = load_view(dataset, idx, view, device)
            for t, (model, args) in models.items():
                with torch.no_grad():
                    pred = model(images, args)
                f1 = float(evaluate(pred, mesh_gt, THRESHOLDS, args)["F1@0.050000"])
                records.append(dict(type=t, idx=idx, view=view, azim=azim, f1=f1))
    os.makedirs("logs", exist_ok=True)
    with open("logs/view_consistency.json", "w") as f:
        json.dump(records, f)

    # F1 vs. input azimuth, in 45-degree bins centered on the front (0) and back (180)
    centers = np.arange(0, 360, 45)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    summary = {}
    for t in TYPES:
        r = [x for x in records if x["type"] == t]
        az = np.array([x["azim"] for x in r]); f1 = np.array([x["f1"] for x in r])
        bins = np.round(((az + 22.5) % 360) // 45).astype(int) % 8
        means = np.array([f1[bins == b].mean() for b in range(8)])
        summary[t] = dict(zip(centers.tolist(), means.round(1).tolist()))
        ax.plot(centers, means, color=COLORS[t], linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=t)
        ax.annotate(t, (centers[-1], means[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color="#1c1c1a")
    ax.set_xticks(centers)
    ax.set_xticklabels(["0\nfront", "45", "90\nside", "135", "180\nback", "225", "270\nside", "315"])
    ax.set_xlabel("Input view azimuth (degrees)", fontsize=10, color="#4a4a45")
    ax.set_ylabel("Mean F1@0.05", fontsize=10, color="#4a4a45")
    ax.set_ylim(0, 100); ax.set_xlim(-15, 345)
    ax.grid(axis="y", color="#e3e2dd", linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#4a4a45", labelsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q2_5_view_f1.png"), dpi=130, facecolor="white")
    plt.close(fig)

    # per-chair spread across its 10 views
    spread = {}
    for t in TYPES:
        per_chair = {}
        for x in records:
            if x["type"] == t:
                per_chair.setdefault(x["idx"], []).append(x["f1"])
        spread[t] = float(np.mean([np.max(v) - np.min(v) for v in per_chair.values()]))
    return summary, spread, len({x["idx"] for x in records})






def encode(model, images):
    with torch.no_grad():
        return model.encoder(model.normalize(images.permute(0, 3, 1, 2))).squeeze(-1).squeeze(-1)


def decode(t, model, feat):
    with torch.no_grad():
        if t == "point":
            return model.decoder(feat)
        if t == "mesh":
            return model.mesh_pred.offset_verts(model.decoder(feat).reshape([-1, 3]))
        return torch.sigmoid(model.decoder(feat))


def interpolation(dataset, models, device, out_dir, idx_a, idx_b, alphas=(0, 0.25, 0.5, 0.75, 1)):
    inputs = []
    for idx in (idx_a, idx_b):
        view = front_view(view_angles(dataset, idx))
        feed_dict, images, _ = load_view(dataset, idx, view, device)
        inputs.append((feed_dict["images"][0].squeeze().cpu().numpy(), images))
    rows = ["point", "mesh"]
    fig, axes = plt.subplots(len(rows), len(alphas) + 2, figsize=(1.75 * (len(alphas) + 2), 2.1 * len(rows) + 0.3))
    for r, t in enumerate(rows):
        model, _ = models[t]
        z_a, z_b = encode(model, inputs[0][1]), encode(model, inputs[1][1])
        axes[r, 0].imshow(inputs[0][0]); axes[r, -1].imshow(inputs[1][0])
        for c, a in enumerate(alphas, start=1):
            pred = decode(t, model, (1 - a) * z_a + a * z_b)
            axes[r, c].imshow(crop(render_prediction(t, pred, device, **FIXED_VIEW)))
            if r == 0:
                axes[r, c].set_title(f"α = {a:g}", fontsize=9)
        axes[r, 0].set_ylabel(t, fontsize=10)
    axes[0, 0].set_title(f"A (sample {idx_a})", fontsize=9); axes[0, -1].set_title(f"B (sample {idx_b})", fontsize=9)
    for ax in axes.flat:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q2_5_interpolation.png"), dpi=110, facecolor="white")
    plt.close(fig)




R90_Y = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # 90-degree rotation about the vertical axis


def voxel_orientation(sample):
    """Chamfer distance between a chair's GT voxels (in the eval frame) and its GT mesh, for 0/90/180/270-degree turns."""
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.structures import Meshes
    vox = visualize.voxels_to_mesh_eval_frame(sample["voxels"][0])
    if vox.isempty():
        return None
    p = sample_points_from_meshes(vox, 3000)
    v = sample["verts"] - sample["verts"].mean(0)
    g = sample_points_from_meshes(Meshes([v], [sample["faces"]]), 3000)
    return [chamfer_distance(p @ torch.linalg.matrix_power(R90_Y, k).T, g)[0].item() for k in range(4)]


def scan_voxel_orientation(dataset, step=3):
    """A chair counts as rotated if some 90/180/270-degree turn fits its mesh at least 2x better than none."""
    torch.manual_seed(0)
    records = []
    for idx in range(0, len(dataset), step):
        cds = voxel_orientation(dataset[(idx, [dataset.views_per_model_list[idx][0]])])
        if cds is None:
            continue
        k = int(np.argmin(cds))
        records.append(dict(idx=idx, cd=cds, best_k=k, ratio=cds[0] / cds[k]))
    os.makedirs("logs", exist_ok=True)
    with open("logs/voxel_orientation_test.json", "w") as f:
        json.dump(records, f)
    print(f"{sum(r['ratio'] >= 2 for r in records)}/{len(records)} test chairs fit their mesh >=2x better after a rotation")
    return records


def voxel_orientation_examples(dataset, device, out_dir, indices):
    """GT mesh (top row) above the GT voxels as provided (bottom row), one column per chair whose voxels are rotated."""
    fig, axes = plt.subplots(2, len(indices), figsize=(2.6 * len(indices), 5.0))
    for c, idx in enumerate(indices):
        sample = dataset[(idx, [dataset.views_per_model_list[idx][0]])]
        v = sample["verts"] - sample["verts"].mean(0)
        gt = visualize.make_mesh(v, visualize.dedupe_faces(v, sample["faces"]))
        vox = visualize.voxels_to_mesh_eval_frame(sample["voxels"][0])
        for r, mesh in enumerate([gt, vox]):
            axes[r, c].imshow(crop(visualize.render_mesh_360(mesh, device=device, **FIXED_VIEW)[0]))
        axes[0, c].set_title(f"sample {idx}", fontsize=9)
    axes[0, 0].set_ylabel("Ground-truth mesh", fontsize=9)
    axes[1, 0].set_ylabel("Ground-truth voxels", fontsize=9)
    for ax in axes.flat:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q2_5_vox_orientation.png"), dpi=110, facecolor="white")
    plt.close(fig)

PATCH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4


def param_patches(dataset, device, out_dir, indices, views=(150, 90, 210)):
    """3.2 parametric decoder: each predicted point colored by the patch that produced it."""
    args = argparse.Namespace(type="point", point_dec="parametric", n_patches=4, n_points=1000, device=device,
                              load_feat=False, arch="resnet18", batch_size=1)
    model = SingleViewto3D(args)
    model.load_state_dict(torch.load("checkpoint_point_param.pth", map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    per_patch = args.n_points // args.n_patches
    colors = torch.tensor([matplotlib.colors.to_rgb(c) for c in PATCH_COLORS]).repeat_interleave(per_patch, 0)
    fig, axes = plt.subplots(len(indices), len(views) + 1, figsize=(2.3 * (len(views) + 1), 2.35 * len(indices) + 0.6))
    for r, idx in enumerate(indices):
        feed_dict, images, _ = load_view(dataset, idx, front_view(view_angles(dataset, idx)), device)
        with torch.no_grad():
            points = model(images, args)[0]
        axes[r, 0].imshow(feed_dict["images"][0].squeeze().cpu().numpy())
        for c, azim in enumerate(views, start=1):
            im = visualize.render_points_360(points, radius=0.02, device=device, colors=colors, azims=[azim], elev=20)[0]
            axes[r, c].imshow(crop(im))
            if r == 0:
                axes[r, c].set_title(f"azimuth {azim}\u00b0", fontsize=9)
        axes[r, 0].set_ylabel(f"sample {idx}", fontsize=9)
    axes[0, 0].set_title("Input", fontsize=9)
    for ax in axes.flat:
        style_axes(ax)
    handles = [matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=8, color=col, label=f"patch {k + 1}")
               for k, col in enumerate(PATCH_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out_dir, "q3_2_param_patches.png"), dpi=110, facecolor="white")
    plt.close(fig)


def threshold_figure(dataset, device, out_dir, indices, thresholds=(0.1, 0.3, 0.5)):
    """Q2.4: the voxel model's prediction for each chair at several occupancy thresholds, from the input camera."""
    model, args = load_model("vox", device)
    fig, axes = plt.subplots(len(indices), len(thresholds) + 2, figsize=(10, 2.1 * len(indices)))
    for r, idx in enumerate(indices):
        angles = view_angles(dataset, idx)
        view = front_view(angles)
        feed_dict, images, mesh_gt = load_view(dataset, idx, view, device)
        cam = dict(azims=[180 - angles[view][0]], elev=angles[view][1])  # our azimuth = 180 - R2N2 azimuth
        with torch.no_grad():
            pred = model(images, args)
        cells = [("Input", feed_dict["images"][0].squeeze().cpu().numpy())]
        for t in thresholds:
            args.vox_thresh = t
            torch.manual_seed(0)  # F1 samples points from both surfaces; fix the samples so the figure is reproducible
            f1 = float(evaluate(pred, mesh_gt, THRESHOLDS, args)["F1@0.050000"])
            mesh = visualize.voxels_to_mesh_eval_frame(pred[0, 0], thresh=t)
            cells.append((f"t = {t}\nF1 {f1:.1f}", crop(visualize.render_mesh_360(mesh, device=device, **cam)[0], 20)))
        gt_verts = mesh_gt.verts_list()[0].to(device)
        gt_verts = gt_verts - gt_verts.mean(0)
        gt = visualize.make_mesh(gt_verts, visualize.dedupe_faces(gt_verts, mesh_gt.faces_list()[0].to(device)))
        cells.append(("Ground truth", crop(visualize.render_mesh_360(gt, device=device, **cam)[0], 20)))
        for c, (title, im) in enumerate(cells):
            axes[r, c].imshow(im)
            axes[r, c].set_title(title if r == 0 or 0 < c <= len(thresholds) else "", fontsize=9)
        axes[r, 0].set_ylabel(f"sample {idx}", fontsize=9)
    args.vox_thresh = 0.5
    for ax in axes.flat:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q2_4_vox_thresh.png"), dpi=110, facecolor="white")
    plt.close(fig)


def main(args):
    dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH)
    os.makedirs(args.out_dir, exist_ok=True)
    todo = [args.only] if args.only else ["thresholds", "view", "orientation", "interp", "patches"]
    if "thresholds" in todo:
        threshold_figure(dataset, args.device, args.out_dir, args.example_idx)
    if "view" in todo:
        models = {t: load_model(t, args.device) for t in TYPES}
        summary, spread, n = view_consistency(dataset, models, args.device, args.out_dir, args.chair_step)
        print(f"view consistency over {n} chairs x 10 views; mean F1 per 45-degree azimuth bin:")
        for t in TYPES:
            print(f"  {t}: {summary[t]}; mean per-chair F1 range across views: {spread[t]:.1f}")
    if "orientation" in todo:
        vox_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
                           dataset_location.SPLITS_PATH, return_voxels=True)
        scan_voxel_orientation(vox_dataset)
        voxel_orientation_examples(vox_dataset, args.device, args.out_dir, args.orientation_idx)
    if "interp" in todo:
        models = {t: load_model(t, args.device) for t in ("point", "mesh")}
        interpolation(dataset, models, args.device, args.out_dir, *args.interp_idx)
    if "patches" in todo:
        param_patches(dataset, args.device, args.out_dir, args.example_idx)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Report figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="interpret")
    parser.add_argument("--only", default=None, choices=["thresholds", "view", "orientation", "interp", "patches"],
                        help="render a single figure (default: all)")
    parser.add_argument("--chair_step", default=12, type=int, help="use every n-th test chair for view consistency")
    parser.add_argument("--example_idx", nargs="+", default=[240, 660, 150], type=int,
                        help="chairs for the threshold and patch figures (the report's examples)")
    parser.add_argument("--interp_idx", nargs=2, default=[240, 660], type=int)
    parser.add_argument("--orientation_idx", nargs="+", default=[510, 276, 54], type=int,
                        help="chairs whose GT voxels are rotated relative to their mesh")
    main(parser.parse_args())
