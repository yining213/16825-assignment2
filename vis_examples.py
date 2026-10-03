"""
Visualize specific test-set examples with a trained checkpoint, without running the full evaluation.

Usage:
    python vis_examples.py --type vox --indices 200 400 --device cpu
    python vis_examples.py --type point --indices 20 140 --front   # front-facing input views
"""
import argparse
import os
import random

import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2

import dataset_location
import visualize
from eval_model import evaluate, get_args_parser, preprocess
from model import SingleViewto3D
from r2n2_custom import R2N2


def view_angles(r2n2_dataset, idx):
    """Returns {view index: (azimuth, elevation)} of the R2N2 renderings available for test sample idx."""
    model = r2n2_dataset._get_item_ids(idx)
    metadata_path = os.path.join(
        r2n2_dataset.r2n2_dir, r2n2_dataset.views_rel_path, model["synset_id"], model["model_id"],
        "rendering", "rendering_metadata.txt",
    )
    with open(metadata_path) as f:
        lines = f.readlines()
    return {v: tuple(float(x) for x in lines[v].split()[:2]) for v in r2n2_dataset.views_per_model_list[idx]}


def main(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    model = SingleViewto3D(args)
    checkpoint = torch.load(f"checkpoint_{args.type}{args.exp}.pth", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    print(f"Loaded checkpoint_{args.type}{args.exp}.pth (step {checkpoint['step']})")

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    for idx in args.indices:
        angles = view_angles(r2n2_dataset, idx)
        if args.front:
            # R2N2 azimuth 0 looks at the front of the chair; pick the view closest to it
            view = min(angles, key=lambda v: min(angles[v][0], 360 - angles[v][0]))
            sample = r2n2_dataset[(idx, [view])]
        else:
            # R2N2 picks a random input view per load; seed it so each example is reproducible
            random.seed(args.seed + idx)
            sample = r2n2_dataset[idx]
            view = None
        feed_dict = collate_batched_R2N2([sample])
        images_gt, mesh_gt = preprocess(feed_dict, args)
        with torch.no_grad():
            predictions = model(images_gt, args)
        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # same rendering as the visualization block in eval_model.py
        recenter = args.type == "vox"  # evaluate() re-centers both voxel predictions and gt points
        if args.type == "vox":
            pred_mesh = visualize.voxels_to_mesh_eval_frame(predictions[0, 0], thresh=args.vox_thresh)  # model outputs probabilities in eval mode
            render_pred = lambda **kw: visualize.render_mesh_360(pred_mesh, device=args.device, **kw)
        elif args.type == "point":
            render_pred = lambda **kw: visualize.render_points_360(predictions[0], radius=0.02, device=args.device, **kw)
        elif args.type == "mesh":
            pred_mesh = visualize.make_mesh(predictions.verts_list()[0], predictions.faces_list()[0])
            render_pred = lambda **kw: visualize.render_mesh_360(pred_mesh, device=args.device, **kw)
        gt_verts = mesh_gt.verts_list()[0].to(args.device)
        if recenter:
            gt_verts = gt_verts - gt_verts.mean(0)
        gt_mesh = visualize.make_mesh(gt_verts, visualize.dedupe_faces(gt_verts, mesh_gt.faces_list()[0].to(args.device)))
        render_gt = lambda **kw: visualize.render_mesh_360(gt_mesh, device=args.device, **kw)

        pred_images, gt_images = render_pred(), render_gt()
        if view is not None:
            # render the still from the input camera: our azimuth = 180 - R2N2 azimuth
            azim, elev = angles[view]
            still = dict(azims=[180 - azim], elev=elev)
            pred_still, gt_still = render_pred(**still)[0], render_gt(**still)[0]
        else:
            pred_still, gt_still = pred_images[0], gt_images[0]

        png_path = os.path.join(args.vis_dir, f"{idx}_{args.type}.png")
        visualize.save_comparison_png(png_path, feed_dict["images"][0].squeeze().cpu().numpy(), pred_still, gt_still)
        visualize.save_gif(os.path.join(args.vis_dir, f"{idx}_{args.type}.gif"), pred_images, gt_images)
        view_info = f"; view {view} (azimuth {angles[view][0]:.0f})" if view is not None else ""
        print(f"[{idx}] model {feed_dict['model_id'][0]}{view_info}; F1@0.05: {float(metrics['F1@0.050000']):.3f} -> {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize examples", parents=[get_args_parser()])
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--front", action="store_true", help="use the most front-facing input view of each chair")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 0
    os.makedirs(args.vis_dir, exist_ok=True)
    main(args)
