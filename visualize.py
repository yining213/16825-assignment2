"""
360-degree renderers for voxels, point clouds and meshes (adapted from assignment 1's starter/utils.py).
"""
import imageio
import numpy as np
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
)


def get_points_renderer(image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)):
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=256, lights=None, device=None):
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def get_cameras(azim, dist=1.5, elev=20, device=None):
    R, T = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    return pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)


def voxels_to_mesh(voxels, thresh=0.5, color=(0.7, 0.7, 1.0)):
    """
    voxels: (Z, Y, X) or (1, Z, Y, X) occupancy probabilities in [0, 1].
    Returns a textured Meshes object in the same world frame as the R2N2 meshes ([-0.5, 0.5]^3).
    """
    voxels = voxels.detach().float()
    if voxels.dim() == 3:
        voxels = voxels.unsqueeze(0)
    # with align="center", cubify puts voxel i's center at -1 + 2i/(D-1), with x <-> W, y <-> H,
    # z <-> D (matching the (Z, Y, X) layout from utils_vox). utils_vox puts voxel i's center at
    # i/D - 0.5, so rescale to that frame.
    D = voxels.shape[-1]
    mesh = cubify(voxels, thresh=thresh, align="center")
    verts = (mesh.verts_list()[0] + 1) * (D - 1) / (2 * D) - 0.5
    faces = mesh.faces_list()[0]
    return make_mesh(verts, faces, color)


def make_mesh(verts, faces, color=(0.7, 0.7, 1.0)):
    verts = verts.detach().float()
    textures = torch.ones_like(verts) * torch.tensor(color, device=verts.device)
    return pytorch3d.structures.Meshes(
        verts=[verts], faces=[faces], textures=pytorch3d.renderer.TexturesVertex([textures]),
    )


def render_mesh_360(mesh, image_size=256, n_frames=36, dist=1.5, elev=20, device=None, azims=None):
    device = device or mesh.device
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images = []
    # azims: optional explicit camera azimuths (degrees) instead of a full turn
    for azim in (np.linspace(0, 360, n_frames, endpoint=False) if azims is None else azims):
        cameras = get_cameras(azim, dist=dist, elev=elev, device=device)
        # place the light at the camera so the visible side is always lit
        lights = pytorch3d.renderer.PointLights(
            location=cameras.get_camera_center(), ambient_color=((0.6, 0.6, 0.6),),
            diffuse_color=((0.4, 0.4, 0.4),), device=device,
        )
        if mesh.isempty():
            rend = torch.ones(1, image_size, image_size, 3, device=device)
        else:
            rend = renderer(face_camera(mesh, cameras.get_camera_center()[0]), cameras=cameras, lights=lights)
        images.append(rend[0, ..., :3].detach().cpu().numpy().clip(0, 1))
    return images


def face_camera(mesh, camera_center):
    """
    Two-sided lighting: flip every face whose normal points away from the camera, so thin parts
    (chair backs, folded predicted surfaces) are lit on whichever side is visible.
    """
    verts, faces = mesh.verts_packed(), mesh.faces_packed()
    tri = verts[faces]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    away = (normals * (camera_center - tri.mean(1))).sum(1) < 0
    faces = torch.where(away[:, None], faces[:, [0, 2, 1]], faces)
    return pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=mesh.textures)


def render_points_360(points, image_size=256, n_frames=36, dist=1.5, elev=20,
                      color=(0.7, 0.7, 1.0), radius=0.01, device=None, azims=None, colors=None):
    """points: (N, 3) or (1, N, 3); colors: optional per-point RGB (N, 3) in [0, 1]"""
    points = points.detach().float().reshape(-1, 3)
    device = device or points.device
    points = points.to(device)
    rgb = torch.ones_like(points) * torch.tensor(color, device=device) if colors is None else colors.float().to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=[points], features=[rgb])
    renderer = get_points_renderer(image_size=image_size, device=device, radius=radius)
    images = []
    # azims: optional explicit camera azimuths (degrees) instead of a full turn
    for azim in (np.linspace(0, 360, n_frames, endpoint=False) if azims is None else azims):
        cameras = get_cameras(azim, dist=dist, elev=elev, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        images.append(rend[0, ..., :3].detach().cpu().numpy().clip(0, 1))
    return images


def save_gif(path, *image_lists, fps=15):
    """Saves one or more equal-length lists of frames side by side as a gif."""
    frames = [
        (np.concatenate(frame_set, axis=1) * 255).astype(np.uint8)
        for frame_set in zip(*image_lists)
    ]
    imageio.mimsave(path, frames, duration=1000 // fps, loop=0) # type: ignore


def voxels_to_mesh_eval_frame(voxels, thresh=0.5, color=(0.7, 0.7, 1.0)):
    """
    voxels: (Z, Y, X) occupancy probabilities. Places the cubified voxels in the same frame that
    eval_model.evaluate() scores them in: index (i0, i1, i2) -> Mem2Ref, then a -pi rotation about y,
    then re-centering. This lines the voxels up with the (re-centered) gt mesh.
    """
    voxels = voxels.detach().float()
    D = voxels.shape[-1]
    mesh = cubify(voxels[None], thresh=thresh, align="center")
    if mesh.isempty():
        return mesh
    c = mesh.verts_list()[0]
    idx = (c + 1) * (D - 1) / 2  # cubify (x, y, z) = voxel index along (dim2, dim1, dim0)
    ref = torch.stack([idx[:, 2], idx[:, 1], idx[:, 0]], dim=1) / D - 0.5  # Mem2Ref of (i0, i1, i2)
    verts = torch.stack([-ref[:, 0], ref[:, 1], -ref[:, 2]], dim=1)  # rotate by -pi about y
    verts = verts - verts.mean(0)
    # swapping x and z is a reflection, so flip the winding to keep the normals pointing outward
    faces = mesh.faces_list()[0][:, [0, 2, 1]]
    return make_mesh(verts, faces, color)


def dedupe_faces(verts, faces):
    """
    ShapeNet meshes store every triangle twice with opposite winding, which cancels the vertex
    normals and renders them dark. Keep one copy of each triangle, oriented away from the center.
    """
    key = faces.sort(dim=1).values
    _, inverse = torch.unique(key, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), len(faces), device=faces.device)
    first = first.scatter_reduce(0, inverse, torch.arange(len(faces), device=faces.device), reduce="amin")
    faces = faces[first]
    tri = verts[faces]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    inward = (normals * (tri.mean(1) - verts.mean(0))).sum(1) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def save_comparison_png(path, *images):
    """Saves images (H, W, 3) in [0, 1] side by side, resized to the same height."""
    import torch.nn.functional as F
    h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        t = torch.as_tensor(np.ascontiguousarray(img)).float().permute(2, 0, 1)[None]
        if t.shape[-2] != h:
            t = F.interpolate(t, size=(h, h * t.shape[-1] // t.shape[-2]), mode="bilinear", align_corners=False)
        resized.append(t[0].permute(1, 2, 0).numpy().clip(0, 1))
    imageio.imwrite(path, (np.concatenate(resized, axis=1) * 255).astype(np.uint8))
