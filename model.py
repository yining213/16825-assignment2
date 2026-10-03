from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class VoxelDecoder(nn.Module):
    # b x 512 -> fc -> b x 256 x 4^3 -> 3 x (transposed conv, 2x upsample) -> b x 1 x 32^3
    def __init__(self, latent_dim=512):
        super(VoxelDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8^3
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 16^3
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 32^3
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat):
        x = self.fc(feat).view(-1, 256, 4, 4, 4)
        return self.deconv(x)

class PointDecoder(nn.Module):
    def __init__(self, latent_dim=512, n_points=1000):
        super(PointDecoder, self).__init__()
        self.n_points = n_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_points * 3)
        )

    def forward(self, feat):
        return self.fc(feat).view(-1, self.n_points, 3)

class MeshDecoder(nn.Module):
    # b x 512 -> MLP -> per-vertex offsets (b x n_verts*3) applied to the template mesh
    def __init__(self, latent_dim=512, n_verts=2562):
        super(MeshDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_verts * 3),
            nn.Tanh(),  # offsets in [-1, 1]: enough to move the unit sphere onto the ~[-0.5, 0.5] chairs
        )

    def forward(self, feat):
        return self.fc(feat)
    
class ImplicitDecoder(nn.Module):
    # occupancy(z, p) = MLP([p; z]) evaluated on a 32^3 grid in (-1, 1)^3
    def __init__(self, latent_dim=512, hidden=256, res=32):
        super(ImplicitDecoder, self).__init__()
        self.res = res
        lin = torch.linspace(-1, 1, res)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy, zz], dim=-1).view(-1, 3))
        # first layer on [p; z], split so the image feature is projected once per image
        self.fc_p = nn.Linear(3, hidden)
        self.fc_z = nn.Linear(latent_dim, hidden)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        B = feat.shape[0]
        h = self.fc_p(self.grid)[None] + self.fc_z(feat)[:, None]   # (1, N, H) + (B, 1, H) -> (B, N, H)
        logits = self.mlp(h)                                          # (B, N, 1)
        return logits.view(B, 1, self.res, self.res, self.res)

class ParametricDecoder(nn.Module):
    # AtlasNet-style: n_patches MLPs, each maps a 2D sample (u, v) in [0, 1]^2 + image feature z -> 3D point
    def __init__(self, latent_dim=512, n_points=1000, n_patches=4, hidden=256):
        super(ParametricDecoder, self).__init__()
        assert n_points % n_patches == 0, "n_points must be divisible by n_patches"
        self.n_patches = n_patches
        self.pts_per_patch = n_points // n_patches
        # fixed uv samples for eval, so predictions are deterministic; training resamples every step
        gen = torch.Generator().manual_seed(0)
        self.register_buffer("uv_eval", torch.rand(n_patches, self.pts_per_patch, 2, generator=gen))
        # first layer on [uv; z], split so the image feature is projected once per image
        self.fc_uv = nn.ModuleList([nn.Linear(2, hidden) for _ in range(n_patches)])
        self.fc_z = nn.ModuleList([nn.Linear(latent_dim, hidden) for _ in range(n_patches)])
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 3),
            )
            for _ in range(n_patches)
        ])

    def forward(self, feat):
        B = feat.shape[0]
        if self.training:
            uv = torch.rand_like(self.uv_eval)
        else:
            uv = self.uv_eval
        patches = []
        for k in range(self.n_patches):
            h = self.fc_uv[k](uv[k])[None] + self.fc_z[k](feat)[:, None]  # (1, M, H) + (B, 1, H) -> (B, M, H)
            patches.append(self.mlp[k](h))                                 # (B, M, 3)
        return torch.cat(patches, dim=1)                                   # (B, n_patches * M, 3)


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32 (occupancy logits)
            if getattr(args, 'vox_dec', 'conv') == 'implicit':
                self.decoder = ImplicitDecoder()
            else:
                self.decoder = VoxelDecoder()
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            if getattr(args, 'point_dec', 'fc') == 'parametric':
                self.decoder = ParametricDecoder(n_points=args.n_points, n_patches=args.n_patches)
            else:
                self.decoder = PointDecoder(n_points=args.n_points)
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.decoder = MeshDecoder(n_verts=mesh_pred.verts_packed().shape[0])

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)
            if not self.training:                        # eval 模式 → 轉成機率
                voxels_pred = torch.sigmoid(voxels_pred)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

