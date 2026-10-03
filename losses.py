import torch
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = torch.nn.functional.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_tgt: b x n_points x 3  
	# squared euclidean distances between all pairs: b x n_src x n_tgt
	dist = torch.cdist(point_cloud_src, point_cloud_tgt) ** 2
	loss_chamfer = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean() # [0]->value, [1]->indices
	return loss_chamfer

def smoothness_loss(mesh_src):
	# uniform laplacian: distance of each vertex from the mean of its neighbors
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	return loss_laplacian
