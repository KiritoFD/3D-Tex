"""Geometry losses shared by the local M2 experiments; no dataset or GT access."""
import torch
import torch.nn.functional as F


def regularizer_graph(vertices, faces):
    """Unique undirected edges and crease weights from incident face/vertex normals."""
    tri = vertices[faces]
    normals = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
    vertex_normals = torch.zeros_like(vertices)
    for corner in range(3):
        vertex_normals.index_add_(0, faces[:, corner], normals)
    vertex_normals = F.normalize(vertex_normals, dim=1)
    crease = torch.zeros(len(vertices), device=vertices.device)
    count = torch.zeros_like(crease)
    for corner in range(3):
        disagreement = 1 - (normals * vertex_normals[faces[:, corner]]).sum(-1).clamp(-1, 1)
        crease.index_add_(0, faces[:, corner], disagreement)
        count.index_add_(0, faces[:, corner], torch.ones_like(disagreement))
    weights = torch.exp(-4 * crease / count.clamp(min=1))
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = torch.unique(edges.sort(dim=1).values, dim=0)
    edges = torch.cat([edges, edges.flip(1)])
    degree = torch.zeros(len(vertices), device=vertices.device)
    degree.index_add_(0, edges[:, 0], torch.ones(len(edges), device=vertices.device))
    return edges, degree, weights


def displacement_loss(delta, edges, degree, weights, voxel):
    neighbor = torch.zeros_like(delta)
    neighbor.index_add_(0, edges[:, 0], delta[edges[:, 1]])
    laplacian = delta - neighbor / degree.clamp(min=1)[:, None]
    return (weights[:, None] * laplacian.abs() / voxel).mean()


def tolerance_loss(rendered, initial, confidence, valid, epsilon, voxel):
    """Dead zone in scene units; Huber on excess error in voxel units."""
    excess = (rendered - initial).abs().sub(epsilon).clamp(min=0) / voxel
    loss = F.huber_loss(excess, torch.zeros_like(excess), delta=1, reduction="none")
    weight = confidence * valid
    return (weight * loss).sum() / weight.sum().clamp(min=1)


def make_lattice(vertices, voxel, spacing_voxels):
    """Fixed trilinear sampling coordinates; parameters are displacements in voxels."""
    if spacing_voxels <= 0:
        raise ValueError("lattice spacing must be positive")
    spacing = voxel * spacing_voxels
    low = vertices.amin(0) - spacing
    dims = torch.ceil((vertices.amax(0) - low) / spacing).long() + 2
    high = low + (dims - 1) * spacing
    grid = (2 * (vertices - low) / (high - low) - 1)[None, :, None, None, :].detach()
    nx, ny, nz = dims.tolist()
    field = torch.zeros((1,3,nz,ny,nx), dtype=vertices.dtype,
                        device=vertices.device, requires_grad=True)
    return field, grid


def lattice_offsets(field, grid):
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)[0, :, :, 0, 0].T.contiguous()
