"""CUDA rasterizer for geometric M2 losses (WSL nvdiffrast)."""
import torch


class NvdRenderer:
    """nvdiffrast full-frame soft renderer (WSL acceleration path).

    render_frame returns (depth, alpha) images; losses gather the pixels they
    need. Gradients flow to vertex positions through barycentrics (depth) and
    through antialias coverage (alpha)."""

    def __init__(self, near=1.0, far=10.0, perspective=False):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        self.near, self.far = near, far
        self.perspective = perspective

    def _rasterize(self, vertices, faces_i32, k, t, res, attrs=None):
        cam = (t[:3, :3] @ vertices.transpose(-1, -2)).transpose(-1, -2) + t[:3, 3]
        z = cam[..., 2].clamp(min=1e-6)
        w = z
        # nvdiffrast NDC: x right, y DOWN (verified vs GT foreground, IoU 0.947)
        x_ndc = (2 * k[0, 0] / res) * cam[..., 0] / z + (2 * k[0, 2] / res - 1)
        y_ndc = (2 * k[1, 1] / res) * cam[..., 1] / z + (2 * k[1, 2] / res - 1)
        z_ndc = (2 * z - self.near - self.far) / (self.far - self.near)
        if self.perspective:
            z_ndc = ((self.far+self.near)/(self.far-self.near)
                     - (2*self.far*self.near)/(self.far-self.near)/z)
        clip = torch.stack([x_ndc * w, y_ndc * w, z_ndc * w, w], -1).contiguous()
        rast, _ = self.dr.rasterize(self.glctx, clip.unsqueeze(0), faces_i32,
                                    resolution=[res, res])
        channels = [cam[..., 2:3], torch.ones_like(cam[..., 2:3])]
        if attrs is not None:
            channels.append(attrs)
        attr = torch.cat(channels, -1).contiguous()
        col, _ = self.dr.interpolate(attr.unsqueeze(0), rast, faces_i32)
        col = self.dr.antialias(col, rast, clip.unsqueeze(0), faces_i32)
        alpha = col[0, ..., 1].clamp(0, 1)
        # Antialiasing mixes attributes with background zero. Undo that coverage
        # factor before using camera-z as a geometric constraint.
        depth = torch.where(alpha > 1e-6, col[0, ..., 0] / alpha.clamp(min=1e-6),
                            torch.zeros_like(alpha))
        return depth, alpha, (col[0, ..., 2:] if attrs is not None else None), cam

    def render_frame(self, vertices, faces_i32, k, t, res):
        depth, alpha, _, _ = self._rasterize(vertices, faces_i32, k, t, res)
        return depth, alpha

    def render_frame_normals(self, vertices, faces_i32, vertex_normals, k, t, res):
        """Depth, alpha and a camera-space vertex-normal buffer in one rasterisation."""
        camera_normals = (t[:3, :3] @ vertex_normals.transpose(-1, -2)).transpose(-1, -2)
        depth, alpha, normals, _ = self._rasterize(vertices, faces_i32, k, t, res,
                                                   attrs=camera_normals)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        # Orient interpolated normals towards the camera (winding is not trusted).
        rows, cols = torch.meshgrid(torch.arange(res, device=depth.device),
                                    torch.arange(res, device=depth.device), indexing="ij")
        point = torch.stack([(cols - k[0, 2]) / k[0, 0] * depth,
                             (rows - k[1, 2]) / k[1, 1] * depth, depth], -1)
        facing = torch.where((normals * point).sum(-1, keepdim=True) > 0, -1.0, 1.0)
        return depth, alpha, normals * facing


