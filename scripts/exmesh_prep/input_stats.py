"""CPU-only topology/availability stats for the two ExMesh input meshes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def stats(path: Path) -> dict:
    mesh = trimesh.load(str(path), process=False, validate=False, force="mesh")
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if len(mesh.face_adjacency):
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(faces))
        components = int(labels.max() + 1)
    else:
        components = int(len(faces))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a = faces.reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    graph = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(len(verts), len(verts))).tocsr()
    components_vertex = int(connected_components(graph, directed=False)[0])
    unreferenced = int(len(verts) - len(np.unique(faces)))
    components_vertex -= unreferenced
    tri = verts[faces]
    cross = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = float(cross.sum() / 2)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256_16": sha16(path),
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "components": components,
        "components_face": components,
        "components_vertex": components_vertex,
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "watertight": bool(mesh.is_watertight),
        "area_m2": round(area, 4),
        "area_cross_sum_m2": round(float(cross.sum()), 4),
        "bbox_min": np.round(verts.min(0), 4).tolist(),
        "bbox_max": np.round(verts.max(0), 4).tolist(),
        "euler_number": int(len(verts) - len(unique) + len(faces)),
    }

def main() -> None:
    out = {}
    for name, path in (("proxy", sys.argv[1]), ("vggt", sys.argv[2])):
        out[name] = stats(Path(path))
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
