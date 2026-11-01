"""Estimate scene extent from standard COLMAP points3D.bin without external packages."""
from __future__ import annotations
import math,struct
def robust_extent(colmap_dir):
    points=[]
    with (colmap_dir/"points3D.bin").open("rb") as f:
        h=f.read(8)
        if len(h)!=8: raise ValueError("Truncated COLMAP point header")
        count=struct.unpack("<Q",h)[0]
        stride=max(1,count//200000)
        for i in range(count):
            record=f.read(51)
            if len(record)!=51: raise ValueError("Truncated COLMAP point record")
            _,x,y,z,_,_,_,_,n=struct.unpack("<Q3d3BdQ",record)
            f.seek(8*n,1)
            if i%stride==0 and all(map(math.isfinite,(x,y,z))):
                points.append((x,y,z))
    if len(points)<10: raise ValueError("Too few valid points")
    extents=[]
    for j in range(3):
        values=sorted(row[j] for row in points)
        extents.append(values[int(.95*(len(values)-1))]-values[int(.05*(len(values)-1))])
    extent=math.sqrt(sum(x*x for x in extents))
    if not math.isfinite(extent) or extent<1e-8: raise ValueError("Degenerate extent")
    return extent
