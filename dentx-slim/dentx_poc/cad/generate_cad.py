#!/usr/bin/env python3
"""
Generates 3D-printable comfort/fit dummies (STL, millimetres) for DentX-Slim Phase B tests.
  dentx_slim_dummy.stl      44 x 30 x 4.0 mm, 5 mm corner radius, 2 mm tapered rim (4.0 -> 2.0 mm), flex-tail stub
  conventional_dummy.stl    44.1 x 30.4 x 7.1 mm reference (KaVo ProXam iS outer dimensions), small edge radius
These are SOLID, non-functional shapes for hand-feel, positioning and comfort studies. Not medical devices.
Author: Sai Venkata Ajay Uppalapati
"""
import os, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
NC = 10  # points per corner arc

def rrect(w, h, r, inset=0.0):
    """Rounded-rectangle ring (counter-clockwise), constant point count, optionally inset."""
    w2, h2 = w / 2 - inset, h / 2 - inset
    rr = max(r - inset, 0.05)
    pts = []
    for (cx, cy, a0) in [(w2 - rr, h2 - rr, 0), (-(w2 - rr), h2 - rr, 90), (-(w2 - rr), -(h2 - rr), 180), (w2 - rr, -(h2 - rr), 270)]:
        for k in range(NC + 1):
            a = np.radians(a0 + 90 * k / NC)
            pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    return np.array(pts)

def loft(rings):
    """rings: list of (points Nx2, z). Returns triangles (M,3,3) of a closed solid with outward normals."""
    tris = []
    n = len(rings[0][0])
    for (p0, z0), (p1, z1) in zip(rings[:-1], rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            a, b = (*p0[i], z0), (*p0[j], z0)
            c, d = (*p1[j], z1), (*p1[i], z1)
            tris += [(a, b, c), (a, c, d)]
    for p, z, up in [(rings[0][0], rings[0][1], False), (rings[-1][0], rings[-1][1], True)]:
        c = (*p.mean(0), z)
        for i in range(n):
            j = (i + 1) % n
            a, b = (*p[i], z), (*p[j], z)
            tris.append((c, a, b) if up else (c, b, a))
    return np.array(tris, float)

def box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in f])

def volume(t):
    return float(np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) / 6.0)

def check_closed(t):
    from collections import Counter
    key = lambda p: tuple(np.round(p, 5))
    e = Counter()
    for tri in t:
        for i in range(3):
            a, b = key(tri[i]), key(tri[(i + 1) % 3])
            e[(a, b)] += 1
    bad = sum(1 for (a, b), c in e.items() if e.get((b, a), 0) != c)
    return bad == 0

def write_stl(path, t, name):
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for tri in t:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); L = np.linalg.norm(n)
            n = n / L if L > 0 else n
            f.write(f" facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n  outer loop\n")
            for v in tri: f.write(f"   vertex {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")

def dentx():
    body = loft([(rrect(44, 30, 5.0), 0.0), (rrect(44, 30, 5.0), 2.0), (rrect(44, 30, 5.0, 2.0), 4.0)])
    tail = box(21.5, 29.5, -7.5, -2.5, 0.0, 1.2)       # short stub of the side-exit flex tail (overlaps body; slicers union it)
    return body, tail

def conventional():
    return loft([(rrect(44.1, 30.4, 0.8), 0.0), (rrect(44.1, 30.4, 0.8), 6.8), (rrect(44.1, 30.4, 0.8, 0.3), 7.1)])

if __name__ == "__main__":
    body, tail = dentx()
    conv = conventional()
    for name, t in [("dentx_slim_body", body), ("conventional_dummy", conv)]:
        print(f"{name}: triangles={len(t)}  volume={volume(t):.1f} mm3  watertight={check_closed(t)}")
    write_stl(os.path.join(OUT, "dentx_slim_dummy.stl"), np.concatenate([body, tail]), "dentx_slim_dummy")
    write_stl(os.path.join(OUT, "conventional_dummy.stl"), conv, "conventional_dummy")
    # preview
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 4.2))
    for k, (t, ttl, col) in enumerate([(conv, "Conventional dummy: 7.1 mm", "#C9D3DD"), (np.concatenate([body, tail]), "DentX-Slim dummy: 4.0 mm, tapered rim", "#7FD1C8")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        light = np.array([0.35, -0.5, 0.8]); light /= np.linalg.norm(light)
        sh = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        base = np.array(matplotlib.colors.to_rgb(col))
        pc = Poly3DCollection(t, facecolors=np.clip(base * sh[:, None], 0, 1), edgecolor="none", linewidths=0)
        ax.add_collection3d(pc)
        ax.set_xlim(-25, 30); ax.set_ylim(-18, 18); ax.set_zlim(0, 18); ax.set_box_aspect((55, 36, 18))
        ax.view_init(elev=24, azim=-58); ax.set_axis_off(); ax.set_title(ttl, fontsize=10, color="#0B2545")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "..", "figures", "fig_cad_preview.png"), dpi=150, bbox_inches="tight")
    print("STL files and preview written")
