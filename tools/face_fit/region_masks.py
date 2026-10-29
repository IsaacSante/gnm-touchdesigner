# Per-vertex anatomical regions of the GNM head, for the Skin Colors page.
# Run by install_face_tracking.py, which embeds the result as masks.npy;
# script1 packs it into the unused .w of sBasis rows 0..9 for glsl1:
#
#   region  0 skin and lips, 1 teeth and gums, 2 tongue,
#           3 inner eye (sclera, iris, pupil), 4 outer cornea, 5 inner mouth
#   sub     skin: lip signed distance mapped to 0..1, 0.5 on the lip border
#           teeth: 1 on the crowns, 0 at the gum line
#           eyes: distance from the eye's centre in metres (front only)
#           inner mouth: depth into the mouth, 0..1
#
# The mesh is 8 disconnected shells; sorted by size they are the head, lower
# and upper teeth, tongue, two cornea domes, and two inner eyes.

from collections import defaultdict

import numpy as np

# Outer vermilion border, one side (x >= 0), mirrored to make the lips
# symmetric: Cupid's bow centre -> peak -> corner -> lower lip -> bottom centre.
LIP_HALF_CONTOUR = [
    (0.0000, 0.2371), (0.0059, 0.2378), (0.0130, 0.2361), (0.0182, 0.2337),
    (0.0235, 0.2309), (0.0218, 0.2262), (0.0185, 0.2236), (0.0125, 0.2216),
    (0.0062, 0.2209), (0.0000, 0.2210),
]


def read_obj(raw):
    verts, faces = [], []
    for ln in raw.decode('utf-8').splitlines():
        if ln.startswith('v '):
            verts.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('f '):
            faces.append([int(x.split('/')[0]) - 1 for x in ln.split()[1:]])
    return np.array(verts, dtype=np.float32), faces


def shells(n_verts, faces):
    """Connected vertex sets, largest first."""
    root = list(range(n_verts))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for f in faces:
        r0 = find(f[0])
        for v in f[1:]:
            root[find(v)] = r0
    groups = defaultdict(list)
    for i in range(n_verts):
        groups[find(i)].append(i)
    return [np.array(g, dtype=np.int32) for g in sorted(groups.values(), key=len, reverse=True)]


def signed_distance(pts, poly):
    """Distance from 2D points to a closed polygon, positive inside."""
    px, py = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    d2 = np.full(len(pts), 1e12, dtype=np.float32)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        inside ^= ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1)
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy + 1e-12), 0.0, 1.0)
        d2 = np.minimum(d2, (px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2)
    d = np.sqrt(d2)
    return np.where(inside, d, -d)


def build(obj_bytes):
    """(region, sub) per vertex, float32."""
    verts, faces = read_obj(obj_bytes)
    parts = shells(len(verts), faces)
    region = np.zeros(len(verts), dtype=np.float32)
    sub = np.zeros(len(verts), dtype=np.float32)

    # head: lips by distance to the mirrored border; the inner mouth behind them
    head = parts[0]
    hp = verts[head]
    right = LIP_HALF_CONTOUR[1:-1]
    poly = np.array([LIP_HALF_CONTOUR[0]] + right + [LIP_HALF_CONTOUR[-1]] +
                    [(-x, y) for x, y in reversed(right)], dtype=np.float32)
    lip = np.clip(0.5 + signed_distance(hp[:, :2], poly) / 0.012, 0.0, 1.0)
    lip[hp[:, 2] < 0.1165] = 0.0
    sub[head] = lip
    cavity = (np.abs(hp[:, 0]) < 0.023) & (hp[:, 1] > 0.224) & (hp[:, 1] < 0.244) & (hp[:, 2] < 0.1145)
    region[head[cavity]] = 5.0
    sub[head[cavity]] = np.clip((0.1165 - hp[cavity, 2]) / 0.008, 0.0, 1.0)

    # teeth: crowns are tooth colour, blending to gums only at the roots
    for idxs in parts[1:3]:
        region[idxs] = 1.0
        pts = verts[idxs]
        y = pts[:, 1]
        if y.mean() < 0.228:                                   # lower teeth
            gum, crown = np.percentile(y, 12), np.percentile(y, 22)
            sub[idxs] = np.clip((y - gum) / max(crown - gum, 1e-5), 0.0, 1.0)
        else:                                                  # upper teeth
            crown, gum = np.percentile(y, 78), np.percentile(y, 88)
            sub[idxs] = np.clip((gum - y) / max(gum - crown, 1e-5), 0.0, 1.0)
        sub[idxs[pts[:, 2] > np.percentile(pts[:, 2], 20)]] = 1.0

    region[parts[3]] = 2.0                                     # tongue

    # eyes: radial distance on the front half picks pupil / iris / sclera
    for idxs, rid in ((parts[4], 4.0), (parts[5], 4.0), (parts[6], 3.0), (parts[7], 3.0)):
        region[idxs] = rid
        pts = verts[idxs]
        r = np.hypot(pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean())
        sub[idxs] = np.where(pts[:, 2] >= np.median(pts[:, 2]), r, 1.0)

    return region, sub
