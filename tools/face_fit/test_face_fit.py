"""Checks the face-fit solver outside TouchDesigner.

    python3 tools/face_fit/test_face_fit.py <gnm_head.npz> [plugin_results.txt]

1. Synthetic: pose a GNM face with a known expression, head rotation and gaze,
   turn it into MediaPipe-style landmarks, and check the solver recovers it.
2. Real (optional): solve landmarks produced by the Apple Face Landmarks
   plugin's test harness (tests/work/results.txt in that repo) and render the
   GNM head with the result to tools/face_fit/work/.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_fit_callbacks as ff  # noqa: E402

gnm = np.load(sys.argv[1])
with open(os.path.join(HERE, 'face_fit.npz'), 'rb') as f:
    d = ff.load_data(f.read())
template = gnm['template_vertex_positions']
expr_names = [str(n) for n in gnm['expression_names']]
idx_of = {('e_' + n): i for i, n in enumerate(expr_names)}
comp_idx = np.array([idx_of[str(c)] for c in d['expr_channels']])
basis_c = gnm['expression_basis'][comp_idx]                 # (K, V, 3)
li, vi = d['landmark_indices'], d['vertex_indices']


def rot_xyz(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def to_mediapipe(model_pts, aspect):
    """Inverse of the solver's axis mapping (scale is arbitrary; use 2.5x)."""
    p = model_pts * 2.5 + [0.5 * aspect, -0.5, 0]
    out = np.zeros((478, 3))
    out[li] = np.stack([p[:, 0] / aspect, -p[:, 1], -p[:, 2] / aspect], -1)
    return out


# ---------------------------------------------------------------- 1. synthetic
rng = np.random.default_rng(0)
print('Synthetic recovery (expression drawn from N(0, 1) on the solved components):')
print('%-24s %10s %10s %10s %12s' % ('case', 'expr corr', 'expr gain', 'head err', 'gaze err'))
for case, (rx, ry, rz), (gl, gr) in [('frontal', (0, 0, 0), ((0, 0), (0, 0))),
                                     ('head turned', (10, -25, 8), ((0, 0), (0, 0))),
                                     ('gaze right/up', (0, 0, 0), ((-10, 15), (-10, 15)))]:
    e = rng.normal(0, 1, len(comp_idx))
    disp = np.einsum('k,kvc->vc', e, basis_c[:, vi])        # expression displacement
    model = d['reference'] + disp
    # gaze: move each iris centre landmark as the eye rotation would
    for side, (grx, gry) in (('left', gl), ('right', gr)):
        row = int(d[side + '_iris_row'])
        r = float(d['eye_radius'])
        model[row, 0] += r * math.tan(math.radians(gry))
        model[row, 1] += -r * math.tan(math.radians(grx))
    R = rot_xyz(rx, ry, rz)
    centre = d['reference'][d['rigid']].mean(0)
    posed = (model - centre) @ R.T + centre
    lms = to_mediapipe(posed, 16 / 9)

    disp_s, head_rot = ff.landmark_displacement(d, lms, 16 / 9)
    expr, head, gaze = ff.solve(d, disp_s, head_rot)
    corr = np.corrcoef(expr, e)[0, 1]
    gain = float(expr @ e / (e @ e))
    head_err = np.abs(head - [rx, ry, rz]).max()
    gaze_err = max(np.abs(gaze['left'] - gl).max(), np.abs(gaze['right'] - gr).max())
    print('%-24s %10.3f %10.3f %9.2f° %11.2f°' % (case, corr, gain, head_err, gaze_err))

# ---------------------------------------------------------------- 2. real
if len(sys.argv) > 2:
    work = os.path.join(HERE, 'work')
    os.makedirs(work, exist_ok=True)
    rows = [l.split() for l in open(sys.argv[2])]
    full_basis = gnm['expression_basis']
    tris = gnm['triangles']
    skin = gnm['vertex_groups'][[str(n) for n in gnm['vertex_group_names']].index('skin')] > 0.5

    def render(verts, path, title):
        """Flat-shaded orthographic front view (painter's algorithm)."""
        S, img = 512, Image.new('RGB', (512, 560), (24, 24, 28))
        dr = ImageDraw.Draw(img)
        t = tris[skin[tris].all(1)]
        v = verts[t]                                           # (T, 3, 3)
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        lit = np.clip(n @ np.array([0.3, 0.4, 0.85]) / 1.0, 0, 1)
        front = n[:, 2] > 0
        order = np.argsort(v[:, :, 2].mean(1))
        c = verts[skin].mean(0); ext = np.abs(verts[skin, :2] - c[:2]).max() * 1.05
        for i in order:
            if not front[i]:
                continue
            pts = [((p[0] - c[0]) / ext * S / 2 + S / 2, -(p[1] - c[1]) / ext * S / 2 + S / 2 + 40) for p in v[i]]
            g = int(60 + 180 * lit[i])
            dr.polygon(pts, fill=(g, int(g * 0.9), int(g * 0.85)))
        dr.text((10, 10), title, fill=(230, 230, 230))
        img.save(path)

    render(template, os.path.join(work, 'neutral.png'), 'GNM neutral')
    for row in rows:
        if int(row[1]) != 2 or not int(row[2]):
            continue
        name = os.path.splitext(os.path.basename(row[0]))[0]
        lms = np.array(row[7:], float).reshape(478, 3)
        w, h = Image.open(row[0]).size
        disp_s, head_rot = ff.landmark_displacement(d, lms, w / h)
        expr, head, gaze = ff.solve(d, disp_s, head_rot)
        full = np.zeros(len(expr_names))
        full[comp_idx] = np.clip(expr * 1.5, -3, 3)             # default gain and clamp
        verts = template + np.einsum('k,kvc->vc', full, full_basis)
        top = np.argsort(-np.abs(expr))[:4]
        print('\n%s: head rx %.1f ry %.1f rz %.1f deg, gaze L %s R %s' % (
            name, *head, np.round(gaze['left'], 1), np.round(gaze['right'], 1)))
        print('  strongest:', ', '.join('%s %.2f' % (d['expr_channels'][i], expr[i]) for i in top))
        render(verts, os.path.join(work, name + '_fit.png'), name + ' (expression only, gain 1.5, clamp 3)')
    print('\nRenders in', os.path.relpath(work))
