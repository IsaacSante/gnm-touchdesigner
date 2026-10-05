# face_fit -- Script CHOP callbacks: drive the GNM head from face landmarks.
#
# Input 0: the component's 636 coefficient channels (override). Output: the
# same channels with tracked expression (and calibrated identity) added, plus
# pose channels (degrees) that xforms_callbacks adds to the Pose page:
#   pose_head_rx/ry/rz, pose_lefteye_rx/ry, pose_righteye_rx/ry
#
# Landmarks come from the CHOP on the Tracking page -- an Info CHOP on the
# Apple Face Landmarks TOP (channels face_present, image_width, image_height,
# p000x ... p477z: MediaPipe's 478 landmarks, normalized, y down).
#
# Per frame (see tools/face_fit/build_face_fit.py for the precomputed data):
#   1. similarity-fit the skull-fixed landmarks onto the neutral reference;
#      the rotation removed is the head pose
#   2. what's left is displacement from neutral: one matrix product gives the
#      expression coefficients
#   3. iris displacement over the eyeball radius gives the gaze angles
# "Calibrate Neutral Face" records the current displacement as the baseline
# (your resting face = zero expression) and fits identity to it.
#
# The solver functions are plain numpy so tools/face_fit/test_face_fit.py can
# run them outside TouchDesigner.

import io
import math

import numpy as np

_data = None
_smoothed = None
_proj_cache = {}


def load_data(raw_bytes):
    z = np.load(io.BytesIO(raw_bytes))
    return {k: z[k] for k in z.files}


def projections(d, depth_weight):
    """Weighted ridge projection matrices (expression, identity).

    c = (B^T W B + lam I)^-1 B^T W y, with W weighting each point's z by
    depth_weight: MediaPipe's depth is its least reliable output, worst on an
    open mouth, and trusting it fully makes the lips balloon outwards.
    """
    key = round(float(depth_weight), 4)
    if key not in _proj_cache:
        out = []
        for basis, lam in ((d['expr_basis'], d['expr_regularization']),
                           (d['id_basis'], d['id_regularization'])):
            b = basis.T.astype(np.float64)                    # (N, K)
            w = np.tile([1.0, 1.0, key], b.shape[0] // 3)
            btw = b.T * w
            out.append(np.linalg.solve(btw @ b + float(lam) * np.eye(b.shape[1]),
                                       btw).astype(np.float32))
        _proj_cache.clear()
        _proj_cache[key] = tuple(out)
    return _proj_cache[key]


def fit_similarity(source, target):
    """Umeyama: scale, rotation, translation minimizing |s R src + t - tgt|^2."""
    sc, tc = source.mean(0), target.mean(0)
    s0, t0 = source - sc, target - tc
    u, sv, vt = np.linalg.svd(t0.T @ s0 / len(source))
    corr = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        corr[-1] = -1.0
    rot = u @ np.diag(corr) @ vt
    scale = float(np.sum(sv * corr) / max(np.sum(s0 * s0) / len(source), 1e-12))
    return scale, rot, tc - scale * rot @ sc


def euler_xyz_deg(rot):
    """Angles (rx, ry, rz) in degrees with rot = Rz @ Ry @ Rx, xforms' convention."""
    ry = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    rx = math.atan2(rot[2, 1], rot[2, 2])
    rz = math.atan2(rot[1, 0], rot[0, 0])
    return np.degrees([rx, ry, rz])


def landmark_displacement(d, landmarks, aspect):
    """Displacement of each corresponded landmark from the neutral reference,
    in GNM model space with head pose removed; and the head rotation."""
    # MediaPipe -> GNM axes: y up, z towards the viewer; x and z share the
    # width normalization, so both take the aspect ratio.
    obs = np.stack([landmarks[:, 0] * aspect, -landmarks[:, 1],
                    -landmarks[:, 2] * aspect], axis=-1)[d['landmark_indices']]
    ref = d['reference']
    rigid = d['rigid']
    scale, rot, trans = fit_similarity(obs[rigid], ref[rigid])
    aligned = scale * obs @ rot.T + trans
    # rot maps the observation into the neutral frame; the head's own rotation
    # is its inverse.
    return aligned - ref, rot.T


def solve(d, disp, head_rot, baseline=None, depth_weight=0.2):
    """Expression coefficients, head Euler angles, and per-eye gaze angles."""
    if baseline is not None:
        disp = disp - baseline
    expr = projections(d, depth_weight)[0] @ disp[d['expr_rows']].ravel()
    r = float(d['eye_radius'])
    gaze = {}
    for side in ('left', 'right'):
        dx, dy = disp[int(d[side + '_iris_row']), :2]
        # +ry swings the eye's front (+z) towards +x; +rx swings it towards -y.
        gaze[side] = np.degrees([-math.atan2(dy, r), math.atan2(dx, r)])
    return expr, euler_xyz_deg(head_rot), gaze


def solve_identity(d, baseline, depth_weight=0.2):
    return projections(d, depth_weight)[1] @ baseline[d['expr_rows']].ravel()


# ---------------------------------------------------------------- TouchDesigner

def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return


def _pose_names():
    return (['pose_head_rx', 'pose_head_ry', 'pose_head_rz'] +
            ['pose_%seye_%s' % (s, a) for s in ('left', 'right') for a in ('rx', 'ry')])


def onCook(scriptOp):
    global _data, _smoothed
    comp = parent()
    src = scriptOp.inputs[0]
    scriptOp.copy(src)
    pose_names = _pose_names()
    pose = np.zeros(len(pose_names))

    lm_chop = comp.par.Landmarkchop.eval()
    present = (comp.par.Tracking.eval() and lm_chop is not None
               and lm_chop['face_present'] is not None
               and lm_chop['face_present'][0] > 0.5)

    if _data is None:
        _data = load_data(bytes(comp.vfs['face_fit.npz'].byteArray))
    d = _data
    baseline = scriptOp.fetch('baseline', None, search=False)
    if baseline is not None and np.shape(baseline) != d['reference'].shape:
        # Calibrated against an older face_fit.npz (different landmark set).
        scriptOp.unstore('baseline')
        scriptOp.unstore('identity')
        baseline = None
        print('GNM face tracking: calibration was made with older tracking data '
              'and has been cleared -- press Calibrate Neutral Face again.')

    if present:
        arr = lm_chop.numpyArray()
        start = lm_chop['p000x'].index
        landmarks = arr[start:start + 478 * 3, 0].reshape(478, 3)
        aspect = lm_chop['image_width'][0] / max(1.0, lm_chop['image_height'][0])
        depth_weight = float(comp.par.Depthweight.eval())
        disp, head_rot = landmark_displacement(d, landmarks, aspect)

        if scriptOp.fetch('calibrate_request', False, search=False):
            scriptOp.unstore('calibrate_request')
            baseline = disp.copy()
            scriptOp.store('baseline', baseline)
            scriptOp.store('identity', solve_identity(d, baseline, depth_weight))
            _smoothed = None

        expr, head, gaze = solve(d, disp, head_rot, baseline, depth_weight)
        # Clamp after the gain: GNM's coefficients are meant to stay within
        # about +/-3, and beyond that shapes extrapolate into artefacts.
        limit = float(comp.par.Maxcoefficient.eval())
        frame = np.concatenate([
            np.clip(expr * comp.par.Expressiongain.eval(), -limit, limit),
            head,
            gaze['left'] * comp.par.Gazegain.eval(),
            gaze['right'] * comp.par.Gazegain.eval()])
        k = float(comp.par.Smoothing.eval())
        _smoothed = frame if _smoothed is None else k * _smoothed + (1 - k) * frame

    if comp.par.Tracking.eval() and _smoothed is not None:
        # Hold the last pose while the face is lost.
        n = len(d['expr_channels'])
        for name, v in zip(d['expr_channels'], _smoothed[:n]):
            c = scriptOp[str(name)]
            if c is not None:
                c[0] += float(v)
        pose = _smoothed[n:]

    identity = scriptOp.fetch('identity', None, search=False)
    if identity is not None and comp.par.Applyidentity.eval():
        for name, v in zip(d['id_channels'], identity):
            c = scriptOp[str(name)]
            if c is not None:
                c[0] += float(v)

    # Pose channels go after the 636 coefficients; the shader only reads the
    # first 636 samples, and xforms reads these by name.
    for name, v in zip(pose_names, pose):
        scriptOp.appendChan(name)[0] = float(v)
    return
