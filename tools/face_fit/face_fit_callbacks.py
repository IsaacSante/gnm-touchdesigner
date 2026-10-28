# face_fit -- Script CHOP callbacks: drive the GNM head from face landmarks.
#
# Input 0: the component's 636 coefficient channels (override). Output: the
# same channels with tracked expression (and calibrated identity) added, plus
# pose channels (degrees) that xforms_callbacks adds to the Pose page:
#   pose_head_rx/ry/rz, pose_lefteye_rx/ry, pose_righteye_rx/ry
# and emotion scores 0..1 with the winner's index (always present):
#   neutral happy surprise angry sad fear dominant_id
#
# Input comes from the face_tracking COMP of Torin Blankensmith's MediaPipe
# plugin, set on the Tracking page. It reads two of its outputs:
#   facePoints    channels x, y, z, one sample per landmark (478 per face).
#                 x is MediaPipe's, y is flipped up and scaled to width units,
#                 z is MediaPipe's. All zeros on x when there is no face.
#   blend_shapes  MediaPipe's 52 face blendshapes, one sample per face.
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


def landmark_displacement(d, landmarks):
    """Displacement of each corresponded landmark from the neutral reference,
    in GNM model space with head pose removed; and the head rotation.

    landmarks: (478, 3) as face_tracking outputs them -- x, y and z all in
    image-width units, y up. Scale and offset don't matter: the similarity
    fit removes them.
    """
    # -> GNM axes: z towards the viewer
    obs = (landmarks * [1.0, 1.0, -1.0])[d['landmark_indices']]
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
    # TouchDesigner's numpy (Accelerate BLAS) raises spurious divide/overflow
    # warnings from matmul on Apple Silicon; the results are finite.
    with np.errstate(all='ignore'):
        expr = projections(d, depth_weight)[0] @ disp[d['expr_rows']].ravel()
    r = float(d['eye_radius'])
    gaze = {}
    for side in ('left', 'right'):
        dx, dy = disp[int(d[side + '_iris_row']), :2]
        # +ry swings the eye's front (+z) towards +x; +rx swings it towards -y.
        gaze[side] = np.degrees([-math.atan2(dy, r), math.atan2(dx, r)])
    return expr, euler_xyz_deg(head_rot), gaze


def solve_identity(d, baseline, depth_weight=0.2):
    with np.errstate(all='ignore'):
        return projections(d, depth_weight)[1] @ baseline[d['expr_rows']].ravel()


def limit_expression(expr, limit):
    """Scale the whole vector down so no coefficient exceeds the limit.

    Clamping each coefficient separately breaks the combination the solver
    found -- e.g. it cuts the jaw-opening component but keeps the smaller
    ones that shape the lips around an open jaw, which leaves odd lips on a
    half-open mouth. Scaling keeps the shape and only reduces its strength.
    """
    peak = float(np.max(np.abs(expr))) if len(expr) else 0.0
    return expr * (limit / peak) if peak > limit > 0 else expr


def tongue_for(d, expr):
    with np.errstate(all='ignore'):
        return d['tongue_follow'] @ expr


EMOTIONS = ['neutral', 'happy', 'surprise', 'angry', 'sad', 'fear']

# Each emotion is a weighted sum of MediaPipe blendshapes (names without the
# Left/Right suffix are the average of both sides), tuned on captured faces so
# a clearly held expression scores about 0.7..1 at the default Emotion Gain.
# headTilt isn't a blendshape: it's the head's pitch/roll, 0 at 9 degrees and
# 1 at 22 -- MediaPipe sees little else in a fearful face.
EMOTION_WEIGHTS = {
    'happy':    {'mouthSmile': 1.0},
    'surprise': {'jawOpen': 0.4, 'browInnerUp': 0.3, 'browOuterUp': 0.3},
    'angry':    {'browDown': 1.0, 'eyeSquint': 0.3},
    'sad':      {'mouthFrown': 2.0, 'eyeLookDown': 0.5},
    'fear':     {'headTilt': 0.6, 'eyeWide': 0.2, 'mouthStretch': 0.2},
}


def head_tilt(head_rot):
    rx, _, rz = euler_xyz_deg(head_rot)
    return float(np.clip((math.hypot(rx, rz) - 9.0) / 13.0, 0.0, 1.0))


def emotion_scores(shapes, baseline=None, gain=1.0):
    """Scores 0..1 in EMOTIONS order from a {blendshape name: value} dict.

    baseline: the same dict captured on the neutral face; resting values
    (many faces rest with some brow raise or lip press) are subtracted.
    neutral is what's left when no other emotion is showing.
    """
    def get(name):
        if name in shapes:
            return max(0.0, shapes[name] - (baseline or {}).get(name, 0.0))
        if name.endswith(('Left', 'Right')):
            return 0.0
        return 0.5 * (get(name + 'Left') + get(name + 'Right'))

    active = [min(1.0, gain * sum(w * get(n) for n, w in EMOTION_WEIGHTS[e].items()))
              for e in EMOTIONS[1:]]
    return np.array([1.0 - max(active)] + active)


# ---------------------------------------------------------------- TouchDesigner

def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return


def _pose_names():
    return (['pose_head_rx', 'pose_head_ry', 'pose_head_rz'] +
            ['pose_%seye_%s' % (s, a) for s in ('left', 'right') for a in ('rx', 'ry')])


def _read_face(ft):
    """(landmarks (478, 3), {blendshape: value}) from face_tracking, or None."""
    pts = ft.op('facePoints') if ft is not None else None
    if pts is None or pts.numSamples < 478 or pts.chan('x') is None:
        return None
    # first face only; channel order there is y, x, z, so go by name
    landmarks = np.stack([pts.chan(a).numpyArray()[:478] for a in 'xyz'], -1)
    if not np.any(landmarks[:, 0]):
        return None                                    # no face: all zeros
    bs = ft.op('blend_shapes')
    shapes = {c.name: float(c[0]) for c in bs.chans()} if bs is not None and bs.numSamples else {}
    return landmarks, shapes


def onCook(scriptOp):
    global _data, _smoothed
    comp = parent()
    src = scriptOp.inputs[0]
    scriptOp.copy(src)
    pose_names = _pose_names()
    pose = np.zeros(len(pose_names))
    emotions = np.eye(len(EMOTIONS))[0]               # neutral until tracked

    tracking = comp.par.Tracking.eval()
    face = _read_face(comp.par.Facetracking.eval()) if tracking else None

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

    if face is not None:
        landmarks, shapes = face
        depth_weight = float(comp.par.Depthweight.eval())
        disp, head_rot = landmark_displacement(d, landmarks)
        shapes['headTilt'] = head_tilt(head_rot)

        if scriptOp.fetch('calibrate_request', False, search=False):
            scriptOp.unstore('calibrate_request')
            baseline = disp.copy()
            scriptOp.store('baseline', baseline)
            scriptOp.store('baseline_shapes', dict(shapes))
            scriptOp.store('identity', solve_identity(d, baseline, depth_weight))
            _smoothed = None

        expr, head, gaze = solve(d, disp, head_rot, baseline, depth_weight)
        # Limit after the gain: GNM's coefficients are meant to stay within
        # about +/-3, and beyond that shapes extrapolate into artefacts.
        limit = float(comp.par.Maxcoefficient.eval())
        frame = np.concatenate([
            limit_expression(expr * comp.par.Expressiongain.eval(), limit),
            head,
            gaze['left'] * comp.par.Gazegain.eval(),
            gaze['right'] * comp.par.Gazegain.eval(),
            emotion_scores(shapes, scriptOp.fetch('baseline_shapes', None, search=False),
                           float(comp.par.Emotiongain.eval()))])
        k = float(comp.par.Smoothing.eval())
        _smoothed = frame if _smoothed is None else k * _smoothed + (1 - k) * frame

    if tracking and _smoothed is not None:
        # Hold the last pose while the face is lost.
        n = len(d['expr_channels'])
        values = list(zip(d['expr_channels'], _smoothed[:n]))
        if comp.par.Tonguefollowsjaw.eval():
            values += list(zip(d['tongue_channels'], tongue_for(d, _smoothed[:n])))
        for name, v in values:
            c = scriptOp[str(name)]
            if c is not None:
                c[0] += float(v)
        pose = _smoothed[n:n + len(pose_names)]
        emotions = _smoothed[n + len(pose_names):]

    identity = scriptOp.fetch('identity', None, search=False)
    if identity is not None and comp.par.Applyidentity.eval():
        for name, v in zip(d['id_channels'], identity):
            c = scriptOp[str(name)]
            if c is not None:
                c[0] += float(v)

    # Pose and emotion channels go after the 636 coefficients; the shader only
    # reads the first 636 samples for the morph, xforms reads pose by name.
    # The channel list never changes, so nothing downstream re-binds.
    for name, v in zip(pose_names, pose):
        scriptOp.appendChan(name)[0] = float(v)
    for name, v in zip(EMOTIONS, emotions):
        scriptOp.appendChan(name)[0] = float(v)
    scriptOp.appendChan('dominant_id')[0] = float(np.argmax(emotions))
    return
