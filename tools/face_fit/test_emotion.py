"""Checks the blendshape emotion scores and the face_tracking landmark path.

    python3 tools/face_fit/test_emotion.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_fit_callbacks as ff  # noqa: E402

E = ff.EMOTIONS


def winner(shapes, baseline=None):
    return E[int(np.argmax(ff.emotion_scores(shapes, baseline, gain=1.5)))]


# A resting face like the one MediaPipe reported in testing: some brow raise,
# lip press and squint with no expression on purpose.
rest = {'browInnerUp': 0.69, 'browOuterUpLeft': 0.31, 'browOuterUpRight': 0.61,
        'mouthPressLeft': 0.71, 'mouthPressRight': 0.34, 'eyeSquintLeft': 0.53,
        'eyeSquintRight': 0.49, 'mouthRollLower': 0.88}
cases = {
    'neutral':  {},
    'happy':    {'mouthSmileLeft': 0.9, 'mouthSmileRight': 0.9, 'cheekSquintLeft': 0.5, 'cheekSquintRight': 0.5},
    'surprise': {'jawOpen': 0.8, 'browInnerUp': 0.9, 'browOuterUpLeft': 0.9, 'browOuterUpRight': 0.9,
                 'eyeWideLeft': 0.5, 'eyeWideRight': 0.5},
    'angry':    {'browDownLeft': 0.9, 'browDownRight': 0.9, 'eyeSquintLeft': 0.5, 'eyeSquintRight': 0.5},
    'sad':      {'mouthFrownLeft': 0.8, 'mouthFrownRight': 0.8, 'browInnerUp': 0.6, 'mouthShrugLower': 0.5},
    'fear':     {'headTilt': 1.0, 'eyeWideLeft': 0.5, 'eyeWideRight': 0.5},
}
fails = 0
for want, shapes in cases.items():
    calibrated = dict(rest, **{k: rest.get(k, 0) + v for k, v in shapes.items()})
    got_raw, got_cal = winner(shapes), winner(calibrated, rest)
    ok = got_raw == want and got_cal == want
    fails += not ok
    print('%-9s raw -> %-9s calibrated -> %-9s %s' % (want, got_raw, got_cal, 'ok' if ok else 'FAIL'))
s = ff.emotion_scores({})
assert s[0] == 1 and not s[1:].any(), s
print('uncalibrated resting face ->', winner(rest), '(why Calibrate Neutral Face matters)')

# Captured faces (capture_landmarks.py): every frame should score as the
# face it was captured as, calibrated against capture_neutral and not.
with open(os.path.join(HERE, 'face_fit.npz'), 'rb') as f:
    d = ff.load_data(f.read())
work = os.path.join(HERE, 'work')


def captured(e):
    path = os.path.join(work, 'capture_%s.npz' % e)
    if not os.path.exists(path) or 'layout' not in np.load(path).files:
        return None
    z = np.load(path)
    names = [str(n) for n in z['blendshape_names']]
    out = []
    for lms, values in zip(z['landmarks'], z['blendshapes']):
        shapes = dict(zip(names, map(float, values)))
        _, rot = ff.landmark_displacement(d, lms)
        shapes['headTilt'] = ff.head_tilt(rot)
        out.append(shapes)
    return out


faces = {e: captured(e) for e in E}
if all(faces.values()):
    rest = {k: float(np.mean([f[k] for f in faces['neutral']])) for k in faces['neutral'][0]}
    for label, base in (('calibrated', rest), ('uncalibrated', None)):
        for e, frames in faces.items():
            got = [winner(f, base) for f in frames]
            right = got.count(e)
            fails += right != len(got)
            print('capture %-9s %-13s %2d/%d frames' % (e, label, right, len(got)))
else:
    print('no face_tracking captures for every emotion in', work, '-- skipped')

sys.exit(1 if fails else 0)
