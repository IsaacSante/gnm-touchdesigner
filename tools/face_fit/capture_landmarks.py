# Save the current face landmarks to a file, for tuning the face fit offline.
# Run from the Textport while making the face you want to capture:
#
#   CAPTURE='open'; p='/Users/isaac/Documents/prototypes/human-control/gnm-touchdesigner/tools/face_fit/capture_landmarks.py'; exec(open(p).read(), dict(globals(), __file__=p))
#
# Reads the face_tracking COMP set on the GNM component's Face Tracking page (or set
# FACE_TRACKING='/path/to/face_tracking';). Writes tools/face_fit/work/capture_<name>.npz
# with the 478 landmarks (face_tracking's layout), the blendshapes, and the
# component's settings and calibration for comparison. Captures 30 frames (half a second) so noise can
# be measured too.

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(globals().get('__file__', '')))


def _capture(name, frames=30):
    comp = op(globals().get('GNM_COMP', '/project1/google_gnm_model'))
    src = globals().get('FACE_TRACKING')
    ft = op(src) if src else comp.par.Facetracking.eval()
    if ft is None or ft.op('facePoints') is None:
        raise RuntimeError('no face_tracking COMP -- set FACE_TRACKING to its path')
    read_face = comp.op('face_fit_callbacks').module._read_face
    out_dir = os.path.join(HERE, 'work')
    os.makedirs(out_dir, exist_ok=True)
    samples, shapes = [], []

    def grab(i):
        face = read_face(ft)
        if face is not None:
            samples.append(face[0].copy())
            shapes.append(face[1])
        if i + 1 < frames:
            run(lambda: grab(i + 1), delayFrames=1)
            return
        if not samples:
            raise RuntimeError('no face seen in %d frames' % frames)
        names = sorted(shapes[0])
        path = os.path.join(out_dir, 'capture_%s.npz' % name)
        np.savez(path, layout='face_tracking', landmarks=np.array(samples),
                 blendshape_names=np.array(names),
                 blendshapes=np.array([[s.get(n, 0.0) for n in names] for s in shapes]),
                 settings=np.array([comp.par.Expressiongain.eval(), comp.par.Maxcoefficient.eval(),
                                    comp.par.Depthweight.eval()]),
                 baseline=comp.op('face_fit').fetch('baseline', np.zeros(0), search=False))
        print('Saved %d frames to %s' % (len(samples), path))

    grab(0)


_capture(globals().get('CAPTURE', 'capture'))
