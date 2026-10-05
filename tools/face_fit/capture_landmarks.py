# Save the current face landmarks to a file, for tuning the face fit offline.
# Run from the Textport while making the face you want to capture:
#
#   CAPTURE='open'; p='/Users/isaac/Documents/prototypes/human-control/gnm-touchdesigner/tools/face_fit/capture_landmarks.py'; exec(open(p).read(), dict(globals(), __file__=p))
#
# Reads the Landmarks CHOP from the GNM component's Tracking page (or set
# LANDMARKS_CHOP='/path/to/info1';). Writes tools/face_fit/work/capture_<name>.npz
# with the 478 landmarks, image size, and the component's current fitted
# expression for comparison. Captures 30 frames (half a second) so noise can
# be measured too.

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(globals().get('__file__', '')))


def _capture(name, frames=30):
    comp = op(globals().get('GNM_COMP', '/project1/google_gnm_model'))
    src = globals().get('LANDMARKS_CHOP')
    chop = op(src) if src else comp.par.Landmarkchop.eval()
    if chop is None or chop['p000x'] is None:
        raise RuntimeError('no landmarks CHOP -- set LANDMARKS_CHOP to your Info CHOP path')
    out_dir = os.path.join(HERE, 'work')
    os.makedirs(out_dir, exist_ok=True)
    samples = []

    def grab(i):
        arr = chop.numpyArray()[:, 0]
        start = chop['p000x'].index
        samples.append(arr[start:start + 478 * 3].reshape(478, 3).copy())
        if len(samples) < frames:
            run(lambda: grab(i + 1), delayFrames=1)
            return
        path = os.path.join(out_dir, 'capture_%s.npz' % name)
        np.savez(path, landmarks=np.array(samples),
                 image_size=np.array([chop['image_width'][0], chop['image_height'][0]]),
                 face_present=float(chop['face_present'][0]),
                 settings=np.array([comp.par.Expressiongain.eval(), comp.par.Maxcoefficient.eval(),
                                    comp.par.Depthweight.eval()]),
                 baseline=comp.op('face_fit').fetch('baseline', np.zeros(0), search=False))
        print('Saved %d frames to %s' % (len(samples), path))

    grab(0)


_capture(globals().get('CAPTURE', 'capture'))
