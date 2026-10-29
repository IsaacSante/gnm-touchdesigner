# Face Tracking and Explore buttons (installed by install_face_tracking.py).
# Random Face picks a new Seed; changing Seed rolls the identity sliders from
# it, so a seed always gives the same face (0 = the average face).

import numpy as np


def _sliders(kind):
    """The component's own I* (identity) or E* (expression) sliders."""
    return [getattr(parent().par, c.val) for c in op('par_map').col(0) if c.val[0] == kind]


def _roll(sliders, rng):
    variety = float(parent().par.Variety.eval())
    for p in sliders:
        p.val = float(np.clip(rng.normal(0.0, variety), -3.0, 3.0))


def onValueChange(par, prev):
    if par.name == 'Seed':
        seed = int(par.eval())
        if seed:
            _roll(_sliders('I'), np.random.default_rng(seed))
        else:
            for p in _sliders('I'):
                p.val = 0.0
    return


def onPulse(par):
    comp = parent()
    fit = op('face_fit')
    if par.name == 'Randomface':
        comp.par.Seed = int(np.random.randint(1, 100000))
    elif par.name == 'Randomexpression':
        # tongue and pupil sliders look broken when random; leave them
        _roll([p for p in _sliders('E') if not p.name.startswith(('Etongue', 'Epupils'))],
              np.random.default_rng())
    elif par.name == 'Reset':
        comp.par.Seed = 0
        for p in _sliders('I') + _sliders('E'):
            p.val = 0.0
    elif par.name == 'Calibrate':
        fit.store('calibrate_request', True)
        fit.cook(force=True)
    elif par.name == 'Clearcalibration':
        fit.unstore('baseline')
        fit.unstore('identity')
        fit.unstore('baseline_shapes')
        fit.cook(force=True)
    return
