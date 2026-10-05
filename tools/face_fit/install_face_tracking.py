# Add face tracking to the GNM head component. Run from the Textport:
#
#   p='/Users/isaac/Documents/prototypes/human-control/gnm-touchdesigner/tools/face_fit/install_face_tracking.py'; exec(open(p).read(), dict(globals(), __file__=p))
#
# Targets /project1/google_gnm_model, or set GNM_COMP='/path/to/comp'; first.
# Safe to run again: it updates what's already there.
#
# What it does inside the component:
#   * embeds face_fit.npz (solver data) in the VFS
#   * adds a Script CHOP `face_fit` between `override` and `to_samples`
#   * replaces `xforms_callbacks` so tracked head/eye pose adds to the Pose page
#   * adds a Tracking page: Landmarks CHOP, Face Tracking, Expression Gain,
#     Max Coefficient, Tongue Follows Jaw, Depth Weight, Smoothing, Gaze Gain, Apply Identity, Calibrate Neutral Face, Clear Calibration
#   * adds a Parameter Execute DAT that handles the two pulses

import os

HERE = os.path.dirname(os.path.abspath(globals().get('__file__', '')))


def _read(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as f:
        return f.read()


def _ensure(comp, op_type, name, x, y):
    o = comp.op(name)
    if o is None:
        o = comp.create(op_type, name)
        o.nodeX, o.nodeY = x, y
    return o


def _page(comp, name):
    for p in comp.customPages:
        if p.name == name:
            return p
    return comp.appendCustomPage(name)


def _par(page, kind, name, label, **attrs):
    comp = page.owner
    existing = getattr(comp.par, name, None)
    if existing is None:
        pg = getattr(page, 'append' + kind)(name, label=label)
        p = pg[0]
        for k, v in attrs.items():
            setattr(p, k, v)
        if 'default' in attrs:
            p.val = attrs['default']
    return getattr(comp.par, name)


def install():
    comp = op(globals().get('GNM_COMP', '/project1/google_gnm_model'))
    if comp is None or comp.op('override') is None or comp.op('to_samples') is None:
        raise RuntimeError('GNM component not found -- set GNM_COMP to its path')

    # 1. solver data in the VFS
    npz = os.path.join(HERE, 'face_fit.npz')
    if not os.path.exists(npz):
        raise RuntimeError('face_fit.npz missing -- run build_face_fit.py first')
    if 'face_fit.npz' in [f.name for f in comp.vfs]:
        comp.vfs['face_fit.npz'].destroy()
    comp.vfs.addFile(npz)

    override = comp.op('override')
    x, y = override.nodeX, override.nodeY

    # 2. face_fit Script CHOP between override and to_samples
    cb = _ensure(comp, textDAT, 'face_fit_callbacks', x + 150, y - 150)
    cb.text = _read('face_fit_callbacks.py')
    fit = _ensure(comp, scriptCHOP, 'face_fit', x + 150, y)
    fit.par.callbacks = cb.name
    fit.inputConnectors[0].connect(override)
    comp.op('to_samples').inputConnectors[0].connect(fit)

    # 3. xforms reads identity and pose from face_fit
    comp.op('xforms_callbacks').text = _read('xforms_callbacks.py')

    # 4. Tracking page
    page = _page(comp, 'Tracking')
    _par(page, 'CHOP', 'Landmarkchop', 'Landmarks CHOP')
    _par(page, 'Toggle', 'Tracking', 'Face Tracking', default=True)
    _par(page, 'Float', 'Expressiongain', 'Expression Gain', default=1.5, normMin=0, normMax=4)
    _par(page, 'Float', 'Maxcoefficient', 'Max Coefficient', default=3.0, normMin=0.5, normMax=5,
         min=0, clampMin=True)
    _par(page, 'Toggle', 'Tonguefollowsjaw', 'Tongue Follows Jaw', default=True)
    _par(page, 'Float', 'Depthweight', 'Depth Weight', default=0.2, normMin=0, normMax=1,
         min=0, max=1, clampMin=True, clampMax=True)
    _par(page, 'Float', 'Smoothing', 'Smoothing', default=0.4, normMin=0, normMax=0.95,
         min=0, max=0.95, clampMin=True, clampMax=True)
    _par(page, 'Float', 'Gazegain', 'Gaze Gain', default=1.0, normMin=0, normMax=3)
    _par(page, 'Toggle', 'Applyidentity', 'Apply Identity', default=True)
    _par(page, 'Pulse', 'Calibrate', 'Calibrate Neutral Face')
    _par(page, 'Pulse', 'Clearcalibration', 'Clear Calibration')

    # 5. pulses -> face_fit storage
    pe = _ensure(comp, parameterexecuteDAT, 'tracking_parexec', x + 300, y - 150)
    pe.text = '''# Tracking page pulses (installed by install_face_tracking.py)

def onPulse(par):
    fit = op('face_fit')
    if par.name == 'Calibrate':
        fit.store('calibrate_request', True)
    elif par.name == 'Clearcalibration':
        fit.unstore('baseline')
        fit.unstore('identity')
    fit.cook(force=True)
    return
'''
    pe.par.op = '..'
    pe.par.pars = 'Calibrate Clearcalibration'
    pe.par.valuechange = False
    pe.par.onpulse = True

    fit.cook(force=True)
    errs = fit.errors() or comp.op('xforms').errors()
    print('Face tracking installed in', comp.path)
    if errs:
        print('  errors:', errs)
    print('  Next: on the Tracking page, set Landmarks CHOP to an Info CHOP on your')
    print('  Apple Face Landmarks TOP, hold a neutral face, press Calibrate Neutral Face.')


install()
