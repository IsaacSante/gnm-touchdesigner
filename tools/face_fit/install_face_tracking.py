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
#   * adds a Face Tracking page: MediaPipe Face Tracking, Face Tracking, Expression Gain,
#     Max Coefficient, Tongue Follows Jaw, Depth Weight, Smoothing, Gaze Gain,
#     Apply Identity, Calibrate Neutral Face, Clear Calibration
#   * adds an Emotion Colors page: Emotion Recolor (off), Emotion Gain, one color
#     per emotion
#   * adds a Skin Colors page: Enable Colors (on), skin, lips, eyes, teeth, gums,
#     tongue; embeds masks.npy (region_masks.py) and replaces script1's
#     callbacks so the regions reach glsl1
#   * replaces glsl1's shader (glsl1_compute.glsl): writes Color from both pages
#   * adds outputs `emotion_out` (CHOP: neutral happy surprise angry sad fear
#     dominant_id) and `emotion_label` (DAT: the dominant emotion's name)
#   * limits `pars_raw` to the morph sliders
#   * adds an Explore page (first): Random Face, Random Expression, Reset Face,
#     Variety, Seed, Face Strength, Expression Strength, Auto Explore
#     (explore_timer.py: new face/expression every Interval seconds)
#   * adds a Parameter Execute DAT for the buttons (parexec_callbacks.py)

import os
import tempfile

import numpy as np

# Emotion Recolor defaults, in face_fit's EMOTIONS order
EMOTION_COLORS = [('neutral', (0.8, 0.8, 0.8)), ('happy', (1.0, 0.78, 0.1)),
                  ('surprise', (0.1, 0.85, 0.9)), ('angry', (0.9, 0.08, 0.08)),
                  ('sad', (0.15, 0.3, 0.95)), ('fear', (0.55, 0.15, 0.85))]

# Skin Colors page defaults
REGION_COLORS = [('skin', (0.87, 0.67, 0.56)), ('lip', (0.72, 0.36, 0.38)),
                 ('iris', (0.30, 0.42, 0.55)), ('pupil', (0.03, 0.03, 0.03)),
                 ('sclera', (0.93, 0.91, 0.88)), ('teeth', (0.93, 0.90, 0.82)),
                 ('gums', (0.75, 0.38, 0.40)), ('tongue', (0.78, 0.40, 0.42))]

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


def _page(comp, name, old=None):
    """The custom page called name; renames one called old (an earlier install)."""
    pages = {p.name: p for p in comp.customPages}
    if name not in pages and old in pages:
        pages[old].name = name
        return pages[old]
    return pages.get(name) or comp.appendCustomPage(name)


def _par(page, kind, name, label, **attrs):
    """Create the parameter if missing. Its settings (default, ranges) are
    refreshed every run, so an upgraded component gets the current defaults;
    its value is only set when it's created."""
    comp = page.owner
    p = getattr(comp.par, name, None)
    created = p is None
    if created:
        p = getattr(page, 'append' + kind)(name, label=label)[0]
    for k, v in attrs.items():
        setattr(p, k, v)
    if created and 'default' in attrs:
        p.val = attrs['default']
    return p


def _vfs_add(comp, path):
    name = os.path.basename(path)
    if name in [f.name for f in comp.vfs]:
        comp.vfs[name].destroy()
    comp.vfs.addFile(path)


def _rgb(page, name, label, rgb):
    """An RGB parameter (name + r/g/b); defaults refreshed, value set if new."""
    created = not hasattr(page.owner.par, name + 'r')
    pars = page.appendRGB(name, label=label) if created else \
        [getattr(page.owner.par, name + c) for c in 'rgb']
    for p, v in zip(pars, rgb):
        p.default = v
        if created:
            p.val = v


def _region_masks(comp):
    ns = {}
    exec(_read('region_masks.py'), ns)
    return ns['build'](bytes(comp.vfs['head_template.obj'].byteArray))


def install():
    comp = op(globals().get('GNM_COMP', '/project1/google_gnm_model'))
    if comp is None or comp.op('override') is None or comp.op('to_samples') is None:
        raise RuntimeError('GNM component not found -- set GNM_COMP to its path')

    # 1. solver data in the VFS
    npz = os.path.join(HERE, 'face_fit.npz')
    if not os.path.exists(npz):
        raise RuntimeError('face_fit.npz missing -- run build_face_fit.py first')
    _vfs_add(comp, npz)

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

    # 4. Face Tracking page
    page = _page(comp, 'Face Tracking', old='Tracking')
    if hasattr(comp.par, 'Landmarkchop'):            # Apple input, before v1.1
        comp.par.Landmarkchop.destroy()
    _par(page, 'COMP', 'Facetracking', 'MediaPipe Face Tracking')
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

    # 5. Emotion Colors page; glsl1 recolors from it
    page = _page(comp, 'Emotion Colors', old='Emotion')
    if hasattr(comp.par, 'Emotiongain') and comp.par.Emotiongain.page.name != page.name:
        comp.par.Emotiongain.destroy()                # was on Tracking before v1.1
    _par(page, 'Toggle', 'Emotionrecolor', 'Emotion Recolor', default=False)
    _par(page, 'Float', 'Emotiongain', 'Emotion Gain', default=1.5, normMin=0, normMax=4)
    glsl = comp.op('glsl1')
    glsl.par.computedat.eval().text = _read('glsl1_compute.glsl')
    # Output Attributes is a sequence; v1.0.2 has one block (P)
    if glsl.seq.attr.numBlocks < 2:
        glsl.seq.attr.numBlocks = 2
    glsl.par.attr1name = 'color'                      # menu token; writes Color
    for name, rgb in EMOTION_COLORS:
        _rgb(page, name.capitalize() + 'color', name.capitalize(), rgb)

    # 6. Skin Colors page; anatomical regions computed once and embedded
    page = _page(comp, 'Skin Colors', old='Colors')
    _par(page, 'Toggle', 'Enablecolors', 'Enable Colors', default=True)
    for name, rgb in REGION_COLORS:
        _rgb(page, name.capitalize() + 'color', name.capitalize(), rgb)
    _par(page, 'Float', 'Lipamount', 'Lip Amount', default=1.0, min=0, max=1,
         clampMin=True, clampMax=True)
    _par(page, 'Float', 'Lipfeather', 'Lip Feather', default=0.2, min=0, max=1,
         clampMin=True, clampMax=True)
    region, sub = _region_masks(comp)
    masks = np.zeros((2, 5 * 4096), dtype=np.float32)
    masks[0, :len(region)], masks[1, :len(sub)] = region, sub
    path = os.path.join(tempfile.mkdtemp(), 'masks.npy')
    np.save(path, masks)
    _vfs_add(comp, path)
    script1 = comp.op('script1')
    script1.par.callbacks.eval().text = _read('script1_callbacks.py')
    script1.cook(force=True)

    # glsl1 uniforms: one float per page setting, every color by name
    floats = [('uRecolor', 'Emotionrecolor'), ('uColors', 'Enablecolors'),
              ('uLipAmount', 'Lipamount'), ('uLipFeather', 'Lipfeather')]
    if glsl.seq.vec.numBlocks < len(floats):
        glsl.seq.vec.numBlocks = len(floats)
    for i, (uniform, par) in enumerate(floats):
        setattr(glsl.par, 'vec%dname' % i, uniform)
        getattr(glsl.par, 'vec%dvaluex' % i).expr = 'float(parent().par.%s)' % par
    names = [n for n, _ in EMOTION_COLORS + REGION_COLORS]
    if glsl.seq.color.numBlocks < len(names):
        glsl.seq.color.numBlocks = len(names)
    for i, name in enumerate(names):
        setattr(glsl.par, 'color%dname' % i, 'u' + name.capitalize())
        for c in 'rgb':
            getattr(glsl.par, 'color%drgb%s' % (i, c)).expr = \
                'parent().par.%scolor%s' % (name.capitalize(), c)

    # 7. emotion_out: the scores as the component's second output
    sel = _ensure(comp, selectCHOP, 'emotion', x + 300, y - 300)
    sel.par.chops = fit.name
    sel.par.channames = ' '.join(n for n, _ in EMOTION_COLORS) + ' dominant_id'
    out = _ensure(comp, outCHOP, 'emotion_out', x + 450, y - 300)
    out.inputConnectors[0].connect(sel)
    # outputs are ordered top to bottom: keep the geometry first
    out.nodeY = min(out.nodeY, comp.op('pop_geo_out').nodeY - 200)

    # 8. emotion_label: the dominant emotion's name, as text
    lcb = _ensure(comp, textDAT, 'emotion_label_callbacks', x + 300, y - 600)
    lcb.text = '''# emotion_label -- the dominant emotion's name (installed by install_face_tracking.py)

def onCook(scriptOp):
    names = op('face_fit_callbacks').module.EMOTIONS
    scriptOp.text = names[int(op('face_fit')['dominant_id'][0])]
'''
    lbl = _ensure(comp, scriptDAT, 'emotion_label', x + 300, y - 450)
    lbl.par.callbacks = lcb.name
    lout = _ensure(comp, outDAT, 'emotion_label_out', x + 450, y - 450)
    lout.inputConnectors[0].connect(lbl)
    lout.nodeY = min(lout.nodeY, out.nodeY - 150)

    # 9. pars_raw reads only the morph sliders, never Tracking or Emotion
    comp.op('pars_raw').par.parameters = ' '.join(r[0].val for r in comp.op('par_map').rows())

    # 10. Explore page: quick ways to try faces
    page = _page(comp, 'Explore')
    _par(page, 'Pulse', 'Randomface', 'Random Face')
    _par(page, 'Pulse', 'Randomexpression', 'Random Expression')
    _par(page, 'Pulse', 'Reset', 'Reset Face')
    _par(page, 'Float', 'Variety', 'Variety', default=1.0, min=0, max=3, clampMin=True, clampMax=True)
    _par(page, 'Int', 'Seed', 'Seed', default=0, min=0, normMax=99999, clampMin=True)
    _par(page, 'Float', 'Facestrength', 'Face Strength', default=1.0, min=0, max=2,
         clampMin=True, clampMax=True)
    _par(page, 'Float', 'Expressionstrength', 'Expression Strength', default=1.0, min=0, max=2,
         clampMin=True, clampMax=True)
    _par(page, 'Toggle', 'Autoexplore', 'Auto Explore', default=False)
    _par(page, 'Float', 'Autointerval', 'Interval (sec)', default=3.0, min=0.1, normMax=30,
         clampMin=True)
    _par(page, 'Menu', 'Autotarget', 'Randomize', menuNames=['face', 'expression', 'both'],
         menuLabels=['Face', 'Expression', 'Face and Expression'], default='both')
    timer = _ensure(comp, executeDAT, 'explore_timer', x + 450, y - 150)
    timer.text = _read('explore_timer.py')
    timer.unstore('*')                                # an earlier timer saved state here
    timer.par.framestart = True

    # 11. Face Tracking and Explore buttons
    pe = _ensure(comp, parameterexecuteDAT, 'tracking_parexec', x + 300, y - 150)
    pe.text = _read('parexec_callbacks.py')
    pe.par.op = '..'
    pe.par.pars = 'Calibrate Clearcalibration Randomface Randomexpression Reset Seed'
    pe.par.valuechange = True
    pe.par.onpulse = True

    # the new pages first, then the model's own (Default, Identity, ...)
    first = ['Explore', 'Face Tracking', 'Skin Colors', 'Emotion Colors']
    comp.sortCustomPages(*first, *[p.name for p in comp.customPages if p.name not in first])

    fit.cook(force=True)
    glsl.cook(force=True)
    errs = fit.errors() or comp.op('xforms').errors() or glsl.errors()
    print('Face tracking installed in', comp.path)
    if errs:
        print('  errors:', errs)
    print('  Next: on the Face Tracking page, set MediaPipe Face Tracking to the face_tracking')
    print('  COMP from Torin Blankensmith\'s MediaPipe plugin, hold a neutral face, and')
    print('  press Calibrate Neutral Face.')


install()
