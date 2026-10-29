# Make the GNM component ready to save as the release .tox. Run from the
# Textport after install_face_tracking.py:
#
#   p='/Users/isaac/Documents/prototypes/human-control/gnm-touchdesigner/tools/prepare_release.py'; exec(open(p).read(), dict(globals(), __file__=p))
#
# Targets /project1/google_gnm_model, or set GNM_COMP='/path/to/comp'; first.
# Clears everything that belongs to whoever tested it (calibration, fitted
# identity, the MediaPipe link, random faces, slider values), removes leftovers,
# and checks for problems. Then save the component: right-click > Save Component.

LEFTOVER_OPS = ['text1', 'text2', 'text3', 'glsl1_info']
LEFTOVER_PARS = ['Colorneutral', 'Colorhappy', 'Colorsurprise', 'Colorangry', 'Colorsad', 'Colorfear',
                 'Valuea', 'Valueb']


def prepare():
    comp = op(globals().get('GNM_COMP', '/project1/google_gnm_model'))
    if comp is None or comp.op('face_fit') is None:
        raise RuntimeError('installed GNM component not found -- set GNM_COMP')
    done = []

    # 1. state saved with the component
    for o in (comp.op('face_fit'), comp.op('explore_timer')):
        if o is not None and o.storage:
            done.append('cleared storage on %s: %s' % (o.name, ', '.join(o.storage)))
            o.unstore('*')

    # 2. parameters back to their defaults: sliders, Explore, Face Tracking link
    plain = [p for p in comp.customPars if p.mode.name == 'CONSTANT' and not p.isPulse]
    changed = [p.name for p in plain if p.val != p.default]
    for p in plain:
        p.val = p.default
    if changed:
        done.append('reset %d parameters: %s' % (len(changed), ', '.join(changed[:12]) +
                                                  (' ...' if len(changed) > 12 else '')))

    # 3. leftovers from development
    for name in LEFTOVER_OPS:
        o = comp.op(name)
        if o is not None:
            o.destroy()
            done.append('removed ' + name)
    callbacks = {o.par.callbacks.eval().name for o in comp.children
                 if hasattr(o.par, 'callbacks') and o.par.callbacks.eval() is not None}
    for o in list(comp.children):
        # the default callback DATs TD docks to Script CHOPs, when nothing uses them
        if o.OPType == 'textDAT' and o.dock is not None and o.name not in callbacks \
                and o.name.endswith(('_callbacks1', 'pars_named_callbacks')):
            o.destroy()
            done.append('removed unused ' + o.name)
    for o in [comp] + list(comp.children):
        for name in LEFTOVER_PARS:
            # an RGB parameter is three (r, g, b) that go together
            for p in [getattr(o.par, name + s, None) for s in ('', 'r')]:
                if p is not None and p.isCustom:
                    done.append('removed parameter %s.%s' % (o.name, name))
                    p.destroy()
    for o in list(comp.children):
        for page in list(o.customPages):
            if not page.pars:
                done.append('removed empty page %s on %s' % (page.name, o.name))
                page.destroy()

    # 4. checks
    problems = []
    for o in comp.findChildren():
        if o.family == 'DAT' and '/project1' in (o.text or ''):
            problems.append('%s mentions /project1' % o.path)
        for p in o.pars():
            if p.mode.name == 'EXPRESSION' and '/project1' in p.expr:
                problems.append('%s.%s expression mentions /project1' % (o.path, p.name))
        if o.errors():
            problems.append('%s error: %s' % (o.path, o.errors()))
    pages = [pg.name for pg in comp.customPages]
    outs = [o.name for o in sorted(comp.children, key=lambda o: -o.nodeY)
            if o.OPType in ('outPOP', 'outCHOP', 'outDAT')]

    print('Prepared', comp.path)
    for line in done:
        print('  -', line)
    print('  pages  :', ' | '.join(pages))
    print('  outputs:', ', '.join(outs))
    print('  VFS    :', ', '.join(f.name for f in comp.vfs))
    if problems:
        print('  PROBLEMS -- fix before saving:')
        for line in problems:
            print('   !', line)
    else:
        print('  no problems found. Now right-click the component > Save Component .tox')


prepare()
