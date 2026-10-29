# Auto Explore (installed by install_face_tracking.py): while it's on, press
# Random Face and/or Random Expression every Interval seconds.
#
# The last tick is kept in memory only (never saved with the .toe) and is
# cleared while Auto Explore is off, so switching it on always starts from the
# current time and fires right away -- in any project, after any restart.

_last = None


def onFrameStart(frame):
    global _last
    comp = parent()
    if not comp.par.Autoexplore.eval():
        _last = None
        return
    now = absTime.seconds
    if _last is not None and 0 <= now - _last < comp.par.Autointerval.eval():
        return
    _last = now
    what = comp.par.Autotarget.eval()
    if what in ('face', 'both'):
        comp.par.Randomface.pulse()
    if what in ('expression', 'both'):
        comp.par.Randomexpression.pulse()
