
import json, math
import numpy as np

_rig = None

def _get_rig():
    global _rig
    if _rig is None:
        _rig = json.loads(bytes(parent().vfs['rig.json'].byteArray).decode('utf-8'))
        _rig['tpos'] = np.array(_rig['template_joint_positions'])
        _rig['orient'] = np.array(_rig['joint_orientations'])
        _rig['jib'] = np.array(_rig['joint_identity_basis'])
    return _rig

def _rot(rx, ry, rz):
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def onSetupParameters(scriptOp): return
def onPulse(par): return

def onCook(scriptOp):
    rig = _get_rig()
    comp = parent()

    # face_fit = sliders + tracked expression/identity + pose channels
    coeffs = op('face_fit')

    # joint rest positions shifted by current identity
    idvals = np.zeros(len(rig['identity_channels']))
    for i, nm in enumerate(rig['identity_channels']):
        ch = coeffs.chan(nm)
        if ch is not None:
            idvals[i] = ch[0]
    jpos = rig['tpos'] + np.tensordot(idvals, rig['jib'], axes=1)

    def par3(prefix):
        return [float(getattr(comp.par, prefix + a)) for a in 'xyz']

    def tracked(prefix):
        # Pose-page sliders act as offsets on top of face tracking.
        out = []
        for a in ('rx', 'ry', 'rz'):
            ch = coeffs.chan(prefix + a)
            out.append(ch[0] if ch is not None else 0.0)
        return out

    def add(a, b):
        return [x + y for x, y in zip(a, b)]

    rots = {0: par3('Neckr'),
            1: add(par3('Headr'), tracked('pose_head_')),
            2: add(par3('Lefteyer'), tracked('pose_lefteye_')),
            3: add(par3('Righteyer'), tracked('pose_righteye_'))}
    t = np.array(par3('Headt'))

    world = [None] * 4
    for j in range(4):
        O = rig['orient'][j]
        R = O @ _rot(*rots[j]) @ O.T
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = jpos[j] - R @ jpos[j]     # rotate about the joint pivot
        parent_idx = rig['joint_parent_indices'][j]
        if parent_idx < 0:
            G = np.eye(4); G[:3, 3] = t      # global translate on the root
            world[j] = G @ M
        else:
            world[j] = world[parent_idx] @ M

    vals = []
    for j in range(4):
        vals += list(world[j][:3, :4].flatten())

    scriptOp.clear()
    scriptOp.numSamples = len(vals)
    c = scriptOp.appendChan('m')
    c.vals = vals
