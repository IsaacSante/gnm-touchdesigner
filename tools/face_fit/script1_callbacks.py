import numpy, io

_basis = None


def _vfs_array(name, key=None):
    raw = io.BytesIO(bytes(parent().vfs[name].byteArray))
    return numpy.load(raw)[key] if key else numpy.load(raw)


def _get_basis():
    global _basis
    if _basis is None:
        _basis = _vfs_array('basis_half.npz', 'basis').astype(numpy.float32)
        # anatomical region and sub-region weight per vertex (region_masks.py)
        # ride in the unused .w of rows 0..4 and 5..9 for glsl1's Colors
        _basis[0:10, :, 3] = _vfs_array('masks.npy').reshape(10, 4096)
    return _basis


def onSetupParameters(scriptOp): return
def onPulse(par): return


def onCook(scriptOp):
    scriptOp.copyNumpyArray(_get_basis())
