# Thread-local stash for (fpn_feats, batch_metas)
# VFE can read from here during its forward. If not set, VFE does nothing special.

import threading
_TLS = threading.local()

def set_painting_context(fpn_feats, batch_metas):
    _TLS.fpn_feats = fpn_feats
    _TLS.batch_metas = batch_metas

def get_painting_context():
    fpn = getattr(_TLS, 'fpn_feats', None)
    metas = getattr(_TLS, 'batch_metas', None)
    return fpn, metas

def clear_painting_context():
    for k in ('fpn_feats', 'batch_metas'):
        if hasattr(_TLS, k):
            delattr(_TLS, k)