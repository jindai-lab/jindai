import os
import glob
from io import BytesIO
import numpy as np
import h5py
from threading import Lock
import config


class StorageManager:

    files = [h5py.File(g, 'r') for g in glob.glob(os.path.join(config.storage, '*.h5')) if not g.endswith('blocks.h5')]
    base = os.path.join(config.storage, 'blocks.h5')
    f = None
    write_counter = 0
    _lock = Lock()

    def __enter__(self, *args):
        StorageManager._lock.acquire()
        if not StorageManager.f or StorageManager.f.mode != 'r+':
            if StorageManager.f:
                StorageManager.f.close()
            StorageManager.f = h5py.File(StorageManager.base, 'r+')
        StorageManager.write_counter = 0
        return self

    def __exit__(self, *args):
        StorageManager.f.flush()
        StorageManager._lock.release()

    def write(self, src, iid):
        assert StorageManager.f, 'Pleae use `with` statement.'

        if isinstance(src, bytes):
            src = BytesIO(src)
        elif isinstance(src, str):
            src = open(src, 'rb')

        k = f'data/{iid}'
        if k in StorageManager.f:
            del StorageManager.f[k]
        StorageManager.f[k] = np.frombuffer(src.read(), dtype='uint8')

        return True

    def read(self, iid, check_only=False):
        if not os.path.exists(StorageManager.base):
            StorageManager.f = h5py.File(StorageManager.base, 'w')
            StorageManager.f.close()
            StorageManager.f = None

        if not StorageManager.f:
            StorageManager.f = h5py.File(StorageManager.base, 'r')

        k = f'data/{iid}'
        for f in [StorageManager.f] + StorageManager.files:
            if k in f:
                if check_only:
                    return True
                else:
                    return BytesIO(f[k][:].tobytes())
        if check_only:
            return False
        else:
            raise OSError(f"No matched ID found: {iid}")

    def exists(self, iid):
        return self.read(iid, True)

    def delete(self, iid):
        k = f'data/{iid}'
        for f in [StorageManager.f] + StorageManager.files:
            if k in f:
                del f[k]
