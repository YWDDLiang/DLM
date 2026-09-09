"""Bind one independent worker before CUDA or model loaders can initialize."""
import os
from pathlib import Path
import runpy
import sys


def bind_device(environment):
    devices = [item for item in environment.get('CUDA_VISIBLE_DEVICES', '').split(',') if item]
    if not devices:
        raise ValueError('independent worker requires explicit allocated CUDA devices')
    rank = int(environment['LOCAL_RANK'])
    device = devices[rank % len(devices)]
    environment['CUDA_VISIBLE_DEVICES'] = device
    return device


if __name__ == '__main__':
    bind_device(os.environ)
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')
