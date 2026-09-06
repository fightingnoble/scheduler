import json

import h5py
import numpy as np

from utils import check_parents_path

# 定义分块大小
CHUNK_SIZE = 25
def get_next_chunk_id(f):
    try:
        chunk_ids = [int(name.split('_')[1]) for name in f.keys()]
        if chunk_ids:
            return max(chunk_ids) + 1
        else:
            return 0
    except (OSError, KeyError):
        return 0

def save_chunk(file_name, data: list, force:bool=False):
    check_parents_path(file_name)
    if len(data) < CHUNK_SIZE and not force:
        return
    if len(data) > 0:
        with h5py.File(file_name, 'a') as f:
            chunk_id = get_next_chunk_id(f)
            dataset_name = f'chunk_{chunk_id}'
            data_as_json = np.array([json.dumps(item) for item in data], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset(dataset_name, data=data_as_json, compression="gzip")
        data.clear()
    if force:
        try:
            load_h5_file(file_name)
            print(f"{file_name} saved and loaded successfully")
        except:
            print(f"{file_name} bad")
            exit()

def load_h5_file(file_name):
    data = []
    with h5py.File(file_name, 'r') as f:
        for name in f.keys():
            # 只有使用 [:]，你才能获得 h5py.Dataset 对象中的实际数据内容
            data_as_json = f[name][:]
            dataset = [json.loads(item) for item in data_as_json]
            data.extend(dataset)
    return data


import os
from global_var import log_dir

def get_log_path_str(args):
    return os.path.join(log_dir, args.root_dir)

# ---------------------------------------------------------------------------
# New: path consistency helpers (to compare old vs new path building results)
# ---------------------------------------------------------------------------

def _normalize_path(p):
    if p is None:
        return None
    return os.path.normpath(p)

def check_paths_equal(old_path, new_path, label: str = ""):
    """
    Compare two paths (after normpath). If mismatch, print a clear diagnostic line.
    This is a soft check used during migration. Return bool.
    """
    op = _normalize_path(old_path)
    np_ = _normalize_path(new_path)
    eq = (op == np_)
    if not eq:
        print(f"[PathCheck] {label} mismatch:\n  old: {op}\n  new: {np_}")
    return eq
