"""
48 x 1876 -> 39 x 48 x 48
마지막 조각은 삭제 처리
""" 

import os
import os.path as p

import numpy as np
from tqdm import tqdm


def split_dataset(root_path, target_path):
    assert p.exists(target_path)

    file_paths = []
    for (root_dir, dirs, files) in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root_dir, file)
            file_paths.append(file_path)

    for file_path in tqdm(file_paths):
        orig = np.load(file_path)
        time_dim, freq_dim = orig.shape
        assert time_dim == 48

        # 조각 수
        n_piece = freq_dim // 48
        file_name = p.splitext(p.basename(file_path))[0]
        dir_name = str(int(file_name) // 1000)
        target_dir = p.join(target_path, dir_name)
        os.makedirs(target_dir, exist_ok=True)

        for i in range(n_piece):
            np.save(
                p.join(target_dir, f"{file_name}_{i}.npy"),
                orig[:, 48 * i:48 * (i + 1)]
            )


if __name__ == "__main__":
    DEBUG_DATA_PATH = '/home/jhkim/workspace/sample_data/arena_mel'
    DATA_PATH = '/data1/melon/arena_mel'
    DEBUG_TARGET_PATH = '/home/jhkim/workspace/sample_data/splitted_sample'
    TARGET_PATH = '/data1/melon/splitted_arena_mel'

    # split_dataset(DEBUG_DATA_PATH, DEBUG_TARGET_PATH)
    split_dataset(DATA_PATH, TARGET_PATH)
