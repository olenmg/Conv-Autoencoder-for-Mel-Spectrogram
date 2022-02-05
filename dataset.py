import os

import torch
import numpy as np
from torch.utils.data import Dataset


class MelSpectDataset(Dataset):
    """
    현재 데이터 구성: 99% 이상이 (48, 1876), shape이 다른 극소수의 데이터가 존재함
    해당 데이터들은 모두 크기가 제각각이므로 구현의 편의성을 위해 일단 무시하도록 함
    
    (48, 1876)을 (48, 1872)로 취급하여 하나의 데이터 파일을 
    39개의 (48, 48) 조각으로 분할함 (1872 = 48 * 39)

    데이터 로딩 과정에서 성능 개선할 부분이 분명 있을 것 같은데 혹시 알고있는게 있으시면 말씀 부탁드립니다 !!
    """
    def __init__(self, root_path):
        self.data_paths = self.get_data_paths(root_path)
        self.valid_idx = []

        for i, data_path in enumerate(self.data_paths):
            data = np.load(data_path)
            if data.shape == (48, 1876): # ignore abnormal data
                self.valid_idx.append(i)

    def __len__(self):
        return len(self.valid_idx) * 39

    def __getitem__(self, idx):
        #TODO: caching .. 데이터 캐싱으로 속도 향상 기대 가능?
        path_idx = self.valid_idx[idx // 39]
        seq_idx = idx % 39
        
        data = np.load(self.data_paths[path_idx])[:, 48 * seq_idx:48 * (seq_idx + 1)]

        return torch.from_numpy(data)

    def get_data_paths(self, root_path):
        """ Get data file paths """
        for (root_dir, _, files) in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root_dir, file)
                self.data_paths.append(file_path)