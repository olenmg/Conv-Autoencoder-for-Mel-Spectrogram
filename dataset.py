import os
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class MelSpectDataset(Dataset):
    """ Dataset class for unsplitted data ... (48, 1876)
    현재 데이터 구성: 99% 이상이 (48, 1876), shape이 다른 극소수의 데이터가 존재함
    해당 데이터들은 모두 크기가 제각각이므로 구현의 편의성을 위해 일단 무시하도록 함
    
    (48, 1876)을 (48, 1872)로 취급하여 하나의 데이터 파일을 
    39개의 (48, 48) 조각으로 분할함 (1872 = 48 * 39)

    데이터 로딩 과정에서 성능 개선할 부분이 분명 있을 것 같은데 혹시 알고있는게 있으시면 말씀 부탁드립니다 !!
    - 현재 데이터셋 로딩이 실행시간에서 큰 비중을 차지
    """
    def __init__(self, root_path, debug):
        self.file_paths = []
        self.get_file_paths(root_path)

        pkl_name = f"valid_idx_{'DEBUG' if debug else 'NO-DEBUG'}.pkl"
        if not os.path.exists(pkl_name):
            valid_idx = []
            for i, file_path in tqdm(enumerate(self.file_paths)):
                data = np.load(file_path)
                if data.shape == (48, 1876): # ignore abnormal data
                    valid_idx.append(i)
            with open(pkl_name, "wb") as f:
                pickle.dump(valid_idx, f)
        
        with open(pkl_name, "rb") as f:
            self.valid_idx = pickle.load(f)

    def __len__(self):
        return len(self.valid_idx) * 39

    def __getitem__(self, idx):
        #TODO: caching .. 데이터 캐싱으로 속도 향상 기대 가능?
        path_idx = self.valid_idx[idx // 39]
        seq_idx = idx % 39
        
        data = np.load(self.file_paths[path_idx])[:, 48 * seq_idx:48 * (seq_idx + 1)]

        return torch.from_numpy(data).unsqueeze(0) # shape: (N, 1, 48, 48)

    def get_file_paths(self, root_path):
        """ Get data file paths """
        for (root_dir, _, files) in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root_dir, file)
                self.file_paths.append(file_path)



class SplittedMelSpectDataset(Dataset):
    """ Dataset for class splitted data ... (48, 48)
    """
    pass