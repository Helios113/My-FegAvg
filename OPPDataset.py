import torch
from pathlib import Path
import pandas as pd
import numpy as np

def OPPDataset(dir: str, split: float):
        data_dir = Path(dir)
        
        data = torch.Tensor(pd.read_csv(data_dir,delimiter=",", header=None).values)
        data_len = data.shape[0]
        labels = data[:,-1]
        data = data[:,:-1]

        # labels = torch.from_numpy(labels).float()
        data = torch.nn.functional.normalize(data, dim = 1)
        data = data.reshape(data_len,-1,102)
        sampling = range(data_len)
        data_idx = np.random.choice(sampling,size = int(data_len*split), replace=False).tolist()
        sampling = list(set(sampling) - set(data_idx))
        test_idx = sampling

        return (data[data_idx],labels[data_idx]), (data[test_idx],labels[test_idx])


