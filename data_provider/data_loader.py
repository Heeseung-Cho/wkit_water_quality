import os
import json
from glob import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

def get_xdata(root_path: str, data_path: str) :
    """_summary_
    Args:
        root_path (str): _description_
        data_path (str): _description_

    Returns:
        _type_: _description_
    """
    data_list = sorted(glob(os.path.join(root_path, data_path, "csv/*.csv")))
    df_raw = pd.DataFrame()
    for file in data_list:
        df_ = pd.read_csv(file)
        if len(df_) == 120:
            df_raw = pd.concat([df_raw, df_], axis = 0, ignore_index= True)
        
    return df_raw

def get_ydata(root_path: str, data_path: str, ):
    """_summary_

    Args:
        root_path (str): _description_
        data_path (str): _description_

    Returns:
        list, list, pd.DataFrame: _description_
    """
    label_list = sorted(glob(os.path.join(root_path, data_path, "hw3_json/*.json")))
    target_cols = ['turbidity', 'EC', 'pH', 'water_temp',
        'DO', 'TOC', 'algae', 'alkalinity', 'blue_algae', 'residual_Cl',
        'blue-green_algae', 'diatomeae', 'cryptophyceae', '2-MIB', 'Geosmin',
        'synedra', 'T-N', 'T-P', 'Mn',]
    target_vals = []
    for js in label_list:
        with open(js, 'r') as f:
            label = json.load(f)
        input_cols = label['h']['input_serial'].split(',')   
        target_cols = [k for k,v in label['w'].items() if (v is not None) & (k in target_cols)]
        vals = [label['w'][k] for k in target_cols]        
        vals = [v.replace(',','') if str(v).__contains__(',') else v for v in vals ]
        target_vals.append(vals)
    target_vals = np.array(target_vals)    
    return input_cols, target_cols, pd.DataFrame(target_vals, columns = target_cols, dtype=float)

class Water_Dataset(Dataset):
    def __init__(self, root_path, data_path, 
                flag='train', size=None, 
                target='diatomeae', scale=True):                
        if size == None:
            self.seq_len = 24 * 5 
        else:
            self.seq_len = size
        self.label_len = 1
        self.pred_len = 1
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read__data__()
        
    def __read__data__(self):
        self.scaler = StandardScaler()
        self.in_cols, self.target_cols, df_target = get_ydata(self.root_path, self.data_path)
        df_raw = get_xdata(self.root_path, self.data_path)
        length = int(min(len(df_raw)/120, len(df_target)))
        border1s = [0, length - 25, length - 10]
        border2s = [length - 25, length - 10, length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        x_data = df_raw[df_raw.columns.intersection(self.in_cols)]
        y_data = df_target[self.target].values

        if self.scale:
            train_data = x_data[border1s[self.set_type]*self.seq_len:border2s[self.set_type]*self.seq_len]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(x_data.values)
        else:
            data = x_data.values

        self.x_data = data[border1*self.seq_len:border2*self.seq_len]
        self.y_data = y_data[border1:border2]        
                
    def __len__(self):
        return len(self.y_data)
    
    def __getitem__(self, index):
        s_begin = index * self.seq_len
        s_end = s_begin + self.seq_len
        seq_x = self.x_data[s_begin:s_end]
        seq_y = self.y_data[index]
        return seq_x, seq_y
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)