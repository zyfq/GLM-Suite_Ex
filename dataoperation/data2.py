
"""
完成从csv文件以numpy的数组格式读取数据，做好数据划分，数据清洗，再以迭代器的方式保存

"""
import  pathlib as Path
import numpy as np
import os

class DataSet:
    def __init__(self, csv_path,train_ratio,val_ratio,test_ratio,seed,skip_header = 1,delimiter = ','):
        ratios = (train_ratio,val_ratio,test_ratio)
        if any(ratio < 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError('分割比例不能小于0，而且总和要为1')
        self.csv_path = csv_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.delimiter = delimiter
        self.skip_header = skip_header

        self.seed = seed
        self.index = 0
        self.data_dct_all = self.spilt_data()
    def read_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"文件{self.csv_path}不存在")
        cf = np.genfromtxt(self.csv_path,float,delimiter = self.delimiter,skip_header = self.skip_header)
        if cf.shape[0] == 0:
            raise ValueError(f"文件{self.csv_path}为空")
        if cf.shape[1] < 2:
            raise ValueError(f"文件{self.csv_path}列数小于2")
        
        return cf

    def _get_data_ratio(self,original_data_length):
        indices = np.arange(original_data_length)
        np.random.default_rng(self.seed).shuffle(indices)
        train_length = int(original_data_length * self.train_ratio)
        val_length = int(original_data_length * (self.train_ratio + self.val_ratio))
        return indices[:train_length] , indices[train_length:val_length] , indices[val_length:]


    def data_StandardScaler(self,train_data_feature,val_data_feature,test_data_feature):
        train_mean = np.mean(train_data_feature,axis=0)
        train_std = np.std(train_data_feature,axis=0)

        train_data_StandardScaler = (train_data_feature - train_mean) / train_std
        val_data_StandardScaler =  (val_data_feature - train_mean) / train_std
        test_data_StandardScaler =  (test_data_feature - train_mean) / train_std
        return train_data_StandardScaler,val_data_StandardScaler,test_data_StandardScaler

    def spilt_data(self):
        """
        接下来我需要随机分配 按比例到3个数组,这样子我就需要重新写个方法实现这个功能
        """
        cf = self.read_csv()
        train_indice, val_indice, test_indice = self._get_data_ratio(len(cf))
        train_data_sample = cf[train_indice]
        val_data_sample = cf[val_indice]
        test_data_sample = cf[test_indice]
        train_data_feature = train_data_sample[:,:-1]
        train_data_label = train_data_sample[:,-1]
        val_data_feature = val_data_sample[:, :-1]
        val_data_label = val_data_sample[:, -1]
        test_data_feature = test_data_sample[:, :-1]
        test_data_label = test_data_sample[:, -1]
        train_data_StandardScaler,val_data_StandardScaler,test_data_StandardScaler = self.data_StandardScaler(train_data_feature,val_data_feature,test_data_feature)
        data_dct = {"train_data_StandardScaler":train_data_StandardScaler,"val_data_StandardScaler":val_data_StandardScaler, "test_data_StandardScaler":test_data_StandardScaler,"train_data_feature": train_data_feature,
         "val_data_feature":val_data_feature,"test_data_feature": test_data_feature, "train_data_label":train_data_label, "val_data_label":val_data_label,"test_data_label": test_data_label}
        return data_dct

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        for k,v in self.data_dct_all.items():
            if self.index >= len(v):
                raise  StopIteration
            self.index += 1
            return v[self.index]

if __name__ == '__main__':
    d = DataSet('../data/Social_Network_Ads.csv',0.6,0.2,0.2,40)
    d.spilt_data()