from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py
import scipy.sparse

class COIL20(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'COIL20.mat')
        self.Y = data['gnd'].flatten().astype(np.int32) - 1
        v1 = data['fea'].astype(np.float32)
        # Construct view 2 with noise
        v2 = v1 + np.random.normal(0, 0.1, v1.shape).astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(v1)
        self.V2 = scaler.fit_transform(v2)

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class ALOI(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ALOI.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), 
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class OutdoorScene(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'OutdoorScene.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), 
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class ORL(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ORL.mat')
        self.Y = data['gt'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['fea'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['fea'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['fea'][0][2].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class EYaleB(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'EYaleB10_mtv.mat')
        self.Y = data['gt'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['fea'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['fea'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['fea'][0][2].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class Animal(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Animal.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Yale(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Yale.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "COIL20":
        dataset = COIL20('./data/')
        dims = [1024, 1024]
        view = 2
        data_size = 1440
        class_num = 20
    elif dataset == "ALOI":
        dataset = ALOI('./data/')
        dims = [77, 13, 64, 125]
        view = 4
        data_size = 10800
        class_num = 100
    elif dataset == "OutdoorScene":
        dataset = OutdoorScene('./data/')
        dims = [512, 432, 256, 48]
        view = 4
        data_size = 2688
        class_num = 8
    elif dataset == "Animal":
        dataset = Animal('./data/')
        dims = [2689, 2000, 2001, 2000]
        view = 4
        data_size = 11673
        class_num = 20
    elif dataset == "Yale":
        dataset = Yale('./data/')
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 165
        class_num = 15
    elif dataset == "ORL":
        dataset = ORL('./data/')
        # Dims: 4096, 3304, 6750
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 400
        class_num = 40
    elif dataset == "EYaleB":
        dataset = EYaleB('./data/')
        # Dims: 1024, 1239, 256
        dims = [1024, 1239, 256]
        view = 3
        data_size = 640
        class_num = 10
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num