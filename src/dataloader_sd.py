import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
from torch.utils.data import DataLoader
np.random.seed(2024) 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") 

# This is for the DEAP dataset.
# For DREAMER and WESAD datasets, only minor adjustments are needed: 
# adjust the values of modalities, channels, 
# and sampling rates at the corresponding positions according to the description in the paper.
# Dataset class for multimodal physiological signals
class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='ULER', split_type='train'):
        super(Multimodal_Datasets, self).__init__()
        # Load preprocessed data from pickle file
        dataset_path = os.path.join(dataset_path, data+'.pkl'  )
        dataset = pickle.load(open(dataset_path, 'rb'))
        self.m1 = torch.tensor(dataset[split_type]['modality1'],dtype=torch.float) 
        self.m2 = torch.tensor(dataset[split_type]['modality2'],dtype=torch.float)
        self.m3 = torch.tensor(dataset[split_type]['modality3'],dtype=torch.float)
        self.m4 = torch.tensor(dataset[split_type]['modality4'],dtype=torch.float) 
        self.labels = torch.tensor(dataset[split_type]['label'],dtype=torch.long)
        self.meta = dataset[split_type]['id'] 
        self.data = data
        
        self.n_modalities = 4 
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.m1.shape[1], self.m2.shape[1], self.m3.shape[1],self.m4.shape[1]
    def get_dim(self):
        return self.m1.shape[2], self.m2.shape[2], self.m3.shape[2],self.m4.shape[2]
    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.m1[index], self.m2[index], self.m3[index],self.m4[index])
        Y = self.labels[index]
        META = self.meta[index] 
        return X, Y, META   
     
# Get data for a specific split        
def get_data(args, split='train'):
    data_path = os.path.join(args.data_path, args.data_index) + f'_{split}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, args.data_index, split)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data

# Read data
def data_reader(args):
    
    train_data = get_data(args, 'train')
    test_data = get_data(args, 'test')
    return train_data,test_data

# Create data loaders for train and test sets
def data_loader(args):
    
    train_data,test_data = data_reader(args)
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=args.batch_size)
    return train_loader,test_loader

