import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


'''Create data loader'''

class MovieDataset(Dataset):
    def __init__(self, filename):
        self.df = pd.read_csv(filename, converters={'input_x': literal_eval})
        # print(self.df['input_x'])

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        ## load the input features and labels
        ##-----------------------------------------------
        ## complete the code to load features and labels
        ##-----------------------------------------------
        input_x = self.df.iloc[index]['input_x']
        label = self.df.iloc[index]['Label']
        

        return torch.tensor(input_x), torch.tensor(label,dtype=torch.float)
        
        