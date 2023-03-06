from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
import pickle

class SWDEDataset(Dataset):
    def __init__(self, dataset_path, domain="university",split="train"):
        self.path = Path(dataset_path) / domain
        self.files = self._get_split(sorted(self.path.glob("**/*.pkl")),split)
        self._idx2file = []        
        for file_id, file in enumerate(self.files):            
            with open(file,'rb') as f:
                features = pickle.load(f)    
            prev_len = len(self._idx2file)        
            self._idx2file.extend(((prev_len,file_id) for _ in range(len(features))) )            
        self.current_features = None
        self.current_file_idx = None

    def _get_split(self,files, split,seed=42):
        train, test = train_test_split(files,test_size=0.2,random_state=seed)
        if split == "train":
            return train
        return test

    def __len__(self):
        return len(self._idx2file)

    def __getitem__(self, idx):        
        idx = len(self) + idx if idx < 0 else idx

        min_idx, file_idx = self._idx2file[idx]
        
        if self.current_file_idx != file_idx:            
            with open(self.files[file_idx],'rb') as f:
                features = pickle.load(f) 
            self.current_features = features  
            self.current_file_idx = file_idx

        return self.current_features[idx - min_idx]


class SWDEAttributeExtractionDataset(SWDEDataset):
    pass
