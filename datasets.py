import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SlopeDataset(Dataset):
    def __init__(self, features_df, labels_df):
        self.sample_ids = features_df['Sample ID'].unique()
        self.features = []
        self.labels = []

        for sid in self.sample_ids:
            sub_f = features_df[features_df['Sample ID'] == sid].sort_values('Position ID')
            sub_l = labels_df[labels_df['Sample ID'] == sid].sort_values('Position ID')

            feat_array = sub_f.loc[:, [
                'Relative Height', 'Elevation', 'Slope', 'Soil Thickness',
                'Profile Curvature', 'Plan Curvature', 'TWI', 'Position Index'
            ]].values
            label_array = sub_l['Landslide Binary Value'].values

            self.features.append(feat_array)
            self.labels.append(label_array)

        self.features = np.stack(self.features)
        self.labels = np.stack(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        sid = self.sample_ids[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), sid
