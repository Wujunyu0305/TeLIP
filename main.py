from datasets import SlopeDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from models import SlopeTransformer
from trainer import train_one_split
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_df = pd.read_csv(r"path/to/your/Geological data.CSV")
    labels_df = pd.read_csv(r"path/to/your/Landslide position data.CSV")
    slope_length_df = pd.read_csv(r"path/to/your/Slope length data.CSV")

    slope_info_map = {
        row['Sample ID']: (row['Slope Length'], row['True Landslide Position'])
        for _, row in slope_length_df.iterrows()
    }

    dataset = SlopeDataset(features_df, labels_df)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    all_norm_errors = []

    for seed in range(10):
        print(f"\nRunning fold {seed + 1} with random seed {seed}...")

        fold_norm_errors = []

        for train_idx, val_idx in kfold.split(np.arange(len(dataset))):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

            model = SlopeTransformer(feature_dim=8, d_model=64, nhead=4, num_layers=2).to(device)
            best_state_dict, best_error = train_one_split(
                model, train_loader, val_loader, slope_info_map, device,
                max_epochs=500, patience=50
            )
            fold_norm_errors.append(best_error)

        mean_fold_error = np.mean(fold_norm_errors)
        all_norm_errors.append(mean_fold_error)
        print(f"Average Norm Pos Error for seed {seed + 1}: {mean_fold_error:.4f}")

    final_mean_error = np.mean(all_norm_errors)
    print(f"\n=== Final Mean Norm Pos Error (MNPE) across all runs = {final_mean_error:.4f}")

if __name__ == "__main__":
    main()
