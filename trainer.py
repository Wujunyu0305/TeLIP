import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.nn import BCELoss
from sklearn.model_selection import KFold

def position_error_metric(pred_probs, sample_ids, slope_info_map):
    pred_positions = pred_probs.argmax(dim=1)
    errors = []
    for i, pred_pos in enumerate(pred_positions):
        sid = sample_ids[i].item() if isinstance(sample_ids[i], torch.Tensor) else sample_ids[i]
        slope_len, true_ls_pos = slope_info_map[sid]
        pred_distance = pred_pos.item() * 5 + 2.5
        error = abs(pred_distance - true_ls_pos) / slope_len
        errors.append(error)
    return errors


def train_one_split(model, train_loader, val_loader, slope_info_map, device, max_epochs=500, patience=50):
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_error = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, sid_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        pos_errors_all = []
        with torch.no_grad():
            for x_batch, y_batch, sid_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                pos_errors = position_error_metric(preds, sid_batch, slope_info_map)
                pos_errors_all.extend(pos_errors)

        val_loss /= len(val_loader.dataset)
        mean_pos_error = np.mean(pos_errors_all)

        print(f"Epoch {epoch:03d} | Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Norm Pos Error = {mean_pos_error:.4f}")

        if mean_pos_error < best_val_error:
            best_val_error = mean_pos_error
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best Norm Pos Error = {best_val_error:.4f}")
            break

    return best_model_state, best_val_error
