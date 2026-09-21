import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import OceanEmbedModel

def train_model(epochs=50, lr=0.001, batch_size=4):
    print("=" * 80)
    print("STAGE 10, 11, 12, 13: 15-DEPTH MODEL TRAINING & EVALUATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed datasets
    data_dir = os.path.join("data", "processed")
    X_norm = np.load(os.path.join(data_dir, "X_norm.npy"))  # Shape: (31, 5, 101, 241)
    Y_norm = np.load(os.path.join(data_dir, "Y_norm.npy"))  # Shape: (31, 15, 101, 241)
    Y_raw = np.load(os.path.join(data_dir, "Y_raw.npy"))    # Shape: (31, 15, 101, 241)
    ocean_mask = np.load(os.path.join(data_dir, "ocean_mask.npy"))
    grid = np.load(os.path.join(data_dir, "grid.npz"))
    depths = grid['depth']

    with open(os.path.join(data_dir, "norm_params.json"), "r") as f:
        norm_params = json.load(f)

    target_mean = norm_params['target']['mean']
    target_std = norm_params['target']['std']

    T, C_in, H, W = X_norm.shape
    K = Y_norm.shape[1]

    print(f"Loaded dataset: X_norm={X_norm.shape}, Y_norm={Y_norm.shape}")
    print(f"Target Depths (K={K}): {depths.tolist()} m")

    train_idx = list(range(0, 25))
    val_idx = list(range(25, 31))

    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y_norm[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y_norm[val_idx], dtype=torch.float32)
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.bool, device=device)

    model = OceanEmbedModel(in_channels=C_in, feature_dim=64, hidden_dim=128, out_depths=K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def masked_mse_loss(pred, target, mask):
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
        diff = (pred - target)[mask_expanded]
        return torch.mean(diff ** 2)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "ocean_embed_model.pth")

    print("\nStarting 15-depth model training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        
        num_batches = int(np.ceil(len(train_idx) / batch_size))
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, len(train_idx))
            
            x_b = X_train[b_start:b_end].to(device)
            y_b = Y_train[b_start:b_end].to(device)

            optimizer.zero_grad()
            pred_b = model(x_b)
            loss = masked_mse_loss(pred_b, y_b, mask_tensor)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * (b_end - b_start)

        epoch_train_loss /= len(train_idx)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            x_v = X_val.to(device)
            y_v = Y_val.to(device)
            pred_v = model(x_v)
            val_loss = masked_mse_loss(pred_v, y_v, mask_tensor).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to '{best_model_path}'")

    # Evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with torch.no_grad():
        pred_val_norm = model(X_val.to(device)).cpu().numpy()

    pred_val_degC = pred_val_norm * target_std + target_mean
    target_val_degC = Y_raw[val_idx]

    val_ocean_mask = np.broadcast_to(ocean_mask, target_val_degC.shape)
    y_true_ocean = target_val_degC[val_ocean_mask]
    y_pred_ocean = pred_val_degC[val_ocean_mask]

    overall_rmse = np.sqrt(np.mean((y_pred_ocean - y_true_ocean) ** 2))
    overall_bias = np.mean(y_pred_ocean - y_true_ocean)
    overall_corr = np.corrcoef(y_pred_ocean, y_true_ocean)[0, 1]

    print("\nOverall Validation Metrics (Un-normalized physical values in °C):")
    print(f"  - Overall RMSE       : {overall_rmse:.4f} °C")
    print(f"  - Overall Correlation: {overall_corr:.4f}")
    print(f"  - Overall Bias       : {overall_bias:.4f} °C")

    # Depth-wise evaluation
    depth_metrics = []
    for d_idx in range(K):
        d_val = depths[d_idx]
        t_d = target_val_degC[:, d_idx]
        p_d = pred_val_degC[:, d_idx]
        mask_d = np.broadcast_to(ocean_mask, t_d.shape)
        
        y_t_d = t_d[mask_d]
        y_p_d = p_d[mask_d]
        
        rmse_d = np.sqrt(np.mean((y_p_d - y_t_d) ** 2))
        bias_d = np.mean(y_p_d - y_t_d)
        corr_d = np.corrcoef(y_p_d, y_t_d)[0, 1]
        depth_metrics.append({
            'depth_m': float(d_val),
            'rmse': float(rmse_d),
            'bias': float(bias_d),
            'correlation': float(corr_d)
        })

    os.makedirs("reports", exist_ok=True)
    metrics_report = {
        "rmse_degC": float(overall_rmse),
        "correlation": float(overall_corr),
        "bias_degC": float(overall_bias),
        "best_val_loss_normalized": float(best_val_loss),
        "depth_metrics": depth_metrics
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=4)

    # Save loss curve plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (MSE)')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss (MSE)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized MSE Loss')
    plt.title('OceanEmbed 15-Depth Training & Validation Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/loss_curve.png', dpi=300)
    plt.close()

    # Save RMSE by depth plot
    plt.figure(figsize=(7, 5))
    d_list = [m['depth_m'] for m in depth_metrics]
    rmse_list = [m['rmse'] for m in depth_metrics]
    plt.plot(rmse_list, d_list, 'o-', color='#0284c7', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel('RMSE (°C)')
    plt.ylabel('Depth (m)')
    plt.title('OceanEmbed Subsurface RMSE by Depth')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/rmse_by_depth.png', dpi=300)
    plt.close()

    print("Saved training artifacts: 'reports/loss_curve.png', 'reports/rmse_by_depth.png', and 'reports/metrics.json'.")

if __name__ == "__main__":
    train_model(epochs=50, lr=0.001)
