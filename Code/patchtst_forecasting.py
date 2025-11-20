import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# SETTINGS
input_length = 5
forecast_steps = 5
target_col = "public_sector_debt"  # Ensure lowercase and matches merged_df
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_patchtst_forecasting(merged_df):
    feature_cols = [col for col in merged_df.columns if col not in ["Ccode", "Year", target_col]]

    # Standardize features
    scaler = StandardScaler()
    merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

    # Build sliding windows
    X_all, y_all, meta_all = [], [], []
    for country, group in merged_df.sort_values("Year").groupby("Ccode"):
        values = group[feature_cols + [target_col]].values
        years = group["Year"].values
        if len(values) < input_length + 1:
            continue
        for i in range(len(values) - input_length):
            X_all.append(values[i:i+input_length, :-1])
            y_all.append(values[i+input_length, -1])
            meta_all.append((country, years[i+input_length]))

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    class PatchTST(nn.Module):
        def __init__(self, input_len, pred_len, n_features, d_model=64, n_heads=4, n_layers=3, patch_len=1, dropout=0.1):
            super().__init__()
            self.patch_embed = nn.Linear(patch_len * n_features, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, input_len, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(input_len * d_model, pred_len)

        def forward(self, x):
            x = self.patch_embed(x) + self.pos_embed[:, :x.size(1)]
            x = self.transformer(x)
            x = self.norm(x)
            return self.head(x.flatten(1))

    dataset = DataLoader(list(zip(X_tensor, y_tensor)), batch_size=32, shuffle=True)
    model = PatchTST(input_length, 1, len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        model.train()
        losses = []
        for xb, yb in dataset:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        preds_all = model(X_tensor.to(device)).cpu().numpy().squeeze()

    results_df = pd.DataFrame(meta_all, columns=["Ccode", "Year"])
    results_df["Actual_Debt"] = y_tensor.numpy()
    results_df["Predicted_Debt"] = preds_all

    future_rows = []
    for country in merged_df["Ccode"].unique():
        group = merged_df[merged_df["Ccode"] == country].sort_values("Year")
        if len(group) < input_length:
            continue
        input_seq = group[feature_cols].values[-input_length:]
        last_year = group["Year"].max()
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        for step in range(forecast_steps):
            with torch.no_grad():
                pred = model(input_tensor).squeeze().cpu().item()
            future_rows.append({
                "Ccode": country,
                "Year": last_year + step + 1,
                "Actual_Debt": None,
                "Predicted_Debt": pred
            })
            new_row = input_tensor[:, -1, :].clone()
            new_row[0, feature_cols.index(target_col)] = pred
            input_tensor = torch.cat([input_tensor[:, 1:], new_row.unsqueeze(1)], dim=1)

    future_df = pd.DataFrame(future_rows)
    full_df = pd.concat([results_df, future_df], ignore_index=True).sort_values(["Ccode", "Year"])

    return full_df

def plot_forecasts(full_df, top_n=5):
    top_countries = full_df["Ccode"].value_counts().index[:top_n]
    for country in top_countries:
        df = full_df[full_df["Ccode"] == country]
        plt.figure(figsize=(8, 4))
        plt.plot(df["Year"], df["Actual_Debt"], label="Actual", marker='o')
        plt.plot(df["Year"], df["Predicted_Debt"], label="Predicted", linestyle="--", marker='x')
        plt.title(f"Public Sector Debt Forecast + 5-Year Projection: {country}")
        plt.xlabel("Year")
        plt.ylabel("Debt (Normalized)")
        plt.legend()
        plt.grid(True)
        plt.show()
