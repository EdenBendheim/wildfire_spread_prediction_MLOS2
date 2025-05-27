import os
import numpy as np
import pandas as pd
import cudf
from cuml.neighbors import KNeighborsClassifier as cuKNN
import time

from cuml.svm import SVC as cuSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.impute import SimpleImputer
import warnings
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import rmm
import geopandas as gpd
from shapely.geometry import Point
import random
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", message=".*CuPy may not function correctly.*")

print("Pre-allocated RMM memory pool (30 GB)")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# PyG imports
import torch_geometric
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
from torch_cluster import knn_graph

# feature columns
VARS = [
    "AvgSurfT_tavg",
    "Rainf_tavg",
    "TVeg_tavg",
    "Wind_f_tavg",
    "Tair_f_tavg",
    "Qair_f_tavg",
    "SoilMoi00_10cm_tavg",
    "SoilTemp00_10cm_tavg",
]

# 1) Load parquet (try cuDF for speed, fallback to pandas)
print("Loading dataset…")
try:
    gdf = cudf.read_parquet("training_dataset.parquet")
    df = gdf.to_pandas()  # for convenience in labeling
    print("  loaded with cuDF")
except Exception:
    df = pd.read_parquet("training_dataset.parquet")
    print("  loaded with pandas")

# ensure date is datetime
df["date"] = pd.to_datetime(df["date"])

# --- Load background GeoDataFrame for plotting ---
training_df = df.copy()
training_df["date"] = pd.to_datetime(training_df["date"]).dt.date
training_gdf = gpd.GeoDataFrame(
    training_df,
    geometry=gpd.points_from_xy(training_df.lon, training_df.lat),
    crs="EPSG:4326"
)

# 2) build binary label: did this cell burn the next day?
df["next_date"] = df["date"] + pd.Timedelta(days=1)
key_cols = ["fireID", "lat", "lon", "next_date"]
df_next = df[key_cols].copy()
df_next["burn_next_day"] = 1
# join back to original on (fireID, lat, lon, date==next_date)
df = df.merge(
    df_next.rename(columns={"next_date": "date"}), 
    on=["fireID","lat","lon","date"], 
    how="left"
)
df["burn_next_day"] = df["burn_next_day"].fillna(0).astype(int)
print("Label distribution:\n", df["burn_next_day"].value_counts())

# 3) train/test split by fireID ensuring no overlap of fire tracks
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["fireID"]))
train_df = df.iloc[train_idx].copy()
test_df  = df.iloc[test_idx].copy()
train_ids = train_df["fireID"].unique().tolist()
test_ids  = test_df["fireID"].unique().tolist()
print(f"Train fires: {len(train_ids)}, Test fires: {len(test_ids)}")
print(f"Train cells: {len(train_df):,}, Test cells: {len(test_df):,}")

# --- Data Snooping Check ---
# We split by fireID, so no single fire appears in both train and test sets.
assert set(train_ids).isdisjoint(set(test_ids)), "Data leak: some fires are in both train and test!"

# Impute missing values in features using training set means
imputer = SimpleImputer(strategy="mean")
train_df.loc[:, VARS] = imputer.fit_transform(train_df[VARS])
test_df.loc[:, VARS]  = imputer.transform(test_df[VARS])

# feature matrix & labels
feature_cols = VARS  # you can add merged area features if desired
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df["burn_next_day"].values
X_test  = test_df[feature_cols].values.astype(np.float32)
y_test  = test_df["burn_next_day"].values

# Convert to cuDF once for GPU models to avoid repeated conversions
X_train_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train, columns=feature_cols))
y_train_cudf = cudf.Series(y_train)
X_test_cudf  = cudf.DataFrame.from_pandas(pd.DataFrame(X_test,  columns=feature_cols))

########################################################################
# A) FAISS GPU KNN (if available)
########################################################################
# B) cuML KNN (cudf) timing
print("\n=== cuML KNN (cudf) ===")
knn = cuKNN(n_neighbors=5)
start_time = time.time()
knn.fit(X_train_cudf, y_train_cudf)
y_pred_cuml = knn.predict(X_test_cudf).values.get()
cuml_time = time.time() - start_time
acc_cuml = accuracy_score(y_test, y_pred_cuml)
print(f"cuML KNN fit+predict completed in {cuml_time:.2f} seconds")
print(f"cuML KNN Accuracy: {acc_cuml:.4f}")

# B) cuML SVM with RBF kernel
svm = cuSVC(kernel="rbf", C=1.0, probability=False)
svm.fit(cudf.DataFrame.from_pandas(pd.DataFrame(X_train, columns=feature_cols)), 
        cudf.Series(y_train))
y_pred_svm = svm.predict(cudf.DataFrame.from_pandas(pd.DataFrame(X_test, columns=feature_cols))).to_array()
print("=== cuML SVM Accuracy:")

########################################################################
# C) PyTorch MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device for NN:", device)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

# standardize features
scaler = StandardScaler().fit(X_train)
Xt_train = scaler.transform(X_train)
Xt_test  = scaler.transform(X_test)

train_ds = TensorDataset(torch.from_numpy(Xt_train), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(Xt_test),  torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=4)

model = MLP(len(feature_cols)).to(device)
opt   = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# train
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"MLP Epoch {epoch}", leave=False):
        xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch:02d}  loss={(total_loss/len(train_ds)):.4f}")

# Fast MLP evaluation: one-shot inference on test set
model.eval()
# optional: half-precision inference
model.half()
with torch.no_grad():
    Xt_test_tensor = torch.from_numpy(Xt_test).to(device).half()
    y_pred_nn = (model(Xt_test_tensor) > 0.5).cpu().numpy().astype(int).ravel()
print("=== MLP Accuracy:", accuracy_score(y_test, y_pred_nn))

########################################################################
# D) GCN via PyTorch Geometric
print("\nBuilding graph for GCN…")
# We'll build one big k-NN graph across all test nodes
# (for simplicity—ideally you'd batch per-day or per-fire)
x_all = torch.from_numpy(X_train).to(device)
y_all = torch.from_numpy(y_train).to(device)
# use knn_graph to build edges (k=8)
edge_index = knn_graph(x_all, k=8, batch=None, loop=False).to(device)

# Mixed precision scaler for faster GCN training
scaler_gcn = GradScaler()

class GCN(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, 64)
        self.conv2 = GCNConv(64, 16)
        self.lin   = nn.Linear(16, 1)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.lin(x)).squeeze()

gcn = GCN(len(feature_cols)).to(device)
opt_gcn = optim.Adam(gcn.parameters(), lr=1e-3)

# single GraphData object
data = Data(x=x_all, edge_index=edge_index, y=y_all)

# training loop
gcn.train()
for epoch in tqdm(range(5), desc="GCN Training"):
    opt_gcn.zero_grad()
    # mixed-precision forward/backward
    with autocast():
        out = gcn(data.x, data.edge_index)
        loss = loss_fn(out.unsqueeze(1), data.y.float().unsqueeze(1))
    scaler_gcn.scale(loss).backward()
    scaler_gcn.step(opt_gcn)
    scaler_gcn.update()
    print(f"GCN Epoch {epoch:02d}  loss={loss.item():.4f}")

# evaluate on test split
# build test graph similarly
x_test_all = torch.from_numpy(X_test).to(device)
edge_test = knn_graph(x_test_all, k=8).to(device)
gcn.eval()
with torch.no_grad():
    out_test = gcn(x_test_all, edge_test).cpu().numpy() > 0.5
print("=== GCN Accuracy:", accuracy_score(y_test, out_test))

# --- Save Predicted vs. Real for Sample Fires ---
import os

# Consolidate results into a DataFrame
results = test_df[["fireID", "lat", "lon", "date"]].copy()
results["actual"]   = y_test
results["pred_knn"] = y_pred_faiss
results["pred_svm"] = y_pred_svm
results["pred_nn"]  = y_pred_nn
results["pred_gcn"] = out_test.astype(int)

# Select a handful of test fireIDs to save
sample_fires = list(test_ids[:5])
os.makedirs("predictions", exist_ok=True)
for fid in sample_fires:
    df_fire = results[results["fireID"] == fid]
    df_fire.to_csv(f"predictions/predictions_fire_{fid}.csv", index=False)

print(f"Saved predicted vs real CSVs for fires: {sample_fires}")

# --- Plotting Examples ---
def plot_examples_for_model(pred_col, model_name):
    """
    Generate 10 random example plots for the given prediction column.
    """
    out_dir = f"plots/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    sample = results.dropna(subset=[pred_col]).sample(10, random_state=42)
    for idx, row in sample.iterrows():
        fid = row.fireID
        d = row.date
        # background
        bg = training_gdf[(training_gdf.fireID==fid)&(training_gdf.date==d)]
        # true spread
        true_cells = results[(results.fireID==fid)&(results.date==d)&(results.actual==1)]
        true_gdf = gpd.GeoDataFrame(
            true_cells,
            geometry=gpd.points_from_xy(true_cells.lon, true_cells.lat),
            crs="EPSG:4326"
        )
        # predicted spread
        pred_cells = results[(results.fireID==fid)&(results.date==d)&(results[pred_col]==1)]
        pred_gdf = gpd.GeoDataFrame(
            pred_cells,
            geometry=gpd.points_from_xy(pred_cells.lon, pred_cells.lat),
            crs="EPSG:4326"
        )
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        if not bg.empty:
            bg.plot(ax=ax, column="AvgSurfT_tavg", cmap="coolwarm", markersize=5, alpha=0.5, zorder=1)
        if not true_gdf.empty:
            true_gdf.plot(ax=ax, color="red", markersize=5, alpha=0.7, zorder=2, label="True")
        if not pred_gdf.empty:
            pred_gdf.plot(ax=ax, color="blue", markersize=5, alpha=0.7, zorder=3, label="Pred")
        ax.set_title(f"{model_name} - Fire {fid} on {d}", fontsize=12)
        ax.legend()
        ax.set_axis_off()
        fig.savefig(os.path.join(out_dir, f"{model_name}_{fid}_{d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

# Generate example plots
plot_examples_for_model("pred_knn", "knn")
plot_examples_for_model("pred_svm", "svm")
plot_examples_for_model("pred_nn", "mlp")
plot_examples_for_model("pred_gcn", "gcn")