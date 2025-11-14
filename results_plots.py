import os
import xarray as xr
import cftime 
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import PixelLSTM
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import brier_score_loss,r2_score, mean_squared_error
from scipy.special import expit  # Sigmoid
import seaborn as sns



# === Load your saved data ===
X_test = np.load("data/X_test.npy")              # Shape: (T, P, F)
static_test = np.load("data/static_test.npy")    # Shape: (P, S)
y_test = np.load("data/y_test.npy")              # Shape: (T, P)
coords_test = torch.load("data/coords_test.npy", map_location='cpu', weights_only=False)


# === Preprocess ===
X_test = torch.tensor(X_test.transpose(1, 0, 2), dtype=torch.float32)  # (P, T, F)
static_test = torch.tensor(static_test, dtype=torch.float32)          # (P, S)
y_test = torch.tensor(y_test.transpose(1, 0), dtype=torch.float32)


# === Define model architecture (must match training) ===
model = PixelLSTM(
    input_size=X_test.shape[2],
    static_size=static_test.shape[1],
    hidden_size=256,
    num_layers=3,
    dropout=0.3987832655685518
)

# === Load the weights ===
checkpoint = torch.load("data/model_checkpoint.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])  # ✅ just the model weights


model.eval()

# === Run inference ===
with torch.no_grad():
    predictions = model(X_test, static_test)  # Shape: (P, T)

# === Convert to NumPy if needed ===
predictions_np = predictions.numpy()

# === Evaluate (optional) =====

mse = torch.nn.functional.mse_loss(predictions, y_test)

print(f"Test MSE: {mse.item():.4f}")

####################################################################################


def compute_core_metrics_per_pixel(y_true, y_pred):
    """
    Computes R², MSE, and RMSE per pixel.

    Parameters:
        y_true: torch.Tensor of shape (P, T)
        y_pred: torch.Tensor of shape (P, T)

    Returns:
        metrics: list of np.ndarray, each of shape (P,)
                 [R², MSE, RMSE]
    """
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    P = y_true_np.shape[0]

    r2_arr = np.full(P, np.nan)
    mse_arr = np.full(P, np.nan)
    rmse_arr = np.full(P, np.nan)

    for p in range(P):
        y_t = y_true_np[p]
        y_p = y_pred_np[p]

        if np.all(np.isnan(y_t)) or np.all(np.isnan(y_p)):
            continue

        try:
            r2_arr[p] = r2_score(y_t, y_p)
            mse_arr[p] = mean_squared_error(y_t, y_p)
            rmse_arr[p] = np.sqrt(mse_arr[p])
        except ValueError:
            continue

    return [r2_arr, mse_arr, rmse_arr]


####################################################################################


def plot_spatial_metric(metric_values, coords, title="Spatial Metric", cmap='viridis', vmin=None, vmax=None):
    """
    Plots a spatial metric (e.g., MSE, R²) over geographic coordinates.

    Parameters:
        metric_values (np.ndarray): Array of shape (P,) with metric per pixel
        coords (array-like): Array of (lat, lon) coordinates, shape (P, 2)
        title (str): Plot title
        cmap (str): Matplotlib colormap name
        vmin, vmax (float): Optional color limits for the metric scale
    """
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='#a6cee3')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, facecolor='azure', alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4)

    # Plot the metric
    sc = ax.scatter(lons, lats, c=metric_values, cmap=cmap, s=15, marker='s', edgecolor='none',
                    vmin=vmin, vmax=vmax)

    # Gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical',aspect=30)
    cbar.ax.tick_params(labelsize=20)  # Tick label font size

    ax.set_title(title, fontsize=30,weight = 'bold', pad=20)


    plt.tight_layout()

    #plt.savefig("RMSE.png", dpi=300, bbox_inches='tight')

    plt.show()

####################################################################################



def plot_metric_distribution(metric_values, title="Metric Distribution", xlabel="Metric Value", bins=30):
    """
    Plots the distribution of spatial metric values using histogram + KDE.

    Parameters:
        metric_values (np.ndarray): 1D array of metric values (e.g., per pixel)
        title (str): Title for the plot
        xlabel (str): Label for x-axis
        bins (int): Number of bins in histogram
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(9, 6))
    ax = sns.histplot(metric_values, bins=bins, kde=True, color="#1f77b4", edgecolor='white')

    # Median line
    median_val = np.median(metric_values)
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2, label=None)
    ax.legend(fontsize=14)

    # Formatting
    ax.set_title(title, fontsize=22, weight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
    ax.set_ylabel("")  # Remove y-axis label
    ax.set_yticks([])  # Remove y-axis ticks

    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{xlabel.lower().replace(' ', '_')}_distribution.png", dpi=300, bbox_inches='tight')

    plt.show()

####################################################################################
metrics = compute_core_metrics_per_pixel(y_test, predictions)



#np.clip(metrics[0],0.7,1)

plot_spatial_metric(metrics[0], coords_test, title="R² (clipped at 0.7)", cmap='RdYlGn', vmin=0.7, vmax=1)
plot_metric_distribution(metrics[0], title="R² Distribution", xlabel="R²")


####################################################################################
plot_spatial_metric(metrics[0], coords_test, title="R²", cmap='RdYlGn', vmin=0, vmax=1)
plot_metric_distribution(metrics[0], title="R² Distribution", xlabel="R²")
####################################################################################
plot_spatial_metric(metrics[1], coords_test, title="MSE (clipped at 0.3)", cmap='RdYlGn_r',vmin = 0,vmax = 0.3)
plot_metric_distribution(metrics[1], title="MSE Distribution", xlabel="MSE")
####################################################################################
plot_spatial_metric(metrics[2], coords_test, title="RMSE (clipped at 0.5)", cmap='RdYlGn_r',vmin = 0,vmax = 0.5)
plot_metric_distribution(metrics[2], title="RMSE Distribution", xlabel="RMSE")
####################################################################################
def plot_good_bad_pixels(y_true, y_pred, num_good=1, num_bad=1):
    _, mse_arr, _ = compute_core_metrics_per_pixel(y_true, y_pred)

    good_indices = np.argsort(mse_arr)[:num_good]
    bad_indices = np.argsort(mse_arr)[-num_bad:]

    t = np.arange(y_true.shape[1])

    for idx in good_indices:
        plt.figure(figsize=(12, 6))
        plt.plot(t, y_true[idx].numpy(), label="True", alpha=0.8, linewidth=2)
        plt.plot(t, y_pred[idx].numpy(), label="Predicted", alpha=0.8, linewidth=2, linestyle='--')
        plt.title(f"Good Pixel (Index {idx}, MSE={mse_arr[idx]:.4f})", fontsize=22, weight='bold')
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"good_pixel_{idx}.png", dpi=600)
        plt.show()

    for idx in bad_indices:
        plt.figure(figsize=(12, 6))
        plt.plot(t, y_true[idx].numpy(), label="True", alpha=0.8, linewidth=2)
        plt.plot(t, y_pred[idx].numpy(), label="Predicted", alpha=0.8, linewidth=2, linestyle='--')
        plt.title(f"Bad Pixel (Index {idx}, MSE={mse_arr[idx]:.4f})", fontsize=22, weight='bold')
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"bad_pixel_{idx}.png", dpi=600)
        plt.show()

####################################################################################
plot_good_bad_pixels(y_test,predictions,2,2)
####################################################################################



landcover = xr.open_dataset("data/landcover_MODIS_006_MCD12Q1_yearly__2001_2020_0_1_degree_regrid_s_africa_l.nc")
landcover_large = landcover

lc_classes = {
    1:  ("Evergreen Needleleaf Forest", "#005500"),
    2:  ("Evergreen Broadleaf Forest", "#007f00"),
    3:  ("Deciduous Needleleaf Forest", "#55aa00"),
    4:  ("Deciduous Broadleaf Forest", "#66ff00"),
    5:  ("Mixed Forest", "#228b22"),
    6:  ("Closed Shrublands", "#dcd29a"),
    7:  ("Open Shrublands", "#cdb266"),
    8:  ("Woody Savannas", "#b8860b"),
    9:  ("Savannas", "#f4a460"),
    10: ("Grasslands", "#88cc44"),
    11: ("Permanent Wetlands", "#33cccc"),
    12: ("Croplands", "#ffff64"),
    13: ("Urban and Built-up", "#ff0000"),
    14: ("Cropland/Natural Veg. Mosaic", "#bfbf00"),
    15: ("Snow and Ice", "#ffffff"),
    16: ("Barren or Sparsely Vegetated", "#d9d9d9"),
    17: ("Water Bodies", "#0000ff"),
    255: ("Unclassified", "#000000")  # optional
}

lc_clean = landcover_large['LC_Type1'].astype('uint8')
lc_plot = lc_clean.isel(time=19)

####################################################################################
def plot_landcover_brier_boxplot_1d(
    brier,
    coords,
    lc_da,
    lc_classes,
    title="ΔMSE Distribution by Land Cover Type"
):


    # Convert torch to numpy if needed
    if hasattr(brier, "cpu"):
        brier = brier.cpu().numpy()

    # Extract landcover values at each (lat, lon) coord
    lc_values = []
    for lat, lon in coords:
        try:
            lc_val = lc_da.sel(lat=lat, lon=lon, method="nearest").values.item()
        except:
            lc_val = np.nan
        lc_values.append(lc_val)

    lc_values = np.array(lc_values).astype(float)

    # Filter valid entries
    valid_mask = ~np.isnan(lc_values) & ~np.isnan(brier)
    brier_valid = brier[valid_mask]
    lc_valid = lc_values[valid_mask].astype(int)

    # Group values by land cover
    box_data = {}
    for lc in np.unique(lc_valid):
        if lc in lc_classes:
            box_data[lc] = brier_valid[lc_valid == lc]

    # Preserve order from lc_classes
    sorted_items = [(k, box_data[k]) for k in lc_classes if k in box_data]


    # Prepare labels and data
    labels = [f"{lc_classes[k][0]} ({len(v)})" for k, v in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [lc_classes[k][1] for k, _ in sorted_items]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bplot = ax.boxplot(values, vert=False, patch_artist=True, labels=labels)

    # Set colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Performance",fontsize = 14,fontweight = 'bold')

    ax.set_xlim(0, 1)
    ax.set_title(title,fontsize = 14, fontweight = 'bold')


    plt.tight_layout()
    plt.savefig('MSE_land.png', dpi = 300)
    plt.show()

####################################################################################
plot_landcover_brier_boxplot_1d(
    brier = metrics[0],
    coords=coords_test,
    lc_da=lc_plot,
    lc_classes=lc_classes,
    title=f"R2 by landcover"
    )
####################################################################################
plot_landcover_brier_boxplot_1d(
    brier = metrics[1],
    coords=coords_test,
    lc_da=lc_plot,
    lc_classes=lc_classes,
    title=f"MSE by landcover"
    )
####################################################################################
