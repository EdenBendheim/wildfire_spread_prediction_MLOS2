import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import os
import pandas as pd
import concurrent.futures
from functools import partial
from tqdm import tqdm
import warnings
import contextily as cx  # Add this import for satellite imagery
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the training dataset and prepare a GeoDataFrame for background temperature
training_df = pd.read_parquet("training_dataset.parquet")
training_df["date"] = pd.to_datetime(training_df["date"]).dt.date
training_gdf = gpd.GeoDataFrame(
    training_df,
    geometry=gpd.points_from_xy(training_df.lon, training_df.lat),
    crs="EPSG:4326"
)

# Determine temperature column for background plotting
temp_candidates = [col for col in training_df.columns if 'TV' in col and 'eg' in col]
# temp_candidates = [col for col in training_df.columns if 'Wind' in col and 'f' in col]
if temp_candidates:
    temp_col = temp_candidates[0]
    print(f"Using temperature column: {temp_col}")
else:
    temp_col = None
    print("No temperature column found; skipping background plotting")

if temp_col is None:
    raise RuntimeError(
        "No surface temperature column found in training_dataset.parquet. "
        "Please regenerate training_dataset.parquet with Create_training.py including surface temperature."
    )

def add_satellite_basemap(ax, bounds, crs="EPSG:4326"):
    """Add satellite imagery as a basemap to the given axes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the basemap to
    bounds : tuple
        The (minx, miny, maxx, maxy) bounds in the CRS coordinates
    crs : str
        The coordinate reference system of the axes
    """
    try:
        minx, miny, maxx, maxy = bounds
        
        # Add the satellite basemap
        cx.add_basemap(ax, crs=crs, source=cx.providers.Esri.WorldImagery, 
                      zoom='auto', attribution=False)
        
        # Reset the extent to make sure the view doesn't change
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    except Exception as e:
        print(f"Warning: Could not add satellite basemap: {e}")

def process_fire(fire_id, gpkg_fp, layer):
    """Process a single fire: extract, reproject, plot at 5-day intervals with time gradient."""
    gdf = gpd.read_file(gpkg_fp, layer=layer)
    subset = gdf[gdf["fireID"] == fire_id].copy()
    subset["time"] = pd.to_datetime(subset["time"])
    subset["date"] = pd.to_datetime(subset["time"]).dt.date
    subset = subset.to_crs(epsg=4326)

    folder = str(fire_id)
    os.makedirs(folder, exist_ok=True)
    daily_dir = os.path.join(folder, "daily")
    os.makedirs(daily_dir, exist_ok=True)

    # Print centroid & bounds
    ctr = subset.geometry.union_all().centroid
    minx, miny, maxx, maxy = subset.total_bounds
    print(f"[{fire_id}] centroid: {ctr.x:.5f},{ctr.y:.5f}  bounds: {minx:.5f},{miny:.5f},{maxx:.5f},{maxy:.5f}")

    # Get all unique dates and select every 5 days
    all_dates = sorted(subset["date"].unique())
    
    if len(all_dates) <= 1:
        interval_dates = all_dates
    else:
        # Always include first day
        interval_dates = [all_dates[0]]
        # Add every 5th day
        for i in range(5, len(all_dates), 5):
            interval_dates.append(all_dates[i])
        # Always include last day if not already included
        if all_dates[-1] not in interval_dates:
            interval_dates.append(all_dates[-1])
    
    print(f"Fire {fire_id}: Selected {len(interval_dates)} dates at 5-day intervals")
    
    # Define a fire-themed gradient colormap - from yellow to dark red
    from matplotlib.colors import LinearSegmentedColormap
    fire_cmap = LinearSegmentedColormap.from_list('fire_gradient', 
                                                ['#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#8B0000'])
    
    # For each interval date, create a visualization showing all previous intervals
    for i, current_date in enumerate(interval_dates):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Get perimeters up to current date to calculate appropriate bounds
        current_perimeters = subset[subset["date"] <= current_date]
        
        if current_perimeters.empty:
            continue
            
        # Calculate bounds for the current date's perimeters with a buffer
        date_minx, date_miny, date_maxx, date_maxy = current_perimeters.total_bounds
        
        # Add buffer (5% of width/height) around the bounds to avoid cutting off the fire
        width = date_maxx - date_minx
        height = date_maxy - date_miny
        buffer_x = width * 0.1  # 10% buffer on each side
        buffer_y = height * 0.1
        
        # Apply buffer to bounds
        date_minx -= buffer_x
        date_maxx += buffer_x
        date_miny -= buffer_y
        date_maxy += buffer_y
        
        # Set initial extent based on current date's bounds
        ax.set_xlim(date_minx, date_maxx)
        ax.set_ylim(date_miny, date_maxy)
        
        # Add satellite imagery using the date-specific bounds
        add_satellite_basemap(ax, (date_minx, date_miny, date_maxx, date_maxy))
        
        # Plot background temperature data if available
        if temp_col:
            bg = training_gdf[
                (training_gdf.fireID == fire_id) &
                (training_df.date == current_date)
            ]
            if not bg.empty and temp_col in bg.columns:
                # Filter points to the visible area for better performance
                visible_bg = bg[
                    (bg.geometry.x >= date_minx) & 
                    (bg.geometry.x <= date_maxx) & 
                    (bg.geometry.y >= date_miny) & 
                    (bg.geometry.y <= date_maxy)
                ]
                
                if not visible_bg.empty:
                    # Plot without auto-legend first
                    temp_plot = visible_bg.plot(
                        ax=ax,
                        column=temp_col,
                        cmap="coolwarm",
                        markersize=5,
                        alpha=0.6,
                        legend=False,  # Changed from True to False
                        zorder=5
                    )
                    
                    # Create manual legend for temperature data
                    from matplotlib.cm import ScalarMappable
                    from matplotlib.colors import Normalize
                    
                    # Get min and max values for color scaling
                    vmin, vmax = visible_bg[temp_col].min(), visible_bg[temp_col].max()
                    
                    # Create a ScalarMappable for the colorbar
                    temp_norm = Normalize(vmin=vmin, vmax=vmax)
                    temp_sm = ScalarMappable(cmap="coolwarm", norm=temp_norm)
                    temp_sm.set_array([])
                    
                    # Add colorbar on right side with proper label
                    temp_cbar = plt.colorbar(temp_sm, ax=ax, pad=0.01, location='right')
                    temp_cbar.set_label("Vegetation Transpiration", fontsize=10)
        
        # Plot perimeters for all previous interval dates up to current date
        intervals_to_plot = interval_dates[:i+1]
        num_intervals = len(intervals_to_plot)
        
        for j, prev_date in enumerate(intervals_to_plot):
            # Get perimeters for this date
            date_perimeter = subset[subset["date"] == prev_date]
            
            if not date_perimeter.empty:
                # Use gradient based on time progression
                color_val = j / max(1, num_intervals - 1)
                perimeter_color = fire_cmap(color_val)
                
                date_perimeter.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor=perimeter_color,
                    linewidth=2,
                    zorder=j+10
                )
        
        # Add horizontal colorbar below the plot
        sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        
        # Use a horizontal colorbar at the bottom
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        
        # Add start and end date labels to the colorbar
        if len(intervals_to_plot) > 1:
            start_date = intervals_to_plot[0].strftime("%Y-%m-%d")
            end_date = intervals_to_plot[-1].strftime("%Y-%m-%d")
            cbar.ax.set_xticks([0, 1])
            cbar.ax.set_xticklabels([start_date, end_date])
            cbar.ax.set_xlabel('Fire Progression Timeline', labelpad=5)
        else:
            # Only one date
            start_date = intervals_to_plot[0].strftime("%Y-%m-%d")
            cbar.ax.set_xticks([0.5])
            cbar.ax.set_xticklabels([start_date])
        
        # Set title and labels
        ax.set_title(f"Fire {fire_id} Progression Through {current_date}", fontsize=14)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_aspect("equal", adjustable="datalim")
        
        # Save figure with padding adjustments but WITHOUT tight_layout
        # fig.tight_layout()  # Comment out or remove this line
        fig.subplots_adjust(bottom=0.15, right=0.85)  # Make room for both colorbars
        fig.savefig(os.path.join(daily_dir, f"interval_{i:02d}_{current_date}.png"), 
                   dpi=200, bbox_inches="tight")
        plt.close(fig)
    
    # Save summary GeoJSON
    out_geojson = os.path.join(folder, f"{folder}.geojson")
    subset.to_file(out_geojson, driver="GeoJSON")
    
    # Create a final visualization with all interval dates using the gradient
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot a dummy transparent polygon first to initialize the axes
    ax.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], 'k-', alpha=0)
    
    # Set initial extent based on data bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # Add satellite imagery as background using bounds
    add_satellite_basemap(ax, (minx, miny, maxx, maxy))
    
    # If we want to add temperature data to the final visualization as well
    if temp_col:
        bg = training_gdf[
            (training_gdf.fireID == fire_id) &
            (training_gdf.date == interval_dates[-1])  # Use the last date for final viz
        ]
        if not bg.empty and temp_col in bg.columns:
            # Filter points to the visible area
            visible_bg = bg[
                (bg.geometry.x >= minx) & 
                (bg.geometry.x <= maxx) & 
                (bg.geometry.y >= miny) & 
                (bg.geometry.y <= maxy)
            ]
            
            if not visible_bg.empty:
                # Plot without auto-legend
                temp_plot = visible_bg.plot(
                    ax=ax,
                    column=temp_col,
                    cmap="coolwarm",
                    markersize=5,
                    alpha=0.6,
                    legend=False,  # Changed from True to False
                    zorder=5
                )
                
                # Add manual colorbar legend
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                
                # Get min and max values for the temperature column
                vmin, vmax = visible_bg[temp_col].min(), visible_bg[temp_col].max()
                
                # Create a ScalarMappable for the colorbar
                temp_norm = Normalize(vmin=vmin, vmax=vmax)
                temp_sm = ScalarMappable(cmap="coolwarm", norm=temp_norm)
                temp_sm.set_array([])
                
                # Add colorbar on right side with proper label
                temp_cbar = plt.colorbar(temp_sm, ax=ax, pad=0.01, location='right')
                temp_cbar.set_label("Vegetation Transpiration", fontsize=10)
    
    # Plot all interval dates with gradient
    for j, interval_date in enumerate(interval_dates):
        date_perimeter = subset[subset["date"] == interval_date]
        if not date_perimeter.empty:
            # Use gradient based on time progression
            color_val = j / max(1, len(interval_dates) - 1)  # normalize to 0-1
            perimeter_color = fire_cmap(color_val)
            
            date_perimeter.plot(
                ax=ax,
                facecolor='none',
                edgecolor=perimeter_color,
                linewidth=2,
                zorder=j+10
            )
    
    # Add horizontal colorbar below the plot for the final image
    sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    
    # Use a horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Add start and end date labels to the colorbar
    if len(interval_dates) > 1:
        start_date = interval_dates[0].strftime("%Y-%m-%d")
        end_date = interval_dates[-1].strftime("%Y-%m-%d")
        cbar.ax.set_xticks([0, 1])
        cbar.ax.set_xticklabels([start_date, end_date])
        cbar.ax.set_xlabel('Fire Progression Timeline', labelpad=5)
    else:
        # Only one date
        start_date = interval_dates[0].strftime("%Y-%m-%d")
        cbar.ax.set_xticks([0.5])
        cbar.ax.set_xticklabels([start_date])
    
    ax.set_title(f"Fire {fire_id} - Complete Progression (5-day intervals)", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    
    # Adjust layout for colorbar WITHOUT using tight_layout
    # fig.tight_layout()  # Comment out or remove this line
    fig.subplots_adjust(bottom=0.15, right=0.85)  # Make room for both colorbars
    fig.savefig(os.path.join(folder, f"{folder}_progression.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    fp = "Largefire/LargeFires_2012-2020.gpkg"
    layer = "perimeter"
    print("layers:", fiona.listlayers(fp))
    print("using layer:", layer)
    gdf = gpd.read_file(fp, layer=layer)
    print("columns:", list(gdf.columns))

    # compute areas
    gdf_area = gdf.to_crs(epsg=5070)
    gdf_area["area_m2"] = gdf_area.geometry.area
    total_areas = (
        gdf_area.groupby("fireID")["area_m2"]
        .sum().sort_values(ascending=False).head(5)
    )
    fire_ids = total_areas.index.tolist()

    # parallel processing
    cpu_count = os.cpu_count() or 2
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as exe:
        # iterate over the futures with a progress bar
        list(
            tqdm(
                exe.map(
                    partial(process_fire, gpkg_fp=fp, layer=layer),
                    fire_ids
                ),
                total=len(fire_ids),
                desc="Processing fires"
            )
        )

if __name__ == "__main__":
    main()