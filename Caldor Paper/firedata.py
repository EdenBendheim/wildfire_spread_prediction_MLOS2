#!/usr/bin/env python3
# filepath: /Users/edenbendheim/Dropbox/wildfire_spread_prediction_MLOS2/Caldor Paper/analyze_fires.py

import geopandas as gpd
import pandas as pd
import numpy as np
from collections import defaultdict
import fiona

def main():
    fp = "Largefire/LargeFires_2012-2020.gpkg"
    
    # List available layers
    print("Available layers in geopackage:", fiona.listlayers(fp))
    
    # Read the perimeter layer (contains fire boundaries over time)
    gdf = gpd.read_file(fp, layer="perimeter")
    
    # Quick exploration of data structure
    print("\nDataset shape:", gdf.shape)
    print("\nColumns:", list(gdf.columns))
    
    # Convert time to datetime
    gdf["time"] = pd.to_datetime(gdf["time"])
    
    # Check CRS and reproject if needed - use EPSG:5070 (Albers equal-area for continental US)
    if gdf.crs != "EPSG:5070":
        print(f"Reprojecting from {gdf.crs} to EPSG:5070 for accurate area calculation")
        gdf_area = gdf.to_crs(epsg=5070)
    else:
        gdf_area = gdf
    
    # Group by fireID to get stats
    fire_stats = defaultdict(dict)
    
    # Calculate duration for each fire
    for fire_id, group in gdf.groupby("fireID"):
        # Get first and last observation
        start_time = group["time"].min()
        end_time = group["time"].max()
        duration = (end_time - start_time).total_seconds() / (60*60*24)  # in days
        
        # Save stats
        fire_stats[fire_id]["duration_days"] = duration
        fire_stats[fire_id]["start_date"] = start_time
        fire_stats[fire_id]["end_date"] = end_time
        fire_stats[fire_id]["num_observations"] = len(group)
        
        # Get corresponding geometries in equal area projection
        area_group = gdf_area[gdf_area["fireID"] == fire_id]
        
        # Check if we should use 'farea' column from dataset
        if "farea" in area_group.columns and not area_group["farea"].isna().all():
            # Native area values might be in hectares or m², determine unit
            max_farea = area_group["farea"].max()
            if max_farea > 10000:  # Likely m²
                fire_stats[fire_id]["max_area_sq_km"] = max_farea 
            else:  # Likely hectares
                fire_stats[fire_id]["max_area_sq_km"] = max_farea 
        else:
            # Calculate area using geometry in equal-area projection
            areas = area_group.geometry.area / 1_000_000  # convert m² to km²
            fire_stats[fire_id]["max_area_sq_km"] = areas.max()

    # Convert to DataFrame for easier analysis
    stats_df = pd.DataFrame.from_dict(fire_stats, orient="index")
    
    # Print summary statistics
    print("\n=== Fire Dataset Statistics ===")
    print(f"Total fires: {len(stats_df):,}")
    print(f"Average fire duration: {stats_df['duration_days'].mean():.1f} days")
    print(f"Median fire duration: {stats_df['duration_days'].median():.1f} days")
    print(f"Shortest fire: {stats_df['duration_days'].min():.1f} days")
    print(f"Longest fire: {stats_df['duration_days'].max():.1f} days")
    
    print("\nAverage area covered: {:.1f} km²".format(stats_df["max_area_sq_km"].mean()))
    print("Median area covered: {:.1f} km²".format(stats_df["max_area_sq_km"].median()))
    print("Largest fire: {:.1f} km²".format(stats_df["max_area_sq_km"].max()))
    
    print("\nFires by year:")
    year_counts = stats_df["start_date"].dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} fires")
    
    # Show top 5 largest fires
    print("\nTop 5 largest fires:")
    largest = stats_df.nlargest(5, "max_area_sq_km")
    for idx, row in largest.iterrows():
        print(f"  Fire ID {idx}: {row['max_area_sq_km']:.1f} km², lasted {row['duration_days']:.1f} days")
    
    # Show top 5 longest fires
    print("\nTop 5 longest fires:")
    longest = stats_df.nlargest(5, "duration_days")
    for idx, row in longest.iterrows():
        print(f"  Fire ID {idx}: {row['duration_days']:.1f} days, area {row['max_area_sq_km']:.1f} km²")

if __name__ == "__main__":
    main()