#!/usr/bin/env python3
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
# ─── Force xarray to use CuPy (GPU) ───────────────────────────────────────────────
os.environ["XARRAY_BACKEND"] = "cupy"

import glob
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

import cupy as cp
import numpy as np
import cudf
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.vectorized import contains

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# Updated paths to use Snapshot instead of Largefire
SNAPSHOT_GPKG = "Snapshot/Finalperimeter_2012-2020.gpkg"
SNAPSHOT_LAYER = "perimeter"  # Assuming the layer name is the same
SERIALIZATION_DIR = "Serialization"

WLDAS_DIR      = "WLDAS"
# match the filename pattern with a dot before D10 (e.g. 20160607.D10.nc)
WLDAS_PATTERN  = "WLDAS_NOAHMP001_DA1_*.D10.nc"

OUT_PARQUET    = "training_dataset.parquet"

# Variables to extract from WLDAS
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

# Buffer zone parameters to include data outside the burned area
BUFFER_KM = 10
# Approximate degrees per kilometer at the equator (1° ≈ 111 km)
DEG_PER_KM = 1.0 / 111.0
BUFFER_DEG = BUFFER_KM * DEG_PER_KM

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def list_wldas_files():
    files = glob.glob(os.path.join(WLDAS_DIR, WLDAS_PATTERN))
    mapping = {}
    for fp in files:
        fn = os.path.basename(fp)
        # strip extensions to isolate date: remove '.nc' then '.D10'
        name, _ = os.path.splitext(fn)          # e.g. 'WLDAS_NOAHMP001_DA1_20210503.D10'
        name_no_d10, _ = os.path.splitext(name)  # e.g. 'WLDAS_NOAHMP001_DA1_20210503'
        date_str = name_no_d10.split("_")[-1]    # '20210503'
        dt = pd.to_datetime(date_str, format="%Y%m%d").date()
        mapping[dt] = fp
    return mapping

def load_fire_footprints():
    """Load all fires from Snapshot layer with their date and geometry."""
    gdf = gpd.read_file(SNAPSHOT_GPKG, layer=SNAPSHOT_LAYER)
    if "time" in gdf.columns:
        gdf["date"] = pd.to_datetime(gdf["time"]).dt.date
    elif "date" in gdf.columns:
        gdf["date"] = pd.to_datetime(gdf["date"]).dt.date
    else:
        raise KeyError("No 'time' or 'date' column in Snapshot layer")
    return gdf[["fireID","date","geometry"]]

def get_serialization_data(fire_id, date):
    """Load serialization data for a specific fire and date if available."""
    year = date.year
    serialization_path = os.path.join(SERIALIZATION_DIR, f"{year}_Serialization", f"{fire_id}_{date.strftime('%Y%m%d')}.parquet")
    
    if os.path.exists(serialization_path):
        return pd.read_parquet(serialization_path)
    return None

def process_day(day, footprints, wldas_fp):
    """
    For each fire footprint on `day`, extract all WLDAS grid cells within the polygon
    and return a cuDF DataFrame with columns: fireID, date, lon, lat, VARS...
    Optionally incorporates serialization data when available.
    """
    # open WLDAS on GPU
    ds = xr.open_dataset(wldas_fp, chunks={"lat":500, "lon":500})[VARS]
    if "time" in ds.dims:
        ds = ds.isel(time=0).drop_vars("time")

    # coordinates as CuPy arrays
    lats = cp.asarray(ds["lat"].values)
    lons = cp.asarray(ds["lon"].values)
    lon2d, lat2d = cp.meshgrid(lons, lats)

    results = []
    for fid, poly in footprints.itertuples(index=False):
        # Check if serialization data exists for this fire and day
        serial_data = get_serialization_data(fid, day)
        if serial_data is not None:
            # If serialization data exists, use it instead of recomputing
            # Convert to cuDF
            df_gpu = cudf.DataFrame.from_pandas(serial_data)
            # Ensure the fireID and date columns are present
            if 'fireID' not in df_gpu.columns:
                df_gpu['fireID'] = fid
            if 'date' not in df_gpu.columns:
                df_gpu['date'] = day
            results.append(df_gpu)
            continue

        # If no serialization data, proceed with original method
        # buffer the footprint geometry to include surrounding area
        buffered_poly = poly.buffer(BUFFER_DEG)

        # compute index ranges based on buffered bounds
        minx, miny, maxx, maxy = buffered_poly.bounds
        lat_idx = cp.where((lats >= miny) & (lats <= maxy))[0]
        lon_idx = cp.where((lons >= minx) & (lons <= maxx))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            continue

        # subset data within buffered box
        sub = ds.isel(lat=lat_idx.get(), lon=lon_idx.get())
        sub_lon = lon2d[lat_idx[:, None], lon_idx[None, :]]
        sub_lat = lat2d[lat_idx[:, None], lon_idx[None, :]]

        # compute mask inside the buffered polygon
        mask = contains(buffered_poly, cp.asnumpy(sub_lon), cp.asnumpy(sub_lat))
        if not mask.any():
            continue

        # gather CuPy arrays for each variable
        data = {"fireID": fid, "date": day}
        cols = {
            "lon": cp.asnumpy(sub_lon[mask]),
            "lat": cp.asnumpy(sub_lat[mask]),
        }
        for var in VARS:
            arr = sub[var].data  # CuPy array slice
            cols[var] = cp.asnumpy(arr[mask])

        # build cuDF DataFrame
        n = len(cols["lon"])
        df_gpu = cudf.DataFrame({
            "fireID": cudf.Series(cp.full(n, fid, dtype=cp.int32)),
            "date":   cudf.Series([day] * n, dtype="datetime64[ns]"),
            "lon":    cudf.Series(cols["lon"]),
            "lat":    cudf.Series(cols["lat"]),
            **{var: cudf.Series(cols[var]) for var in VARS}
        })
        results.append(df_gpu)

    if results:
        return cudf.concat(results, ignore_index=True)
    # return empty cuDF
    empty = cudf.DataFrame({c: cudf.Series([], dtype="float64") for c in ["fireID","date","lon","lat"] + VARS})
    return empty

def main():
    fires = load_fire_footprints()
    wldas_map = list_wldas_files()

    # ─── DEBUG: inspect date keys for intersection ─────────────────────────────
    fire_dates = set(fires["date"])
    wldas_dates = set(wldas_map.keys())
    print(f"Fire dates range: {min(fire_dates)} → {max(fire_dates)}")
    print(f"WLDAS dates range: {min(wldas_dates)} → {max(wldas_dates)}")
    sample_fire = sorted(fire_dates)[:5]
    sample_wldas = sorted(wldas_dates)[:5]
    print(f"Sample fire dates: {sample_fire}")
    print(f"Sample WLDAS dates: {sample_wldas}")

    groups = fires.groupby("date")
    days = sorted(set(groups.groups).intersection(wldas_map.keys()))

    print(f"Processing {len(days)} days using GPU & {os.cpu_count()} CPU cores")

    tasks = [
        (day, groups.get_group(day)[["fireID","geometry"]], wldas_map[day])
        for day in days
    ]
    # unpack task parameters into parallel iterables for pickleable mapping
    days_list, footprints_list, wldas_list = zip(*tasks)

    with concurrent.futures.ProcessPoolExecutor() as exe:
        dfs = list(
            tqdm(
                exe.map(
                    process_day,
                    days_list,
                    footprints_list,
                    wldas_list
                ),
                total=len(days_list),
                desc="Processing days"
            )
        )

    # concatenate all GPU tables
    full_gpu = cudf.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(full_gpu):,}")

    # write out as parquet
    full_gpu.to_parquet(OUT_PARQUET)
    print("Saved training dataset to", OUT_PARQUET)

if __name__ == "__main__":
    main()
