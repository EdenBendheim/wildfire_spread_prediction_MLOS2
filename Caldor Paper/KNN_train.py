#!/usr/bin/env python3
"""
Fire Data Compilation Script

This script compiles fire perimeter data and ground weather/vegetation data into a unified structure.
Each fire is represented as an array of daily records, where each day contains:
- Fire perimeter segments (all fire boundaries up to that date)
- Ground data for that specific day (weather, vegetation, soil data)

Data Sources:
- Fire perimeters: Largefire/LargeFires_2012-2020.gpkg
- Ground data: training_dataset.parquet
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import pickle
import os
from dataclasses import dataclass, field
from shapely.geometry import Polygon, MultiPolygon
import warnings
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from shapely.ops import unary_union
from shapely.affinity import translate, scale
from shapely.geometry import LineString, Point
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
from functools import partial
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DayGroundData:
    """Container for ground weather/vegetation data for a specific day"""
    date: date
    fire_id: int
    coordinates: List[Tuple[float, float]]  # (lat, lon) pairs
    avg_surf_temp: List[float]             # AvgSurfT_tavg
    rainfall: List[float]                  # Rainf_tavg
    vegetation_transpiration: List[float]   # TVeg_tavg
    wind_speed: List[float]                # Wind_f_tavg
    air_temp: List[float]                  # Tair_f_tavg
    air_humidity: List[float]              # Qair_f_tavg
    soil_moisture: List[float]             # SoilMoi00_10cm_tavg
    soil_temp: List[float]                 # SoilTemp00_10cm_tavg
    burn_decay: List[float] = field(default_factory=list) # New field for burn decay

    def __post_init__(self):
        """Validate that all data lists have the same length as coordinates if coordinates exist."""
        if self.coordinates: # If there are coordinates, all other lists must match its length
            expected_len = len(self.coordinates)
            data_lists_to_check = [
                self.avg_surf_temp, self.rainfall, self.vegetation_transpiration,
                self.wind_speed, self.air_temp, self.air_humidity,
                self.soil_moisture, self.soil_temp, self.burn_decay
            ]
            list_names = [ # Corresponding names for error messages
                "avg_surf_temp", "rainfall", "vegetation_transpiration",
                "wind_speed", "air_temp", "air_humidity",
                "soil_moisture", "soil_temp", "burn_decay"
            ]
            for i, data_list in enumerate(data_lists_to_check):
                if len(data_list) != expected_len:
                    raise ValueError(
                        f"Data list '{list_names[i]}' length ({len(data_list)}) "
                        f"does not match coordinates length ({expected_len}) for date {self.date}, fire {self.fire_id}."
                    )
        else: # If there are no coordinates, all other lists must also be empty
            all_other_lists = [
                self.avg_surf_temp, self.rainfall, self.vegetation_transpiration,
                self.wind_speed, self.air_temp, self.air_humidity,
                self.soil_moisture, self.soil_temp, self.burn_decay
            ]
            if any(len(lst) > 0 for lst in all_other_lists):
                raise ValueError(
                    f"Coordinates list is empty, but some other data lists are not. All should be empty for date {self.date}, fire {self.fire_id}."
                )
    
    @property
    def num_points(self) -> int:
        """Number of data points for this day"""
        return len(self.coordinates)
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all variables"""
        variables = {
            'avg_surf_temp': self.avg_surf_temp,
            'rainfall': self.rainfall,
            'vegetation_transpiration': self.vegetation_transpiration,
            'wind_speed': self.wind_speed,
            'air_temp': self.air_temp,
            'air_humidity': self.air_humidity,
            'soil_moisture': self.soil_moisture,
            'soil_temp': self.soil_temp,
            'burn_decay': self.burn_decay # Added burn_decay
        }
        
        stats = {}
        for var_name, values in variables.items():
            if values:  # Check if list is not empty
                clean_values = [v for v in values if not pd.isna(v)]
                if clean_values:
                    stats[var_name] = {
                        'mean': np.mean(clean_values),
                        'std': np.std(clean_values),
                        'min': np.min(clean_values),
                        'max': np.max(clean_values),
                        'count': len(clean_values)
                    }
                else:
                    stats[var_name] = {'count': 0}
            else:
                stats[var_name] = {'count': 0}
        
        return stats


@dataclass
class DayFireData:
    """Container for fire data for a specific day"""
    date: date
    fire_id: int
    cumulative_perimeters: List[Polygon]  # All fire perimeters up to this date
    daily_perimeter: Optional[Polygon]    # Just this day's perimeter (if any)
    ground_data: Optional[DayGroundData]  # Ground data for this day
    
    @property
    def total_area(self) -> float:
        """Calculate total fire area from cumulative perimeters"""
        if not self.cumulative_perimeters:
            return 0.0
        
        # Union all perimeters and calculate area
        try:
            if len(self.cumulative_perimeters) == 1:
                total_geom = self.cumulative_perimeters[0]
            else:
                # Create MultiPolygon and get its union
                from shapely.ops import unary_union
                total_geom = unary_union(self.cumulative_perimeters)
            
            # Area in square degrees (approximate)
            return total_geom.area if hasattr(total_geom, 'area') else 0.0
        except Exception as e:
            print(f"Warning: Could not calculate area for fire {self.fire_id} on {self.date}: {e}")
            return 0.0
    
    @property
    def has_ground_data(self) -> bool:
        """Check if this day has ground data"""
        return self.ground_data is not None and self.ground_data.num_points > 0


@dataclass
class FireDataset:
    """Container for all data related to a single fire"""
    fire_id: int
    daily_data: List[DayFireData] = field(default_factory=list)
    
    @property
    def duration_days(self) -> int:
        """Number of days this fire lasted"""
        return len(self.daily_data)
    
    @property
    def start_date(self) -> Optional[date]:
        """First date of the fire"""
        return self.daily_data[0].date if self.daily_data else None
    
    @property
    def end_date(self) -> Optional[date]:
        """Last date of the fire"""
        return self.daily_data[-1].date if self.daily_data else None
    
    @property
    def max_area(self) -> float:
        """Maximum area reached by this fire"""
        if not self.daily_data:
            return 0.0
        return max(day.total_area for day in self.daily_data)
    
    def get_day_data(self, target_date: date) -> Optional[DayFireData]:
        """Get data for a specific date"""
        for day_data in self.daily_data:
            if day_data.date == target_date:
                return day_data
        return None
    
    def get_summary(self) -> Dict:
        """Get summary information about this fire"""
        return {
            'fire_id': self.fire_id,
            'duration_days': self.duration_days,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'max_area': self.max_area,
            'days_with_ground_data': sum(1 for day in self.daily_data if day.has_ground_data),
            'total_ground_data_points': sum(day.ground_data.num_points for day in self.daily_data if day.has_ground_data)
        }


class FireDataCompiler:
    """Main class for compiling fire perimeter and ground data"""
    
    def __init__(self, 
                 perimeter_file: str = "Largefire/LargeFires_2012-2020.gpkg",
                 ground_data_file: str = "training_dataset.parquet"):
        self.perimeter_file = perimeter_file
        self.ground_data_file = ground_data_file
        self.perimeter_data = None
        self.ground_data = None
        
    def load_data(self):
        """Load both perimeter and ground data"""
        print("Loading fire perimeter data...")
        self.perimeter_data = gpd.read_file(self.perimeter_file)
        self.perimeter_data['date'] = pd.to_datetime(self.perimeter_data['time']).dt.date
        print(f"Loaded {len(self.perimeter_data)} perimeter records for {len(self.perimeter_data['fireID'].unique())} fires")
        
        print("Loading ground data...")
        self.ground_data = pd.read_parquet(self.ground_data_file)
        self.ground_data['date'] = pd.to_datetime(self.ground_data['date']).dt.date
        print(f"Loaded {len(self.ground_data)} ground data records for {len(self.ground_data['fireID'].unique())} fires")
        
    def compile_fire_data(self, fire_id: int) -> Optional[FireDataset]:
        """Compile all data for a specific fire"""
        if self.perimeter_data is None or self.ground_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Get perimeter data for this fire
        fire_perimeters_df = self.perimeter_data[self.perimeter_data['fireID'] == fire_id].copy()
        fire_ground_df = self.ground_data[self.ground_data['fireID'] == fire_id].copy()
        
        if fire_perimeters_df.empty:
            print(f"Warning: No perimeter data found for fire {fire_id}")
            return None # Cannot calculate burn decay without perimeters
            
        # Sort by date
        fire_perimeters_df = fire_perimeters_df.sort_values('date')
        fire_ground_df = fire_ground_df.sort_values('date')
        
        # Get all unique dates for this fire
        all_dates = sorted(list(set(fire_perimeters_df['date'].unique()) | set(fire_ground_df['date'].unique())))
        
        daily_data_list = []
        
        # Tracks consecutive days a ground point has been in the fire.
        # Key: (lat, lon) tuple for ground data point. Value: integer count of consecutive days.
        point_burn_streaks: Dict[Tuple[float, float], int] = {}

        # This list will store the actual Polygon/MultiPolygon objects that define the fire boundary each day.
        # It accumulates *new* perimeters as they appear.
        active_perimeters_so_far: List[Polygon] = [] 
        
        # Pre-compute the unary union once per day and cache it
        current_total_fire_polygon = None
        
        # Pre-create all ground points as a GeoDataFrame for reuse
        all_ground_coords = {}
        for current_date in all_dates:
            daily_ground_records = fire_ground_df[fire_ground_df['date'] == current_date]
            if not daily_ground_records.empty:
                coords = list(zip(daily_ground_records['lat'], daily_ground_records['lon']))
                all_ground_coords[current_date] = {
                    'coords': coords,
                    'records': daily_ground_records,
                    'points_gdf': gpd.GeoDataFrame(
                        geometry=[Point(lon, lat) for lat, lon in coords],
                        crs='EPSG:4326'
                    ) if coords else None
                }
        
        for current_date in all_dates:
            # Get *new* perimeter for this specific date from fire_perimeters_df
            daily_perimeter_geom_data = fire_perimeters_df[fire_perimeters_df['date'] == current_date]
            new_perimeter_for_today = None
            
            polygon_updated = False
            if not daily_perimeter_geom_data.empty:
                geom = daily_perimeter_geom_data.iloc[0]['geometry']
                # Ensure geom is a valid Polygon or MultiPolygon before adding
                if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty and geom.is_valid:
                    active_perimeters_so_far.append(geom)
                    new_perimeter_for_today = geom 
                    polygon_updated = True
            
            # Only recompute union if polygon was updated
            if polygon_updated or current_total_fire_polygon is None:
                current_day_cumulative_perimeters_list = active_perimeters_so_far.copy()
                if current_day_cumulative_perimeters_list:
                    try:
                        # Filter out any invalid or empty geometries before union
                        valid_geoms = [g for g in current_day_cumulative_perimeters_list if g.is_valid and not g.is_empty]
                        if valid_geoms:
                            current_total_fire_polygon = unary_union(valid_geoms)
                            if current_total_fire_polygon.is_empty or not current_total_fire_polygon.is_valid:
                                current_total_fire_polygon = None
                        else: # No valid geoms to union
                            current_total_fire_polygon = None
                    except Exception as e:
                        print(f"Warning: Could not form unary_union for fire {fire_id} on {current_date}: {e}")
                        current_total_fire_polygon = None
            
            current_day_cumulative_perimeters_list = active_perimeters_so_far.copy()
            ground_data_obj = None
            
            # Process ground data if available for this date
            if current_date in all_ground_coords:
                ground_info = all_ground_coords[current_date]
                coordinates = ground_info['coords']
                daily_ground_records = ground_info['records']
                points_gdf = ground_info['points_gdf']
                
                # Fast point-in-polygon check using GeoPandas
                if current_total_fire_polygon and points_gdf is not None and len(coordinates) > 0:
                    try:
                        # Create a temporary GeoDataFrame with the fire polygon
                        fire_gdf = gpd.GeoDataFrame([1], geometry=[current_total_fire_polygon], crs='EPSG:4326')
                        
                        # Use spatial join for batch point-in-polygon
                        joined = gpd.sjoin(points_gdf, fire_gdf, how='left', predicate='within')
                        within_fire = ~joined['index_right'].isna()
                        
                    except Exception as e:
                        print(f"Warning: Spatial join failed for {current_date}, falling back to individual checks: {e}")
                        # Fallback to individual checks
                        within_fire = points_gdf.geometry.apply(lambda pt: current_total_fire_polygon.contains(pt))
                    
                    burn_decay_values_for_current_day = []
                    for i, (coord_tuple, is_within) in enumerate(zip(coordinates, within_fire)):
                        if is_within:
                            point_burn_streaks[coord_tuple] = point_burn_streaks.get(coord_tuple, 0) + 1
                            days_burning = point_burn_streaks[coord_tuple]
                            # Pre-computed decay formula for efficiency
                            if days_burning <= 10:  # Cache first 10 values
                                decay_value = sum(0.8**i for i in range(days_burning))
                            else:
                                # Use geometric series formula for large values: sum = (1 - r^n) / (1 - r)
                                decay_value = (1 - 0.8**days_burning) / (1 - 0.8)
                            burn_decay_values_for_current_day.append(decay_value)
                        else:
                            point_burn_streaks[coord_tuple] = 0 # Reset streak
                            burn_decay_values_for_current_day.append(0.0)
                else:
                    # No fire polygon or no coordinates
                    burn_decay_values_for_current_day = [0.0] * len(coordinates)
                    # Reset all streaks
                    for coord_tuple in coordinates:
                        point_burn_streaks[coord_tuple] = 0
                
                ground_data_obj = DayGroundData(
                    date=current_date,
                    fire_id=fire_id,
                    coordinates=coordinates,
                    avg_surf_temp=daily_ground_records['AvgSurfT_tavg'].tolist(),
                    rainfall=daily_ground_records['Rainf_tavg'].tolist(),
                    vegetation_transpiration=daily_ground_records['TVeg_tavg'].tolist(),
                    wind_speed=daily_ground_records['Wind_f_tavg'].tolist(),
                    air_temp=daily_ground_records['Tair_f_tavg'].tolist(),
                    air_humidity=daily_ground_records['Qair_f_tavg'].tolist(),
                    soil_moisture=daily_ground_records['SoilMoi00_10cm_tavg'].tolist(),
                    soil_temp=daily_ground_records['SoilTemp00_10cm_tavg'].tolist(),
                    burn_decay=burn_decay_values_for_current_day 
                )
            
            day_fire_data_entry = DayFireData(
                date=current_date,
                fire_id=fire_id,
                cumulative_perimeters=current_day_cumulative_perimeters_list,
                daily_perimeter=new_perimeter_for_today,
                ground_data=ground_data_obj
            )
            daily_data_list.append(day_fire_data_entry)
        
        if not daily_data_list and not fire_perimeters_df.empty :
             print(f"Warning: No daily data compiled for fire {fire_id}, though perimeter data existed. Check date alignment or processing logic.")
             return None
        elif not daily_data_list and fire_perimeters_df.empty: # This case is handled at the start
            pass


        return FireDataset(fire_id=fire_id, daily_data=daily_data_list)
    
    def compile_all_fires(self, fire_ids: Optional[List[int]] = None) -> List[FireDataset]:
        """Optimized compile data for all fires using vectorized operations"""
        if self.perimeter_data is None or self.ground_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if fire_ids is None:
            # Get intersection of fire IDs that exist in both datasets
            perimeter_fire_ids = set(self.perimeter_data['fireID'].unique())
            ground_fire_ids = set(self.ground_data['fireID'].unique())
            fire_ids = sorted(perimeter_fire_ids & ground_fire_ids)
            print(f"Found {len(fire_ids)} fires with both perimeter and ground data")
        
        # Pre-process all ground data into spatial structure once
        print("Pre-processing ground data...")
        ground_gdf = gpd.GeoDataFrame(
            self.ground_data,
            geometry=gpd.points_from_xy(self.ground_data['lon'], self.ground_data['lat']),
            crs='EPSG:4326'
        )
        
        # Group ground data by fire_id and date for efficient lookup
        ground_grouped = ground_gdf.groupby(['fireID', 'date'])
        ground_lookup = {}
        for (fire_id, date), group in ground_grouped:
            ground_lookup[(fire_id, date)] = group
        
        # Group perimeter data by fire_id and date
        perimeter_grouped = self.perimeter_data.groupby(['fireID', 'date'])
        perimeter_lookup = {}
        for (fire_id, date), group in perimeter_grouped:
            if not group.empty:
                geom = group.iloc[0]['geometry']
                if isinstance(geom, (Polygon, MultiPolygon)) and not geom.is_empty and geom.is_valid:
                    perimeter_lookup[(fire_id, date)] = geom
        
        compiled_fires = []
        failed_fires = []
        
        # Process fires in batches for memory efficiency
        batch_size = 100
        for batch_start in range(0, len(fire_ids), batch_size):
            batch_fire_ids = fire_ids[batch_start:batch_start + batch_size]
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(fire_ids)-1)//batch_size + 1}: fires {batch_start+1}-{min(batch_start+batch_size, len(fire_ids))}")
            
            for fire_id in batch_fire_ids:
                try:
                    fire_dataset = self._compile_fire_data_vectorized(fire_id, perimeter_lookup, ground_lookup)
                    if fire_dataset and fire_dataset.daily_data:
                        compiled_fires.append(fire_dataset)
                    else:
                        failed_fires.append(fire_id)
                except Exception as e:
                    failed_fires.append(fire_id)
                    print(f"  ✗ Fire {fire_id}: Error - {e}")
        
        print(f"\nCompilation complete: {len(compiled_fires)} success, {len(failed_fires)} failed")
        return compiled_fires
    
    def _compile_fire_data_vectorized(self, fire_id: int, perimeter_lookup: dict, ground_lookup: dict) -> Optional[FireDataset]:
        """Vectorized version using pre-processed lookup tables"""
        
        # Get all dates for this fire from both datasets
        perimeter_dates = [date for (fid, date) in perimeter_lookup.keys() if fid == fire_id]
        ground_dates = [date for (fid, date) in ground_lookup.keys() if fid == fire_id]
        all_dates = sorted(set(perimeter_dates + ground_dates))
        
        if not all_dates:
            return None
        
        # Pre-compute cumulative fire polygons
        cumulative_polygons = []
        perimeter_by_date = {}
        cumulative_union = None
        
        for current_date in all_dates:
            new_perimeter = perimeter_lookup.get((fire_id, current_date))
            
            if new_perimeter is not None:
                cumulative_polygons.append(new_perimeter)
                # Efficiently update cumulative union
                if cumulative_union is None:
                    cumulative_union = new_perimeter
                else:
                    try:
                        cumulative_union = cumulative_union.union(new_perimeter)
                    except Exception:
                        cumulative_union = unary_union(cumulative_polygons)
            
            perimeter_by_date[current_date] = {
                'new_perimeter': new_perimeter,
                'cumulative_polygons': cumulative_polygons.copy(),
                'cumulative_union': cumulative_union
            }
        
        # Pre-compute burn decay using vectorized operations
        point_burn_streaks = {}
        daily_data_list = []
        
        for current_date in all_dates:
            perimeter_info = perimeter_by_date[current_date]
            ground_group = ground_lookup.get((fire_id, current_date))
            
            ground_data_obj = None
            if ground_group is not None and not ground_group.empty:
                # Vectorized coordinate extraction
                coordinates = list(zip(ground_group['lat'], ground_group['lon']))
                
                # Vectorized point-in-polygon check
                if perimeter_info['cumulative_union'] and len(coordinates) > 0:
                    # Use spatial join for batch processing
                    fire_poly_gdf = gpd.GeoDataFrame([1], geometry=[perimeter_info['cumulative_union']], crs='EPSG:4326')
                    joined = gpd.sjoin(ground_group, fire_poly_gdf, how='left', predicate='within')
                    within_fire = ~joined['index_right'].isna().values
                    
                    # Vectorized burn decay calculation
                    burn_decay_values = self._calculate_burn_decay_vectorized(
                        coordinates, within_fire, point_burn_streaks
                    )
                else:
                    burn_decay_values = np.zeros(len(coordinates))
                    # Reset all streaks
                    for coord_tuple in coordinates:
                        point_burn_streaks[coord_tuple] = 0
                
                # Vectorized data extraction
                ground_data_obj = DayGroundData(
                    date=current_date,
                    fire_id=fire_id,
                    coordinates=coordinates,
                    avg_surf_temp=ground_group['AvgSurfT_tavg'].values.tolist(),
                    rainfall=ground_group['Rainf_tavg'].values.tolist(),
                    vegetation_transpiration=ground_group['TVeg_tavg'].values.tolist(),
                    wind_speed=ground_group['Wind_f_tavg'].values.tolist(),
                    air_temp=ground_group['Tair_f_tavg'].values.tolist(),
                    air_humidity=ground_group['Qair_f_tavg'].values.tolist(),
                    soil_moisture=ground_group['SoilMoi00_10cm_tavg'].values.tolist(),
                    soil_temp=ground_group['SoilTemp00_10cm_tavg'].values.tolist(),
                    burn_decay=burn_decay_values.tolist()
                )
            
            day_fire_data = DayFireData(
                date=current_date,
                fire_id=fire_id,
                cumulative_perimeters=perimeter_info['cumulative_polygons'],
                daily_perimeter=perimeter_info['new_perimeter'],
                ground_data=ground_data_obj
            )
            daily_data_list.append(day_fire_data)
        
        return FireDataset(fire_id=fire_id, daily_data=daily_data_list)
    
    def _calculate_burn_decay_vectorized(self, coordinates: List[Tuple[float, float]], 
                                       within_fire: np.ndarray, 
                                       point_burn_streaks: Dict[Tuple[float, float], int]) -> np.ndarray:
        """Vectorized burn decay calculation using numpy"""
        n_points = len(coordinates)
        burn_decay_values = np.zeros(n_points, dtype=np.float64)
        
        # Pre-compute geometric series values for common streak lengths (up to 50 days)
        max_precompute = 50
        decay_lookup = np.array([sum(0.8**i for i in range(days)) for days in range(max_precompute + 1)])
        
        for i, (coord_tuple, is_within) in enumerate(zip(coordinates, within_fire)):
            if is_within:
                point_burn_streaks[coord_tuple] = point_burn_streaks.get(coord_tuple, 0) + 1
                days_burning = point_burn_streaks[coord_tuple]
                
                # Use pre-computed values for common cases, formula for longer streaks
                if days_burning <= max_precompute:
                    burn_decay_values[i] = decay_lookup[days_burning]
                else:
                    # Geometric series formula: (1 - r^n) / (1 - r)
                    burn_decay_values[i] = (1 - 0.8**days_burning) / 0.2
            else:
                point_burn_streaks[coord_tuple] = 0
                burn_decay_values[i] = 0.0
        
        return burn_decay_values
    
    def save_compiled_data(self, compiled_fires: List[FireDataset], filename: str = "compiled_fire_data.pkl"):
        """Save compiled fire data to a pickle file"""
        print(f"Saving compiled data to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(compiled_fires, f)
        print(f"Saved {len(compiled_fires)} fires to {filename}")
        
        # Also save a summary
        summary_data = []
        for fire in compiled_fires:
            summary_data.append(fire.get_summary())
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = filename.replace('.pkl', '_summary.csv')
        summary_df.to_csv(summary_filename, index=False)
        print(f"Saved summary to {summary_filename}")
    
    @staticmethod
    def load_compiled_data(filename: str = "compiled_fire_data.pkl") -> List[FireDataset]:
        """Load compiled fire data from a pickle file"""
        print(f"Loading compiled data from {filename}...")
        with open(filename, 'rb') as f:
            compiled_fires = pickle.load(f)
        print(f"Loaded {len(compiled_fires)} fires from {filename}")
        return compiled_fires


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


def visualize_fire_progression(fire_dataset: FireDataset, output_dir: str = None, interval_days: int = 5):
    """
    Visualize fire progression with TVeg background and satellite imagery.
    
    Parameters:
    -----------
    fire_dataset : FireDataset
        The compiled fire dataset to visualize
    output_dir : str, optional
        Directory to save images. If None, uses 'fire_{fire_id}_visualization'
    interval_days : int
        Show progression every N days (default: 5)
    """
    fire_id = fire_dataset.fire_id
    
    if output_dir is None:
        output_dir = f"fire_{fire_id}_visualization"
    
    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    
    # Get all days with data and select every N days
    all_days = fire_dataset.daily_data
    
    if len(all_days) <= 1:
        interval_days_data = all_days
    else:
        # Always include first day
        interval_days_data = [all_days[0]]
        # Add every Nth day
        for i in range(interval_days, len(all_days), interval_days):
            interval_days_data.append(all_days[i])
        # Always include last day if not already included
        if all_days[-1] not in interval_days_data:
            interval_days_data.append(all_days[-1])
    
    print(f"Fire {fire_id}: Visualizing {len(interval_days_data)} days at {interval_days}-day intervals")
    
    # Define fire-themed gradient colormap
    fire_cmap = LinearSegmentedColormap.from_list('fire_gradient', 
                                                ['#FFFF00', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#8B0000'])
    
    # Calculate overall bounds for the fire (for the final summary image)
    all_perimeters = []
    for day_data in all_days:
        if day_data.cumulative_perimeters:
            all_perimeters.extend(day_data.cumulative_perimeters)
    
    if not all_perimeters:
        print(f"Warning: No perimeter data found for fire {fire_id}")
        return
    
    # Convert to GeoDataFrame to get overall bounds for summary
    from shapely.ops import unary_union
    overall_geom = unary_union(all_perimeters)
    summary_minx, summary_miny, summary_maxx, summary_maxy = overall_geom.bounds
    
    # Add buffer around overall bounds for summary
    summary_width = summary_maxx - summary_minx
    summary_height = summary_maxy - summary_miny
    summary_buffer_x = summary_width * 0.15  # 15% buffer
    summary_buffer_y = summary_height * 0.15
    
    summary_minx -= summary_buffer_x
    summary_maxx += summary_buffer_x
    summary_miny -= summary_buffer_y
    summary_maxy += summary_buffer_y
    
    print(f"Fire {fire_id} overall bounds: ({summary_minx:.5f}, {summary_miny:.5f}) to ({summary_maxx:.5f}, {summary_maxy:.5f})")
    
    # Create visualizations for each interval day
    for i, current_day_data in enumerate(interval_days_data):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calculate dynamic bounds for this specific day
        days_to_plot = interval_days_data[:i+1]
        current_perimeters = []
        for day_data in days_to_plot:
            if day_data.cumulative_perimeters:
                current_perimeters.extend(day_data.cumulative_perimeters)
        
        if current_perimeters:
            # Get bounds for fire up to current day
            current_geom = unary_union(current_perimeters)
            current_minx, current_miny, current_maxx, current_maxy = current_geom.bounds
            
            # Add dynamic buffer based on fire size
            current_width = current_maxx - current_minx
            current_height = current_maxy - current_miny
            
            # Use larger buffer for smaller fires, smaller buffer for larger fires
            min_buffer = 0.01  # Minimum buffer in degrees
            max_buffer_ratio = 0.3  # Maximum buffer as ratio of fire size
            
            if current_width < 0.01 or current_height < 0.01:  # Very small fire
                buffer_x = buffer_y = min_buffer
            else:
                buffer_x = max(min_buffer, current_width * max_buffer_ratio)
                buffer_y = max(min_buffer, current_height * max_buffer_ratio)
            
            current_minx -= buffer_x
            current_maxx += buffer_x
            current_miny -= buffer_y
            current_maxy += buffer_y
        else:
            # Fallback if no perimeters available
            current_minx, current_miny, current_maxx, current_maxy = summary_minx, summary_miny, summary_maxx, summary_maxy
        
        # Set extent for current day
        ax.set_xlim(current_minx, current_maxx)
        ax.set_ylim(current_miny, current_maxy)
        
        # Add satellite basemap
        add_satellite_basemap(ax, (current_minx, current_miny, current_maxx, current_maxy))
        
        # Plot TVeg background data if available
        if current_day_data.ground_data and current_day_data.ground_data.vegetation_transpiration:
            # Create points from coordinates and TVeg data
            coords = current_day_data.ground_data.coordinates
            tveg_values = current_day_data.ground_data.vegetation_transpiration
            
            # Filter points within visible bounds
            visible_coords = []
            visible_tveg = []
            for coord, tveg in zip(coords, tveg_values):
                lat, lon = coord
                if (current_minx <= lon <= current_maxx and 
                    current_miny <= lat <= current_maxy and 
                    not pd.isna(tveg)):
                    visible_coords.append((lon, lat))  # Note: lon, lat for plotting
                    visible_tveg.append(tveg)
            
            if visible_coords and visible_tveg:
                # Plot TVeg as scatter points
                lons, lats = zip(*visible_coords)
                scatter = ax.scatter(lons, lats, c=visible_tveg, cmap='coolwarm', 
                                   s=10, alpha=0.7, zorder=5)
                
                # Add colorbar for TVeg
                tveg_cbar = plt.colorbar(scatter, ax=ax, pad=0.01, location='right')
                tveg_cbar.set_label("Vegetation Transpiration", fontsize=10)
        
        # Plot fire perimeters up to current day
        days_to_plot = interval_days_data[:i+1]
        num_days = len(days_to_plot)
        
        for j, day_data in enumerate(days_to_plot):
            if day_data.daily_perimeter:
                # Create GeoDataFrame for plotting
                gdf = gpd.GeoDataFrame([{'geometry': day_data.daily_perimeter}], crs='EPSG:4326')
                
                # Use gradient based on time progression
                color_val = j / max(1, num_days - 1)
                perimeter_color = fire_cmap(color_val)
                
                gdf.plot(ax=ax, facecolor='none', edgecolor=perimeter_color, 
                        linewidth=2.5, zorder=j+10)
        
        # Add horizontal colorbar for fire progression
        sm = ScalarMappable(cmap=fire_cmap, norm=Normalize(0, 1))
        sm.set_array([])
        
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        
        # Add date labels to colorbar
        if len(days_to_plot) > 1:
            start_date = days_to_plot[0].date.strftime("%Y-%m-%d")
            end_date = days_to_plot[-1].date.strftime("%Y-%m-%d")
            cbar.ax.set_xticks([0, 1])
            cbar.ax.set_xticklabels([start_date, end_date])
            cbar.ax.set_xlabel('Fire Progression Timeline', labelpad=5)
        else:
            date_str = days_to_plot[0].date.strftime("%Y-%m-%d")
            cbar.ax.set_xticks([0.5])
            cbar.ax.set_xticklabels([date_str])
        
        # Set title and labels
        current_date_str = current_day_data.date.strftime("%Y-%m-%d")
        ax.set_title(f"Fire {fire_id} Progression Through {current_date_str}", fontsize=14)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_aspect("equal", adjustable="datalim")
        
        # Save figure
        fig.subplots_adjust(bottom=0.15, right=0.85)
        fig.savefig(os.path.join(daily_dir, f"interval_{i:02d}_{current_date_str}.png"), 
                   dpi=200, bbox_inches="tight")
        plt.close(fig)
    
    # Create final summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Set extent
    ax.set_xlim(summary_minx, summary_maxx)
    ax.set_ylim(summary_miny, summary_maxy)
    
    # Add satellite basemap
    add_satellite_basemap(ax, (summary_minx, summary_miny, summary_maxx, summary_maxy))
    
    # Plot TVeg from last day if available
    if interval_days_data[-1].ground_data and interval_days_data[-1].ground_data.vegetation_transpiration:
        coords = interval_days_data[-1].ground_data.coordinates
        tveg_values = interval_days_data[-1].ground_data.vegetation_transpiration
        
        visible_coords = []
        visible_tveg = []
        for coord, tveg in zip(coords, tveg_values):
            lat, lon = coord
            if (summary_minx <= lon <= summary_maxx and 
                summary_miny <= lat <= summary_maxy and 
                not pd.isna(tveg)):
                visible_coords.append((lon, lat))
                visible_tveg.append(tveg)
        
        if visible_coords and visible_tveg:
            lons, lats = zip(*visible_coords)
            scatter = ax.scatter(lons, lats, c=visible_tveg, cmap='coolwarm', 
                               s=10, alpha=0.7, zorder=5)
            tveg_cbar = plt.colorbar(scatter, ax=ax, pad=0.01, location='right')
            tveg_cbar.set_label("Vegetation Transpiration", fontsize=10)
    
    # Plot all interval fire perimeters
    for j, day_data in enumerate(interval_days_data):
        if day_data.daily_perimeter:
            gdf = gpd.GeoDataFrame([{'geometry': day_data.daily_perimeter}], crs='EPSG:4326')
            
            color_val = j / max(1, len(interval_days_data) - 1)
            perimeter_color = fire_cmap(color_val)
            
            gdf.plot(ax=ax, facecolor='none', edgecolor=perimeter_color, 
                    linewidth=2.5, zorder=j+10)
    
    # Add colorbar for complete progression
    sm = ScalarMappable(cmap=fire_cmap, norm=Normalize(0, 1))
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    if len(interval_days_data) > 1:
        start_date = interval_days_data[0].date.strftime("%Y-%m-%d")
        end_date = interval_days_data[-1].date.strftime("%Y-%m-%d")
        cbar.ax.set_xticks([0, 1])
        cbar.ax.set_xticklabels([start_date, end_date])
        cbar.ax.set_xlabel('Fire Progression Timeline', labelpad=5)
    else:
        date_str = interval_days_data[0].date.strftime("%Y-%m-%d")
        cbar.ax.set_xticks([0.5])
        cbar.ax.set_xticklabels([date_str])
    
    ax.set_title(f"Fire {fire_id} - Complete Progression ({interval_days}-day intervals)", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    
    fig.subplots_adjust(bottom=0.15, right=0.85)
    fig.savefig(os.path.join(output_dir, f"fire_{fire_id}_complete_progression.png"), 
               dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Fire {fire_id} visualization complete. Saved to {output_dir}")


def visualize_fire_by_id(fire_id: int, interval_days: int = 5, 
                        output_dir: str = None,
                        perimeter_file: str = "Largefire/LargeFires_2012-2020.gpkg",
                        ground_data_file: str = "training_dataset.parquet"):
    """
    Easy-to-use function to visualize a fire by its ID.
    
    Parameters:
    -----------
    fire_id : int
        The ID of the fire to visualize
    interval_days : int
        Show progression every N days (default: 5)
    output_dir : str, optional
        Directory to save images. If None, uses 'fire_{fire_id}_visualization'
    perimeter_file : str
        Path to the fire perimeter data file
    ground_data_file : str
        Path to the ground data file
        
    Returns:
    --------
    FireDataset or None
        The compiled fire dataset if successful, None otherwise
    """
    print(f"=== Visualizing Fire {fire_id} ===")
    
    # Initialize compiler
    compiler = FireDataCompiler(perimeter_file, ground_data_file)
    
    # Load data
    print("Loading data...")
    compiler.load_data()
    
    # Compile the specific fire
    print(f"Compiling fire {fire_id}...")
    fire_dataset = compiler.compile_fire_data(fire_id)
    
    if fire_dataset:
        summary = fire_dataset.get_summary()
        print(f"\nFire {summary['fire_id']}:")
        print(f"  Duration: {summary['duration_days']} days ({summary['start_date']} to {summary['end_date']})")
        print(f"  Max area: {summary['max_area']:.6f}")
        print(f"  Days with ground data: {summary['days_with_ground_data']}")
        print(f"  Total ground data points: {summary['total_ground_data_points']}")
        
        # Visualize the fire progression
        print(f"\nCreating visualization...")
        visualize_fire_progression(fire_dataset, output_dir=output_dir, interval_days=interval_days)
        
        output_dir_name = output_dir if output_dir else f"fire_{fire_id}_visualization"
        print(f"Visualization complete! Check the '{output_dir_name}' directory.")
        return fire_dataset
    else:
        print(f"Fire {fire_id} not found or failed to compile")
        return None


def extract_spatial_grid_features_batched(fires: List[FireDataset], grid_size: int = 9, 
                                         batch_size: int = 10, use_parallel: bool = True,
                                         n_jobs: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient batch processing for fire spread prediction feature extraction.
    Only considers points with burn_decay=0 in current day to predict fire spread.
    
    Parameters:
    -----------
    fires : List[FireDataset]
        List of compiled fire datasets
    grid_size : int
        Size of the spatial grid (default: 9 for 9x9)
    batch_size : int
        Number of fires to process at once (default: 10)
    use_parallel : bool
        Whether to use parallel processing (default: True)
    n_jobs : int
        Number of parallel jobs (default: number of CPU cores)
    
    Returns:
    --------
    X : np.ndarray
        Feature matrix (9x9 grids with 8 variables = 648 features per point)
    y : np.ndarray
        Binary target vector (1 = fire spread occurred, 0 = no spread)
    """
    print(f"Processing {len(fires)} fires in batches of {batch_size}")
    
    all_features = []
    all_targets = []
    total_points = 0
    
    for batch_start in range(0, len(fires), batch_size):
        batch_end = min(batch_start + batch_size, len(fires))
        batch_fires = fires[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(fires)-1)//batch_size + 1}: "
              f"fires {batch_start+1}-{batch_end}")
        
        # Process this batch
        X_batch, y_batch = extract_spatial_grid_features(
            batch_fires, grid_size, use_parallel, n_jobs
        )
        
        if len(X_batch) > 0:
            all_features.append(X_batch)
            all_targets.append(y_batch)
            total_points += len(X_batch)
            print(f"Batch {batch_start//batch_size + 1} yielded {len(X_batch)} samples")
    
    if len(all_features) == 0:
        return np.array([]), np.array([])
    
    # Concatenate all batches
    print("Combining all batches...")
    X = np.vstack(all_features)
    y = np.hstack(all_targets)
    
    print(f"Total features extracted: {total_points}")
    
    return X, y


def extract_spatial_grid_features(fires: List[FireDataset], grid_size: int = 9, 
                                 use_parallel: bool = True, n_jobs: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 9x9 spatial grid features for fire spread prediction with parallel processing.
    Only considers points with burn_decay=0 in current day to predict fire spread.
    
    Parameters:
    -----------
    fires : List[FireDataset]
        List of compiled fire datasets
    grid_size : int
        Size of the spatial grid (default: 9 for 9x9)
    use_parallel : bool
        Whether to use parallel processing (default: True)
    n_jobs : int
        Number of parallel jobs (default: number of CPU cores)
    
    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
        where n_features = grid_size^2 * 8 (8 variables per grid cell)
        Only includes points that had burn_decay=0 in current day
    y : np.ndarray
        Binary target vector (1 if fire spread occurred: burn_decay 0->1, 0 otherwise)
    """
    print(f"Extracting {grid_size}x{grid_size} spatial grid features for fire spread prediction...")
    print("Only considering points with burn_decay=0 in current day")
    
    # All ground data variables
    variable_names = [
        'avg_surf_temp', 'rainfall', 'vegetation_transpiration', 'wind_speed',
        'air_temp', 'air_humidity', 'soil_moisture', 'soil_temp'
    ]
    
    # Prepare data for processing
    fire_day_pairs = []
    for fire_idx, fire in enumerate(fires):
        for day_idx in range(len(fire.daily_data) - 1):  # Exclude last day
            current_day = fire.daily_data[day_idx]
            next_day = fire.daily_data[day_idx + 1]
            
            # Skip if either day lacks ground data
            if (current_day.has_ground_data and next_day.has_ground_data and
                len(current_day.ground_data.coordinates) > 0 and
                len(next_day.ground_data.coordinates) > 0):
                fire_day_pairs.append((fire_idx, fire, day_idx, current_day, next_day))
    
    print(f"Found {len(fire_day_pairs)} fire-day pairs to process")
    
    if use_parallel and len(fire_day_pairs) > 1:
        n_jobs = n_jobs or min(cpu_count(), len(fire_day_pairs))
        print(f"Using {n_jobs} parallel processes")
        
        # Create partial function with fixed parameters
        process_func = partial(
            process_fire_day_pair_optimized,
            grid_size=grid_size,
            variable_names=variable_names
        )
        
        # Process in parallel
        with Pool(n_jobs) as pool:
            results = pool.map(process_func, fire_day_pairs)
    else:
        print("Using sequential processing")
        results = []
        for i, fire_day_pair in enumerate(fire_day_pairs):
            if i % 10 == 0:
                print(f"Processing pair {i+1}/{len(fire_day_pairs)}")
            result = process_fire_day_pair_optimized(fire_day_pair, grid_size, variable_names)
            results.append(result)
    
    # Combine results
    all_features = []
    all_targets = []
    total_points_processed = 0
    
    for features, targets in results:
        if len(features) > 0:
            all_features.extend(features)
            all_targets.extend(targets)
            total_points_processed += len(features)
    
    if len(all_features) == 0:
        print("No features extracted!")
        return np.array([]), np.array([])
    
    X = np.array(all_features)
    y = np.array(all_targets)
    
    print(f"Extracted features for {total_points_processed} data points")
    print(f"Positive targets: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    return X, y


def process_fire_day_pair_optimized(fire_day_pair, grid_size, variable_names):
    """
    Process a single fire-day pair to extract features for fire spread prediction.
    Only considers points with burn_decay=0 in current day (outside burned areas).
    
    Returns:
    --------
    tuple: (features_list, targets_list)
    """
    fire_idx, fire, day_idx, current_day, next_day = fire_day_pair
    
    current_coords = np.array(current_day.ground_data.coordinates)
    current_burn_decay = np.array(current_day.ground_data.burn_decay)
    next_coords = np.array(next_day.ground_data.coordinates)
    next_burn_decay = np.array(next_day.ground_data.burn_decay)
    
    # Create spatial index for current day coordinates
    tree = KDTree(current_coords)
    
    # Pre-extract all variable data for current day
    current_vars = {}
    for var_name in variable_names:
        current_vars[var_name] = np.array(getattr(current_day.ground_data, var_name))
    
    features_list = []
    targets_list = []
    
    # Vectorized processing where possible
    grid_spacing = 0.009  # roughly 1km at mid-latitudes
    half_size = grid_size // 2
    
    # Process each target point
    for target_idx, (target_lat, target_lon) in enumerate(next_coords):
        try:
            # First check if this point had burn_decay=0 in current day
            # Find closest point in current day to this target location
            distances, indices = tree.query([[target_lat, target_lon]], k=1, distance_upper_bound=grid_spacing * 0.5)
            
            # Skip if no nearby point found or if the nearby point has burn_decay > 0
            if np.isinf(distances[0]) or current_burn_decay[indices[0]] > 0:
                continue
            
            # Extract grid features using optimized method
            grid_features = extract_grid_around_point_vectorized(
                target_lat, target_lon, 
                current_vars, tree, current_coords,
                grid_size, variable_names, grid_spacing, half_size
            )
            
            if grid_features is not None:
                # Create binary target (fire spread: 0 -> >0)
                target_burn_decay = next_burn_decay[target_idx]
                binary_target = 1 if target_burn_decay > 0 else 0
                
                features_list.append(grid_features)
                targets_list.append(binary_target)
                
        except Exception as e:
            continue  # Skip problematic points
    
    return features_list, targets_list


def extract_grid_around_point_vectorized(target_lat: float, target_lon: float,
                                        current_vars: Dict[str, np.ndarray],
                                        tree: KDTree, coord_array: np.ndarray,
                                        grid_size: int, variable_names: List[str],
                                        grid_spacing: float, half_size: int) -> Optional[np.ndarray]:
    """
    Optimized vectorized grid extraction around a target point.
    """
    # Generate all grid coordinates at once
    i_indices = np.arange(grid_size)
    j_indices = np.arange(grid_size)
    ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Calculate all grid cell coordinates vectorized
    lat_offsets = (ii - half_size) * grid_spacing
    lon_offsets = (jj - half_size) * grid_spacing
    
    cell_lats = target_lat + lat_offsets.flatten()
    cell_lons = target_lon + lon_offsets.flatten()
    
    # Query all points at once
    grid_coords = np.column_stack([cell_lats, cell_lons])
    distances, indices = tree.query(grid_coords, k=1, distance_upper_bound=grid_spacing * 1.5)
    
    # Initialize output grid
    grid_features = np.full((grid_size, grid_size, len(variable_names)), np.nan)
    
    # Fill grid efficiently
    valid_mask = ~np.isinf(distances)
    valid_indices = indices[valid_mask]
    valid_positions = np.where(valid_mask)[0]
    
    for pos_idx, data_idx in zip(valid_positions, valid_indices):
        i = pos_idx // grid_size
        j = pos_idx % grid_size
        
        for var_idx, var_name in enumerate(variable_names):
            value = current_vars[var_name][data_idx]
            if not np.isnan(value):
                grid_features[i, j, var_idx] = value
    
    # Check data sufficiency
    total_cells = grid_size * grid_size * len(variable_names)
    filled_cells = np.sum(~np.isnan(grid_features))
    
    if filled_cells < total_cells * 0.25:
        return None
    
    # Fill remaining NaN values efficiently
    for var_idx in range(len(variable_names)):
        var_slice = grid_features[:, :, var_idx]
        if np.all(np.isnan(var_slice)):
            # Use global mean for this variable
            var_data = current_vars[variable_names[var_idx]]
            clean_values = var_data[~np.isnan(var_data)]
            if len(clean_values) > 0:
                grid_features[:, :, var_idx] = np.mean(clean_values)
            else:
                grid_features[:, :, var_idx] = 0.0
        else:
            # Use local mean
            var_mean = np.nanmean(var_slice)
            if not np.isnan(var_mean):
                var_slice[np.isnan(var_slice)] = var_mean
            else:
                var_slice[np.isnan(var_slice)] = 0.0
    
    return grid_features.flatten()


def extract_grid_around_point(target_lat: float, target_lon: float,
                            ground_data: 'DayGroundData',
                            tree: KDTree, coord_array: np.ndarray,
                            grid_size: int, variable_names: List[str]) -> Optional[np.ndarray]:
    """
    Extract grid features around a target point.
    
    Parameters:
    -----------
    target_lat, target_lon : float
        Coordinates of the target point (center of grid)
    ground_data : DayGroundData
        Ground data for the day
    tree : KDTree
        Spatial index for coordinates
    coord_array : np.ndarray
        Array of coordinates for KDTree
    grid_size : int
        Size of the grid (e.g., 9 for 9x9)
    variable_names : List[str]
        Names of variables to extract
    
    Returns:
    --------
    np.ndarray or None
        Flattened grid features or None if insufficient data
    """
    # Define grid spacing (in degrees - approximately 1km spacing)
    grid_spacing = 0.009  # roughly 1km at mid-latitudes
    half_size = grid_size // 2
    
    # Initialize grid for each variable
    grid_features = np.full((grid_size, grid_size, len(variable_names)), np.nan)
    
    # Get variable data arrays
    variable_data = {}
    for var_name in variable_names:
        variable_data[var_name] = getattr(ground_data, var_name)
    
    # Fill grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate grid cell center coordinates
            lat_offset = (i - half_size) * grid_spacing
            lon_offset = (j - half_size) * grid_spacing
            
            cell_lat = target_lat + lat_offset
            cell_lon = target_lon + lon_offset
            
            # Find nearest neighbor within reasonable distance
            try:
                distances, indices = tree.query([cell_lat, cell_lon], k=1, distance_upper_bound=grid_spacing * 1.5)
                
                if not np.isinf(distances):  # Found a neighbor
                    neighbor_idx = indices
                    
                    # Extract all variables for this neighbor
                    for var_idx, var_name in enumerate(variable_names):
                        value = variable_data[var_name][neighbor_idx]
                        if not pd.isna(value):
                            grid_features[i, j, var_idx] = value
                            
            except Exception as e:
                continue
    
    # Check if we have sufficient data (at least 25% of grid cells filled)
    total_cells = grid_size * grid_size * len(variable_names)
    filled_cells = np.sum(~np.isnan(grid_features))
    
    if filled_cells < total_cells * 0.25:
        return None
    
    # Replace remaining NaN values with mean of available values for each variable
    for var_idx in range(len(variable_names)):
        var_slice = grid_features[:, :, var_idx]
        if np.all(np.isnan(var_slice)):
            # If all NaN for this variable, use global mean
            var_data = variable_data[variable_names[var_idx]]
            clean_values = [v for v in var_data if not pd.isna(v)]
            if clean_values:
                grid_features[:, :, var_idx] = np.mean(clean_values)
            else:
                grid_features[:, :, var_idx] = 0.0
        else:
            # Use mean of available values in this grid
            var_mean = np.nanmean(var_slice)
            if not np.isnan(var_mean):
                var_slice[np.isnan(var_slice)] = var_mean
            else:
                var_slice[np.isnan(var_slice)] = 0.0
    
    # Flatten to 1D feature vector
    return grid_features.flatten()


def train_svm_classifier(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[SVC, StandardScaler]:
    """
    Train SVM classifier on spatial grid features with memory optimization.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    
    Returns:
    --------
    svm_model : SVC
        Trained SVM classifier
    scaler : StandardScaler
        Fitted feature scaler
    """
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training SVM...")
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # For large datasets, use a subset for initial training
    if len(X_train_scaled) > 10000:
        print("Large dataset detected, using stratified subset for faster training")
        from sklearn.model_selection import train_test_split
        X_subset, _, y_subset, _ = train_test_split(
            X_train_scaled, y_train, 
            train_size=10000, 
            stratify=y_train, 
            random_state=42
        )
        X_train_scaled = X_subset
        y_train = y_subset
        print(f"Using subset of size: {X_train_scaled.shape}")
    
    # Use balanced class weights to handle imbalanced data
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        cache_size=1000  # Increase cache size for better performance
    )
    
    svm_model.fit(X_train_scaled, y_train)
    
    print(f"SVM training complete. Support vectors: {svm_model.n_support_}")
    
    return svm_model, scaler


def evaluate_svm_model(svm_model: SVC, scaler: StandardScaler, 
                      X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate trained SVM model on test data.
    
    Parameters:
    -----------
    svm_model : SVC
        Trained SVM classifier
    scaler : StandardScaler
        Fitted feature scaler
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    """
    print("Scaling test features...")
    X_test_scaled = scaler.transform(X_test)
    
    print("Making predictions...")
    y_pred = svm_model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== SVM Model Evaluation ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test samples: {len(y_test)}")
    print(f"Positive predictions: {np.sum(y_pred)} ({np.mean(y_pred)*100:.1f}%)")
    print(f"Actual positives: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Burn', 'Burn']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    import time
    start_time = time.time()
    
    fire_id_to_inspect = 2563
    date_to_inspect_str = "2018-09-13"
    target_date = datetime.strptime(date_to_inspect_str, "%Y-%m-%d").date()

    print(f"--- Fire Spread Prediction System ---")
    print(f"Predicting fire spread from unburned areas (burn_decay=0 -> burn_decay>0)")
    print(f"Processing Fire {fire_id_to_inspect} for analysis")
    print(f"Target date for inspection: {target_date.strftime('%Y-%m-%d')}")
    print(f"Available CPU cores: {cpu_count()}")

    compiler = FireDataCompiler()
    try:
        print("Loading data...")
        load_start = time.time()
        compiler.load_data()
        print(f"Data loading took {time.time() - load_start:.2f} seconds")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Compile all fires in the dataset
    print("Compiling fires...")
    compile_start = time.time()
    compiled_fires = compiler.compile_all_fires()
    print(f"Fire compilation took {time.time() - compile_start:.2f} seconds")
    print(f"Total fires compiled: {len(compiled_fires)}")

    # For faster testing, limit the number of fires
    max_fires_for_testing = 50  # Adjust this number based on your needs
    if len(compiled_fires) > max_fires_for_testing:
        print(f"Limiting to first {max_fires_for_testing} fires for faster processing")
        compiled_fires = compiled_fires[:max_fires_for_testing]

    # Split the fires into training and testing sets
    train_fires, test_fires = train_test_split(compiled_fires, test_size=0.2, random_state=42)
    
    print(f"Training fires: {len(train_fires)}")
    print(f"Test fires: {len(test_fires)}")

    if len(train_fires) > 0 and len(train_fires[0].daily_data) > 0:
        coords_of_fire_1 = train_fires[0].daily_data[0].ground_data.coordinates
        print(f"Sample coordinates from first fire: {len(coords_of_fire_1)} points")
    
    # Extract spatial grid features for fire spread prediction (only burn_decay=0 points)
    print("Extracting spatial grid features for fire spread training...")
    print("(Only considering points with burn_decay=0 in current day)")
    feature_start = time.time()
    
    # Use batched processing for larger datasets
    if len(train_fires) > 20:
        print("Using batched processing for large dataset")
        X_train, y_train = extract_spatial_grid_features_batched(
            train_fires, batch_size=10, use_parallel=True
        )
    else:
        X_train, y_train = extract_spatial_grid_features(train_fires, use_parallel=True)
    
    feature_time = time.time() - feature_start
    print(f"Training feature extraction took {feature_time:.2f} seconds")
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Fire spread cases (burn_decay 0->1): {np.sum(y_train)} out of {len(y_train)} total points")
    
    if X_train.size == 0:
        print("Error: No training features extracted!")
        return
    
    # Extract spatial grid features for testing
    print("Extracting spatial grid features for fire spread testing...")
    test_feature_start = time.time()
    
    if len(test_fires) > 20:
        print("Using batched processing for large test dataset")
        X_test, y_test = extract_spatial_grid_features_batched(
            test_fires, batch_size=10, use_parallel=True
        )
    else:
        X_test, y_test = extract_spatial_grid_features(test_fires, use_parallel=True)
    test_feature_time = time.time() - test_feature_start
    print(f"Test feature extraction took {test_feature_time:.2f} seconds")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Test fire spread cases (burn_decay 0->1): {np.sum(y_test)} out of {len(y_test)} total points")
    
    if X_test.size == 0:
        print("Warning: No test features extracted! Using a subset of training data for testing.")
        # Split training data for testing
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
    # Train SVM classifier for fire spread prediction
    print("Training SVM classifier for fire spread prediction...")
    train_start = time.time()
    svm_model, scaler = train_svm_classifier(X_train, y_train)
    train_time = time.time() - train_start
    print(f"SVM training took {train_time:.2f} seconds")
    
    # Evaluate fire spread prediction model
    print("Evaluating fire spread prediction model...")
    eval_start = time.time()
    evaluate_svm_model(svm_model, scaler, X_test, y_test)
    eval_time = time.time() - eval_start
    print(f"Model evaluation took {eval_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\n=== Fire Spread Prediction Performance Summary ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Data loading: {load_start and (compile_start - load_start):.2f}s")
    print(f"Fire compilation: {compile_start and (feature_start - compile_start):.2f}s") 
    print(f"Feature extraction: {feature_time + test_feature_time:.2f}s")
    print(f"Model training: {train_time:.2f}s")
    print(f"Model evaluation: {eval_time:.2f}s")
    print(f"System focuses on predicting fire spread from unburned areas only")
                



    

    

if __name__ == "__main__":
    main()

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from shapely.ops import unary_union
from shapely.affinity import translate, scale
from shapely.geometry import LineString, Point
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def translate_multipolygon_for_ml(polygon, method='radial', n_params=36):
    """
    Convert polygon to numerical features for ML (stub function).
    
    This is a simplified version for compatibility with existing code.
    """
    if method == 'radial':
        return _polygon_to_radial_features(polygon, n_params)
    elif method == 'coordinates':
        return _polygon_to_coordinate_features(polygon, n_params)
    else:
        # Default: return simple centroid + area features
        centroid = polygon.centroid
        return np.array([centroid.x, centroid.y, polygon.area])


def _polygon_to_radial_features(polygon, n_radial=36):
    """
    Convert polygon to centroid + radial distances representation.
    
    This method represents a polygon using its centroid coordinates plus distances
    from the centroid to the polygon boundary at regular angular intervals.
    This is particularly effective for fire perimeters as it captures shape
    information while being robust to polygon complexity.
    """
    try:
        centroid = polygon.centroid
        angles = np.linspace(0, 2*np.pi, n_radial, endpoint=False)
        
        distances = []
        for angle in angles:
            # Cast ray from centroid at this angle
            ray_length = 1000  # Large number to ensure intersection with boundary
            ray_end_x = centroid.x + ray_length * np.cos(angle)
            ray_end_y = centroid.y + ray_length * np.sin(angle)
            ray = LineString([(centroid.x, centroid.y), (ray_end_x, ray_end_y)])
            
            try:
                # Find intersection with polygon boundary
                intersection = polygon.boundary.intersection(ray)
                if intersection.is_empty:
                    distances.append(0.0)
                else:
                    if hasattr(intersection, 'geoms'):
                        # Multiple intersections, take the farthest from centroid
                        max_dist = max(centroid.distance(geom) for geom in intersection.geoms 
                                     if hasattr(geom, 'coords'))
                    else:
                        max_dist = centroid.distance(intersection)
                    distances.append(max_dist)
            except Exception:
                distances.append(0.0)
        
        # Return [centroid_x, centroid_y, distance_0, distance_1, ...]
        return np.array([centroid.x, centroid.y] + distances)
    
    except Exception as e:
        # Fallback: return zeros with correct shape
        print(f"Warning: Failed to parameterize polygon - {e}")
        return np.zeros(n_radial + 2)


def _polygon_to_coordinate_features(polygon, n_coords=20):
    """
    Convert polygon to fixed number of coordinate pairs.
    
    This method samples n_coords coordinate pairs from the polygon boundary,
    either by downsampling (if polygon has more points) or upsampling 
    (if polygon has fewer points) using linear interpolation.
    """
    try:
        # Get exterior coordinates (remove duplicate last point)
        coords = list(polygon.exterior.coords)[:-1]
        
        if len(coords) == 0:
            # Empty polygon
            return np.zeros(n_coords * 2)
        
        # Resample to fixed number of points
        if len(coords) > n_coords:
            # Downsample by selecting evenly spaced points
            indices = np.linspace(0, len(coords)-1, n_coords, dtype=int)
            coords = [coords[i] for i in indices]
        elif len(coords) < n_coords:
            # Upsample by linear interpolation
            coords = np.array(coords)
            indices = np.linspace(0, len(coords)-1, n_coords)
            coords_interp = []
            for idx in indices:
                i = int(idx)
                if i >= len(coords) - 1:
                    coords_interp.append(coords[-1])
                else:
                    # Linear interpolation between coords[i] and coords[i+1]
                    t = idx - i
                    x = coords[i][0] * (1-t) + coords[i+1][0] * t
                    y = coords[i][1] * (1-t) + coords[i+1][1] * t
                    coords_interp.append((x, y))
            coords = coords_interp
        
        # Flatten coordinate pairs into single array
        return np.array(coords).flatten()
    
    except Exception as e:
        # Fallback: return zeros with correct shape
        print(f"Warning: Failed to parameterize polygon coordinates - {e}")
        return np.zeros(n_coords * 2)


def features_to_polygon(features, method='radial', n_params=36):
    """
    Convert feature vector back to shapely Polygon.
    
    This is the inverse operation of translate_multipolygon_for_ml, converting
    a numerical feature vector back into a polygon geometry.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature vector from translate_multipolygon_for_ml
    method : str
        The parameterization method used ('radial' or 'coordinates')
    n_params : int
        Number of parameters used in original parameterization
    
    Returns:
    --------
    shapely.geometry.polygon.Polygon
        Reconstructed polygon from features
    """
    if method == 'radial':
        return _radial_features_to_polygon(features)
    elif method == 'coordinates':
        return _coordinate_features_to_polygon(features)
    else:
        raise ValueError(f"Unknown method: {method}")


def _radial_features_to_polygon(features):
    """Convert radial representation back to polygon"""
    try:
        centroid_x, centroid_y = features[0], features[1]
        distances = features[2:]
        angles = np.linspace(0, 2*np.pi, len(distances), endpoint=False)
        
        # Calculate boundary points
        boundary_points = []
        for angle, distance in zip(angles, distances):
            if distance > 0:
                x = centroid_x + distance * np.cos(angle)
                y = centroid_y + distance * np.sin(angle)
                boundary_points.append((x, y))
        
        if len(boundary_points) < 3:
            # Fallback to small circle around centroid
            circle_radius = 0.001
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            boundary_points = [
                (centroid_x + circle_radius * np.cos(a), 
                 centroid_y + circle_radius * np.sin(a))
                for a in angles
            ]
        
        return Polygon(boundary_points)
    
    except Exception:
        # Fallback polygon
        centroid_x, centroid_y = features[0], features[1]
        return Polygon([(centroid_x-0.001, centroid_y-0.001),
                       (centroid_x+0.001, centroid_y-0.001),
                       (centroid_x+0.001, centroid_y+0.001),
                       (centroid_x-0.001, centroid_y+0.001)])


def _coordinate_features_to_polygon(features):
    """Convert coordinate array back to polygon"""
    try:
        coords = features.reshape(-1, 2)
        return Polygon(coords)
    except Exception:
        # Fallback if invalid polygon
        centroid = coords.mean(axis=0) if len(coords) > 0 else np.array([0, 0])
        return Polygon([(centroid[0]-0.001, centroid[1]-0.001),
                       (centroid[0]+0.001, centroid[1]-0.001),
                       (centroid[0]+0.001, centroid[1]+0.001),
                       (centroid[0]-0.001, centroid[1]+0.001)])



# Add compatibility function for existing fire datasets
def extract_polygon_features_from_fire_dataset(fire_dataset, method='radial', n_params=36):
    """
    Extract polygon features from a FireDataset for use in machine learning models.
    
    This function processes all daily fire perimeters in a FireDataset and converts
    them to numerical features suitable for ML training.
    
    Parameters:
    -----------
    fire_dataset : FireDataset
        A FireDataset object containing daily fire data
    method : str
        Parameterization method ('radial' or 'coordinates')
    n_params : int
        Number of parameters for parameterization
    
    Returns:
    --------
    list of np.ndarray
        List of feature vectors, one for each day with perimeter data
    list of date
        Corresponding dates for each feature vector
    """
    features_list = []
    dates_list = []
    
    for day_data in fire_dataset.daily_data:
        if day_data.daily_perimeter is not None:
            try:
                features = translate_multipolygon_for_ml(
                    day_data.daily_perimeter, 
                    method=method, 
                    n_params=n_params
                )
                features_list.append(features)
                dates_list.append(day_data.date)
            except Exception as e:
                print(f"Warning: Failed to extract features for {day_data.date}: {e}")
    
    return features_list, dates_list