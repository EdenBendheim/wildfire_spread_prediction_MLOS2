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
from shapely.ops import unary_union
from shapely.affinity import translate, scale
from shapely.geometry import LineString, Point
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
    
    def __post_init__(self):
        """Validate that all data lists have the same length"""
        lengths = [
            len(self.coordinates), len(self.avg_surf_temp), len(self.rainfall),
            len(self.vegetation_transpiration), len(self.wind_speed), len(self.air_temp),
            len(self.air_humidity), len(self.soil_moisture), len(self.soil_temp)
        ]
        if len(set(lengths)) > 1:
            raise ValueError(f"All data lists must have same length. Got: {lengths}")
    
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
            'soil_temp': self.soil_temp
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
        fire_perimeters = self.perimeter_data[self.perimeter_data['fireID'] == fire_id].copy()
        fire_ground = self.ground_data[self.ground_data['fireID'] == fire_id].copy()
        
        if fire_perimeters.empty:
            print(f"Warning: No perimeter data found for fire {fire_id}")
            return None
            
        # Sort by date
        fire_perimeters = fire_perimeters.sort_values('date')
        fire_ground = fire_ground.sort_values('date')
        
        # Get all unique dates for this fire
        all_dates = sorted(set(fire_perimeters['date'].unique()) | set(fire_ground['date'].unique()))
        
        daily_data = []
        cumulative_perimeters = []
        
        for current_date in all_dates:
            # Get perimeter for this specific date
            daily_perimeter_data = fire_perimeters[fire_perimeters['date'] == current_date]
            daily_perimeter = None
            
            if not daily_perimeter_data.empty:
                # Add this day's perimeter to cumulative list
                geom = daily_perimeter_data.iloc[0]['geometry']
                if isinstance(geom, (Polygon, MultiPolygon)):
                    cumulative_perimeters.append(geom)
                    daily_perimeter = geom
            
            # Get ground data for this date
            daily_ground_data = fire_ground[fire_ground['date'] == current_date]
            ground_data_obj = None
            
            if not daily_ground_data.empty:
                # Extract ground data
                coordinates = list(zip(daily_ground_data['lat'], daily_ground_data['lon']))
                
                ground_data_obj = DayGroundData(
                    date=current_date,
                    fire_id=fire_id,
                    coordinates=coordinates,
                    avg_surf_temp=daily_ground_data['AvgSurfT_tavg'].tolist(),
                    rainfall=daily_ground_data['Rainf_tavg'].tolist(),
                    vegetation_transpiration=daily_ground_data['TVeg_tavg'].tolist(),
                    wind_speed=daily_ground_data['Wind_f_tavg'].tolist(),
                    air_temp=daily_ground_data['Tair_f_tavg'].tolist(),
                    air_humidity=daily_ground_data['Qair_f_tavg'].tolist(),
                    soil_moisture=daily_ground_data['SoilMoi00_10cm_tavg'].tolist(),
                    soil_temp=daily_ground_data['SoilTemp00_10cm_tavg'].tolist()
                )
            
            # Create day fire data with cumulative perimeters
            day_fire_data = DayFireData(
                date=current_date,
                fire_id=fire_id,
                cumulative_perimeters=cumulative_perimeters.copy(),
                daily_perimeter=daily_perimeter,
                ground_data=ground_data_obj
            )
            
            daily_data.append(day_fire_data)
        
        return FireDataset(fire_id=fire_id, daily_data=daily_data)
    
    def compile_all_fires(self, fire_ids: Optional[List[int]] = None) -> List[FireDataset]:
        """Compile data for all fires or a specific list of fire IDs"""
        if self.perimeter_data is None or self.ground_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if fire_ids is None:
            # Get intersection of fire IDs that exist in both datasets
            perimeter_fire_ids = set(self.perimeter_data['fireID'].unique())
            ground_fire_ids = set(self.ground_data['fireID'].unique())
            fire_ids = sorted(perimeter_fire_ids & ground_fire_ids)
            print(f"Found {len(fire_ids)} fires with both perimeter and ground data")
        
        compiled_fires = []
        failed_fires = []
        
        for i, fire_id in enumerate(fire_ids):
            try:
                print(f"Processing fire {fire_id} ({i+1}/{len(fire_ids)})...")
                fire_dataset = self.compile_fire_data(fire_id)
                if fire_dataset:
                    compiled_fires.append(fire_dataset)
                    print(f"  ✓ Fire {fire_id}: {fire_dataset.duration_days} days, max area: {fire_dataset.max_area:.6f}")
                else:
                    failed_fires.append(fire_id)
                    print(f"  ✗ Fire {fire_id}: Failed to compile")
            except Exception as e:
                failed_fires.append(fire_id)
                print(f"  ✗ Fire {fire_id}: Error - {e}")
        
        print(f"\nCompilation complete:")
        print(f"  Successfully compiled: {len(compiled_fires)} fires")
        print(f"  Failed: {len(failed_fires)} fires")
        if failed_fires:
            print(f"  Failed fire IDs: {failed_fires[:10]}{'...' if len(failed_fires) > 10 else ''}")
        
        return compiled_fires
    
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


def main():
    # Load the data and get the fire perimeter for day 10 of fire 2623
    compiler = FireDataCompiler()
    compiler.load_data()

    # Compile the specific fire
    fire_dataset = compiler.compile_fire_data(2623)

    if fire_dataset and len(fire_dataset.daily_data) >= 10:
        # Get day 10 (index 9 since it's 0-based)
        day_10_data = fire_dataset.daily_data[9]
        
        # Get the perimeter polygon for day 10
        day_10_perimeter = day_10_data.daily_perimeter
        
        if day_10_perimeter:
            print(f"Fire 2623 Day 10 perimeter loaded successfully")
            print(f"Date: {day_10_data.date}")
            print(f"Perimeter type: {type(day_10_perimeter)}")
            print(f"Perimeter area: {day_10_perimeter.area}")
            print(f"Perimeter bounds: {day_10_perimeter.bounds}")
            
            # Demonstrate the new MultiPolygon translation functionality
            print("\n" + "="*60)
            print("DEMONSTRATING MULTIPOLYGON TRANSLATION FOR ML")
            print("="*60)
            
            # Translate the perimeter to ML features using radial method
            print("\n1. Converting polygon to ML features (radial method):")
            features_radial = translate_multipolygon_for_ml(day_10_perimeter, method='radial', n_params=36)
            print(f"   Original polygon area: {day_10_perimeter.area:.6f}")
            print(f"   Features shape: {features_radial.shape}")
            print(f"   Centroid: ({features_radial[0]:.6f}, {features_radial[1]:.6f})")
            print(f"   Sample radial distances: {features_radial[2:8]}")
            
            # Convert back to polygon
            reconstructed_poly = features_to_polygon(features_radial, method='radial', n_params=36)
            print(f"   Reconstructed area: {reconstructed_poly.area:.6f}")
            print(f"   Area preservation ratio: {reconstructed_poly.area/day_10_perimeter.area:.4f}")
            
            # Try coordinate method too
            print("\n2. Converting polygon to ML features (coordinate method):")
            features_coords = translate_multipolygon_for_ml(day_10_perimeter, method='coordinates', n_params=20)
            print(f"   Features shape: {features_coords.shape}")
            print(f"   Sample coordinates: {features_coords[:8]}")
            
            # Extract features from entire fire dataset
            print("\n3. Extracting features from entire fire dataset:")
            all_features, all_dates = extract_polygon_features_from_fire_dataset(
                fire_dataset, method='radial', n_params=36
            )
            print(f"   Total days with polygon data: {len(all_features)}")
            if all_features:
                print(f"   Feature vector length: {len(all_features[0])}")
                print(f"   Date range: {min(all_dates)} to {max(all_dates)}")
                
                # Show progression of fire size
                areas_over_time = []
                for i, (features, date) in enumerate(zip(all_features, all_dates)):
                    reconstructed = features_to_polygon(features, method='radial')
                    areas_over_time.append(reconstructed.area)
                    if i < 5:  # Show first 5 days
                        print(f"   Day {i+1} ({date}): area = {areas_over_time[i]:.6f}")
                
                print(f"   Maximum fire area: {max(areas_over_time):.6f}")
                print(f"   Fire growth ratio: {max(areas_over_time)/areas_over_time[0]:.2f}x")
            
            print("\n" + "="*60)
            print("MultiPolygon translation demonstration complete!")
            print("Use translate_multipolygon_for_ml() to convert polygons to ML features")
            print("Use features_to_polygon() to convert ML predictions back to polygons")
            print("="*60)
        else:
            print(f"No perimeter data found for day 10 of fire 2623")
            # Check if cumulative perimeters are available
            if day_10_data.cumulative_perimeters:
                print(f"But cumulative perimeters available: {len(day_10_data.cumulative_perimeters)} polygons")
    else:
        print(f"Fire 2623 not found or has fewer than 10 days of data")
        if fire_dataset:
            print(f"Fire 2623 has {len(fire_dataset.daily_data)} days of data")
        
        # Still demonstrate the translation functionality with a simple example
        print("\nDemonstrating MultiPolygon translation with simple example:")
        demonstrate_multipolygon_translation()




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


def translate_multipolygon_for_ml(multipolygon, method='radial', n_params=36):
    """
    Translate a shapely MultiPolygon object into a numerical feature vector for machine learning.
    
    This function converts complex polygon shapes into a standardized numerical representation
    that can be used as input to machine learning models. Two parameterization methods are 
    supported:
    
    Parameters:
    -----------
    multipolygon : shapely.geometry.multipolygon.MultiPolygon or shapely.geometry.polygon.Polygon
        The input polygon geometry to convert
    method : str, default='radial'
        The parameterization method to use:
        - 'radial': Centroid + radial distances (recommended for fire perimeters)
        - 'coordinates': Fixed number of coordinate pairs
    n_params : int, default=36
        Number of parameters for the parameterization:
        - For 'radial': Number of radial rays (total features = n_params + 2 for centroid)
        - For 'coordinates': Number of coordinate pairs (total features = n_params * 2)
    
    Returns:
    --------
    np.ndarray
        Feature vector representing the polygon:
        - For 'radial' method: [centroid_x, centroid_y, dist_0, dist_1, ..., dist_n]
        - For 'coordinates' method: [x_0, y_0, x_1, y_1, ..., x_n, y_n]
    
    Examples:
    ---------
    >>> from shapely.geometry import Polygon, MultiPolygon
    >>> # Single polygon
    >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> features = translate_multipolygon_for_ml(poly)
    >>> print(features.shape)  # (38,) for default radial with 36 params
    
    >>> # MultiPolygon (will use largest component)
    >>> multi = MultiPolygon([poly, Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])])
    >>> features = translate_multipolygon_for_ml(multi)
    
    >>> # Different parameterization
    >>> features_coords = translate_multipolygon_for_ml(poly, method='coordinates', n_params=20)
    >>> print(features_coords.shape)  # (40,) for 20 coordinate pairs
    """
    
    # Handle MultiPolygon by taking the largest component
    if isinstance(multipolygon, MultiPolygon):
        # Take the polygon with the largest area
        polygon = max(multipolygon.geoms, key=lambda p: p.area)
    elif isinstance(multipolygon, Polygon):
        polygon = multipolygon
    else:
        raise ValueError(f"Input must be a Polygon or MultiPolygon, got {type(multipolygon)}")
    
    if method == 'radial':
        return _polygon_to_radial_features(polygon, n_params)
    elif method == 'coordinates':
        return _polygon_to_coordinate_features(polygon, n_params)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'radial' or 'coordinates'")


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


def demonstrate_multipolygon_translation():
    """
    Demonstrate the MultiPolygon translation functionality with examples.
    
    This function shows how to use the translation functions with various
    polygon types and validates the round-trip conversion (polygon -> features -> polygon).
    """
    print("Demonstrating MultiPolygon Translation for ML")
    print("=" * 50)
    
    # Create sample polygons
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    triangle = Polygon([(2, 2), (3, 2), (2.5, 3)])
    multi_poly = MultiPolygon([square, triangle])
    
    # Test radial parameterization
    print("\n1. Radial Parameterization:")
    features_radial = translate_multipolygon_for_ml(multi_poly, method='radial', n_params=12)
    print(f"   Original: MultiPolygon with {len(multi_poly.geoms)} components")
    print(f"   Features shape: {features_radial.shape}")
    print(f"   Centroid: ({features_radial[0]:.3f}, {features_radial[1]:.3f})")
    print(f"   Sample distances: {features_radial[2:6]}")
    
    # Reconstruct polygon
    reconstructed = features_to_polygon(features_radial, method='radial')
    print(f"   Reconstructed area: {reconstructed.area:.6f}")
    print(f"   Original largest area: {max(p.area for p in multi_poly.geoms):.6f}")
    
    # Test coordinate parameterization
    print("\n2. Coordinate Parameterization:")
    features_coords = translate_multipolygon_for_ml(square, method='coordinates', n_params=8)
    print(f"   Features shape: {features_coords.shape}")
    print(f"   Sample coordinates: {features_coords[:6]}")
    
    # Reconstruct polygon
    reconstructed_coords = features_to_polygon(features_coords, method='coordinates')
    print(f"   Reconstructed area: {reconstructed_coords.area:.6f}")
    print(f"   Original area: {square.area:.6f}")
    
    # Test with fire-like irregular shape
    print("\n3. Fire-like Irregular Shape:")
    import math
    
    # Create an irregular "fire-like" shape
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    # Vary radius to create irregular boundary
    radii = [1 + 0.3*math.sin(3*a) + 0.2*np.random.random() for a in angles]
    fire_coords = [(r*np.cos(a), r*np.sin(a)) for a, r in zip(angles, radii)]
    fire_poly = Polygon(fire_coords)
    
    fire_features = translate_multipolygon_for_ml(fire_poly, method='radial', n_params=36)
    print(f"   Fire polygon area: {fire_poly.area:.6f}")
    print(f"   Features shape: {fire_features.shape}")
    
    fire_reconstructed = features_to_polygon(fire_features, method='radial')
    print(f"   Reconstructed area: {fire_reconstructed.area:.6f}")
    print(f"   Area preservation: {fire_reconstructed.area/fire_poly.area:.3f}")
    
    print("\nTranslation demonstration complete!")
    
    return {
        'original_multi': multi_poly,
        'radial_features': features_radial,
        'coordinate_features': features_coords,
        'fire_polygon': fire_poly,
        'fire_features': fire_features
    }


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