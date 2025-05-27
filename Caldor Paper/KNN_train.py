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


def main():
    """Main function to demonstrate the fire data compilation"""
    print("=== Fire Data Compilation Script ===")
    
    # Initialize compiler
    compiler = FireDataCompiler()
    
    # Load data
    compiler.load_data()
    
    # Test with a small subset first (first 5 fires)
    print("\n=== Testing with first 5 fires ===")
    perimeter_fire_ids = set(compiler.perimeter_data['fireID'].unique())
    ground_fire_ids = set(compiler.ground_data['fireID'].unique())
    common_fire_ids = sorted(perimeter_fire_ids & ground_fire_ids)[:5]
    
    test_fires = compiler.compile_all_fires(fire_ids=common_fire_ids)
    
    # Display sample results
    print("\n=== Sample Results ===")
    for fire in test_fires[:2]:  # Show first 2 fires
        summary = fire.get_summary()
        print(f"\nFire {summary['fire_id']}:")
        print(f"  Duration: {summary['duration_days']} days ({summary['start_date']} to {summary['end_date']})")
        print(f"  Max area: {summary['max_area']:.6f}")
        print(f"  Days with ground data: {summary['days_with_ground_data']}")
        print(f"  Total ground data points: {summary['total_ground_data_points']}")
        
        # Show sample day data
        if fire.daily_data:
            sample_day = fire.daily_data[0]
            print(f"  Sample day ({sample_day.date}):")
            print(f"    Cumulative perimeters: {len(sample_day.cumulative_perimeters)}")
            print(f"    Has ground data: {sample_day.has_ground_data}")
            if sample_day.ground_data:
                print(f"    Ground data points: {sample_day.ground_data.num_points}")
                stats = sample_day.ground_data.get_summary_stats()
                if 'vegetation_transpiration' in stats and 'count' in stats['vegetation_transpiration'] and stats['vegetation_transpiration']['count'] > 0:
                    print(f"    Avg vegetation transpiration: {stats['vegetation_transpiration']['mean']:.2f}")
    
    # Save test results
    compiler.save_compiled_data(test_fires, "test_compiled_fire_data.pkl")
    
    print(f"\n=== Test Complete ===")
    print(f"Successfully compiled {len(test_fires)} fires.")
    print("To compile all fires, call: compiler.compile_all_fires()")


if __name__ == "__main__":
    main()