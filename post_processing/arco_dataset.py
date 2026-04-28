import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import gcsfs
import pandas as pd

from typing import Tuple

class ARCOStreamDataset(Dataset):
    """
    Streams ERA5 data directly from Google Cloud.  
    https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
    """
    def __init__(self, start_date: str, end_date: str, normalize=True):
        """
        Date format: 'YYYY-MM-DD'
        """
        self.normalize = normalize
        print("Connecting to Google Cloud ARCO-ERA5... (this takes a few seconds)")
        
        # Connect to the public Google Cloud Storage bucket
        fs = gcsfs.GCSFileSystem(token='anon')
        mapper = fs.get_mapper('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
        
        # Open the massive cloud dataset lazily (downloads metadata only)
        self.ds = xr.open_zarr(mapper, consolidated=True)
        
        # Slice the dataset to the exact time period you want to test.
        # ERA5 is hourly. 1 month = ~720 timesteps.
        self.ds = self.ds.sel(time=slice(start_date, end_date))
        self.times = self.ds.time.values
        
        # Target shape from the ClimateNet dataset
        self.target_shape = (768, 1152)
        # Mapping ClimateNet names to ERA5 names
        self.variable_map = {
            # CGNet Features
            'TMQ': 'total_column_water_vapour',   # Total column water vapour
            'U850': 'u_component_of_wind',        # Requires level=850
            'V850': 'v_component_of_wind',        # Requires level=850
            'PSL': 'mean_sea_level_pressure',     # Mean sea level pressure
        }
        # Other features (unused)
        other_variables = {
            'PRECT': 'total_precipitation',     # Total precipitation
            'QREFHT': '2m_dewpoint_temperature',# 2m dewpoint (closest proxy to reference height humidity)

            # --- Pressure ---
            'PS': 'surface_pressure',           # Surface pressure

            # --- Temperature ---
            'TS': 'skin_temperature',           # Skin temperature (surface)
            'TREFHT': '2m_temperature',         # 2 meter temperature (reference height)
            'T200': 'temperature',              # Requires level=200
            'T500': 'temperature',              # Requires level=500

            # --- Wind Vectors (U/V) ---
            'UBOT': '10m_u_component_of_wind',  # Lowest model level / 10m U wind
            'VBOT': '10m_v_component_of_wind',  # Lowest model level / 10m V wind

            # --- Geopotential Height ---
            'ZBOT': 'surface_geopotential',     # Surface geopotential (or topography)
            'Z1000': 'geopotential',            # Requires level=1000
            'Z200': 'geopotential',             # Requires level=200

            # --- Vorticity ---
            'VRT850': 'vorticity',              # Relative vorticity (Requires level=850)
            'VRTBOT': 'vorticity',              # Relative vorticity (Requires level=1000 for "bottom")

            # --- Derived Variables ---
            'WS850': 'DERIVED_WS850',           # Needs to be calculated from U850 & V850
            'WSBOT': 'DERIVED_WSBOT',           # Needs to be calculated from UBOT & VBOT
        }


    def __len__(self):
        return len(self.times)

    
    def get_all_times(self):
        """Returns a list of all timestamps in the dataset in 'YYYY-MM-DD HH:00 UTC' format."""

        times = []
        for idx in range(len(self)):
            times.append(pd.to_datetime(self.times[idx]).strftime('%Y-%m-%d %H:00 UTC'))
        return times
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Returns (features, labels, timestamp) for the given index. Labels are empty tensors since this is for inference only."""
        # 1. Isolate the specific timestep
        # This is where the actual download from the cloud happens
        current_data = self.ds.isel(time=idx)
        # Centers the map on the Americas
        current_data = current_data.roll(longitude=-504, roll_coords=True)
        
        processed_features = []
        
        # 2. Extract and format each variable
        for climatenet_var, era5_var in self.variable_map.items():
            if era5_var in current_data:
                # If it's a 3D variable (like Temperature at different heights), 
                # you have to select the specific pressure level (e.g., 200 hPa)
                if 'level' in current_data[era5_var].dims:
                    # Example: Assuming climatenet_var is 'T200'
                    level_val = int(climatenet_var[1:]) if climatenet_var[1:].isdigit() else 850
                    var_data = current_data[era5_var].sel(level=level_val).values
                else:
                    # 2D surface variables like Total Column Water Vapour
                    var_data = current_data[era5_var].values
                
                # 3. Handle grid size differences
                # ERA5 0.25 degree resolution is (721, 1440). 
                # Your UNet expects (768, 1152). We crop/pad exactly like your previous dataset.
                if var_data.shape != self.target_shape:
                    cropped_data = np.zeros(self.target_shape)
                    h = min(var_data.shape[0], self.target_shape[0])
                    w = min(var_data.shape[1], self.target_shape[1])
                    cropped_data[:h, :w] = var_data[:h, :w]
                    var_data = cropped_data
                    
                processed_features.append(var_data)
            else: 
                ValueError(f'"{era5_var}" not in ARCO dataset')
                
        # Fill missing features with zeros if you haven't mapped all 20 yet
        # while len(processed_features) < 20:
        #     processed_features.append(np.zeros(self.target_shape))

        # 4. Stack and Standardize
        data = np.stack(processed_features, axis=0)
        if self.normalize:
            data = (data - data.mean(axis=(1, 2), keepdims=True)) / (data.std(axis=(1, 2), keepdims=True) + 1e-7)

        current_time = self.times[idx]
        timestamp_str = pd.to_datetime(current_time).strftime('%Y-%m-%d %H:00 UTC')

        return torch.from_numpy(data).float(), torch.empty(1), timestamp_str