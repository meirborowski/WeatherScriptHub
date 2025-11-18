import os
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np

# Configuration for Regions
REGIONS = {
    'global': None,
    'europe': [-70, 45, 30, 85], # Extended to include Greenland
    'us': [-125, -65, 25, 50],
    'asia': [60, 150, 0, 60],
    'australia': [110, 160, -45, -10],
    'south_america': [-85, -30, -60, 15],
    'africa': [-20, 55, -35, 40],
    'north_atlantic': [-80, 10, 20, 70],
    'north_hemisphere': [-180, 180, 0, 90]
}

def get_dataset(grib_file, var_config):
    """
    Attempts to open the GRIB file with appropriate filters for the variable.
    """
    filter_keys = var_config.get('filter_keys', {})
    try:
        # backend_kwargs={'errors': 'ignore'} helps skip missing keys
        ds = xr.open_dataset(grib_file, engine='cfgrib', 
                             backend_kwargs={'filter_by_keys': filter_keys})
        return ds
    except Exception as e:
        # print(f"Debug: Failed to open with keys {filter_keys}: {e}")
        return None

# --- Processing Functions ---

def process_2m_temp(ds):
    if 't2m' in ds:
        data = ds['t2m']
    elif 't' in ds:
        data = ds['t']
    else:
        return None, "Temperature variable not found"
    return data - 273.15, "2m Temperature (°C)"

def process_2m_dewpoint(ds):
    if 'd2m' in ds:
        data = ds['d2m']
    elif '2d' in ds:
        data = ds['2d']
    else:
        return None, "Dewpoint variable not found"
    return data - 273.15, "2m Dewpoint (°C)"

def process_850_temp(ds):
    if 't' in ds:
        data = ds['t']
    else:
        return None, "850hPa Temperature not found"
    return data - 273.15, "850hPa Temperature (°C)"

def process_500_temp(ds):
    if 't' in ds:
        data = ds['t']
    else:
        return None, "500hPa Temperature not found"
    return data - 273.15, "500hPa Temperature (°C)"

def process_500_gph(ds):
    if 'gh' in ds:
        data = ds['gh']
    else:
        return None, "Geopotential Height not found"
    # Convert meters to decameters
    return data / 10.0, "500hPa Geopotential Height (dam)"

def process_mslp(ds):
    if 'prmsl' in ds:
        data = ds['prmsl']
    elif 'msl' in ds:
        data = ds['msl']
    else:
        return None, "MSLP variable not found"
    return data / 100.0, "Mean Sea Level Pressure (hPa)"

def process_wind_10m(ds):
    u, v = None, None
    if 'u10' in ds and 'v10' in ds:
        u, v = ds['u10'], ds['v10']
    elif 'u' in ds and 'v' in ds:
        u, v = ds['u'], ds['v']
    
    if u is not None and v is not None:
        speed = np.sqrt(u**2 + v**2)
        return speed, "10m Wind Speed (m/s)"
    return None, "Wind variables not found"

def process_jet_200(ds):
    if 'u' in ds and 'v' in ds:
        speed = np.sqrt(ds['u']**2 + ds['v']**2)
        return speed, "200hPa Wind Speed (m/s)"
    return None, "200hPa Wind not found"

def process_precip(ds):
    if 'tp' in ds:
        data = ds['tp']
    elif 'apcp' in ds:
        data = ds['apcp']
    else:
        return None, "Precipitation variable not found"
    return data, "Total Precipitation (mm)"

def process_cape(ds):
    if 'cape' in ds:
        data = ds['cape']
    else:
        return None, "CAPE not found"
    return data, "CAPE (J/kg)"

def process_tcc(ds):
    if 'tcc' in ds:
        data = ds['tcc']
    else:
        return None, "Cloud Cover not found"
    return data, "Total Cloud Cover (%)"

# --- Variable Configuration ---

VARIABLES = {
    # Surface / 2m
    '2m_temp': {
        'filter_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'},
        'process_func': process_2m_temp,
        'cmap': 'coolwarm',
        'title': '2m Temperature'
    },
    '2m_dewpoint': {
        'filter_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'},
        'process_func': process_2m_dewpoint,
        'cmap': 'BrBG',
        'title': '2m Dewpoint Temperature'
    },
    'mslp': {
        'filter_keys': {'typeOfLevel': 'meanSea', 'stepType': 'instant'},
        'process_func': process_mslp,
        'cmap': 'viridis',
        'title': 'Mean Sea Level Pressure'
    },
    '10m_wind': {
        'filter_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10, 'stepType': 'instant'},
        'process_func': process_wind_10m,
        'cmap': 'YlOrRd',
        'title': '10m Wind Speed'
    },
    'precip': {
        'filter_keys': {'stepType': 'accum'},
        'process_func': process_precip,
        'cmap': 'Blues',
        'title': 'Total Precipitation'
    },
    'cape': {
        'filter_keys': {'typeOfLevel': 'surface', 'stepType': 'instant'},
        'process_func': process_cape,
        'cmap': 'RdPu',
        'title': 'Convective Available Potential Energy (CAPE)'
    },
    'cloud_cover': {
        'filter_keys': {'typeOfLevel': 'atmosphere', 'stepType': 'instant'},
        'process_func': process_tcc,
        'cmap': 'Greys',
        'title': 'Total Cloud Cover'
    },
    
    # Upper Air
    '850_temp': {
        'filter_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 850, 'stepType': 'instant'},
        'process_func': process_850_temp,
        'cmap': 'coolwarm',
        'title': '850hPa Temperature'
    },
    '500_gph': {
        'filter_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 500, 'stepType': 'instant'},
        'process_func': process_500_gph,
        'cmap': 'jet',
        'title': '500hPa Geopotential Height'
    },
    '200_jet': {
        'filter_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 200, 'stepType': 'instant'},
        'process_func': process_jet_200,
        'cmap': 'plasma',
        'title': '200hPa Jet Stream'
    },
    
    # Composite Maps
    'synoptic': {
        'type': 'composite',
        'title': '500hPa Geopotential Height (dam), Temperature (°C) & MSLP (hPa)',
        'layers': [
             {
                'name': '500_gph',
                'filter_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 500, 'stepType': 'instant'},
                'process_func': process_500_gph,
                'plot_type': 'filled_contour',
                'cmap': 'nipy_spectral', 
                'levels': np.arange(460, 600, 4), 
                'cbar_label': '500hPa Geopotential Height (dam)',
                'alpha': 0.9
             },
             {
                'name': 'mslp',
                'filter_keys': {'typeOfLevel': 'meanSea', 'stepType': 'instant'},
                'process_func': process_mslp,
                'plot_type': 'contour',
                'colors': 'white',
                'levels': np.arange(900, 1060, 5),
                'linewidths': 1.5,
                'label_contours': True,
                'fmt': '%d'
             },
             {
                'name': '500_temp',
                'filter_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 500, 'stepType': 'instant'},
                'process_func': process_500_temp,
                'plot_type': 'contour',
                'colors': 'gray',
                'linestyles': 'dashed',
                'levels': np.arange(-50, 10, 5),
                'linewidths': 1.0,
                'label_contours': True,
                'fmt': '%d'
             },
             {
                'name': '552_line',
                'filter_keys': {'typeOfLevel': 'isobaricInhPa', 'level': 500, 'stepType': 'instant'},
                'process_func': process_500_gph,
                'plot_type': 'contour',
                'colors': 'black',
                'levels': [552],
                'linewidths': 2.5,
                'label_contours': True,
                'fmt': '%d'
             }
        ]
    },
    'precip_mslp': {
        'type': 'composite',
        'title': 'Total Precipitation (mm) & MSLP (hPa)',
        'layers': [
             {
                'name': 'precip',
                'filter_keys': {'stepType': 'accum'},
                'process_func': process_precip,
                'plot_type': 'filled_contour',
                # Custom color list for precipitation (White -> Blue -> Purple -> Red)
                'colors': ['#ffffff', '#b4f0f0', '#96d2fa', '#78b4f0', '#3c96f5', '#0000ff', '#0000aa', '#ff00ff', '#aa00ff', '#ff0000'],
                'levels': [0.1, 1, 2, 5, 10, 15, 20, 30, 50, 100],
                'cbar_label': 'Total Precipitation (mm)',
                'alpha': 0.9,
                'extend': 'max'
             },
             {
                'name': 'mslp',
                'filter_keys': {'typeOfLevel': 'meanSea', 'stepType': 'instant'},
                'process_func': process_mslp,
                'plot_type': 'contour',
                'colors': 'black',
                'levels': np.arange(900, 1060, 5),
                'linewidths': 1.0,
                'label_contours': True,
                'fmt': '%d'
             }
        ]
    }
}

def wrap_data(data_array):
    """
    Adds a cyclic point to the data array to prevent white lines at the meridian.
    Returns data, lon, lat.
    """
    data = data_array.values
    lon = data_array.longitude.values
    lat = data_array.latitude.values
    
    data_c, lon_c = add_cyclic_point(data, coord=lon)
    return data_c, lon_c, lat

def get_projection(region_name):
    """
    Determines the best projection for the region.
    """
    if region_name in ['europe', 'us', 'asia', 'north_atlantic', 'north_hemisphere']:
        # Use North Polar Stereographic for NH regions
        # Center the projection on the region's longitude to keep "North Up"
        if region_name == 'north_hemisphere':
            center_lon = -45 # Good view of Atlantic/Europe/US
        elif region_name in REGIONS:
            extent = REGIONS[region_name]
            center_lon = (extent[0] + extent[1]) / 2
        else:
            center_lon = 0
        return ccrs.NorthPolarStereo(central_longitude=center_lon)
    else:
        return ccrs.PlateCarree()

def create_composite_map(grib_file, output_path, var_config, region_name):
    fig = plt.figure(figsize=(15, 10))
    
    # Use dynamic projection based on region
    projection = get_projection(region_name)
    ax = plt.axes(projection=projection)
    
    # Set extent
    if region_name in REGIONS and REGIONS[region_name] is not None:
        # When using a different projection, we must specify crs=ccrs.PlateCarree() for the extent
        ax.set_extent(REGIONS[region_name], crs=ccrs.PlateCarree())
        
        # For full hemisphere views, we might need to adjust the boundary to be circular
        # But set_extent usually handles the rectangular crop in the projected space
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')
    
    valid_time_str = "Unknown"
    
    for layer in var_config['layers']:
        print(f"  Processing layer: {layer['name']}...")
        ds = get_dataset(grib_file, layer)
        if ds is None:
            print(f"  Skipping layer {layer['name']} (load failed)")
            continue
            
        data, label = layer['process_func'](ds)
        if data is None:
            print(f"  Skipping layer {layer['name']} (process failed)")
            continue
            
        if valid_time_str == "Unknown" and 'valid_time' in data.coords:
            valid_time_str = data.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').item()
            
        # Wrap data to avoid white line at meridian
        data_c, lon_c, lat = wrap_data(data)
            
        try:
            if layer['plot_type'] == 'filled_contour':
                # Filled contours (shading)
                kwargs = {
                    'transform': ccrs.PlateCarree(),
                    'levels': layer.get('levels'),
                    'alpha': layer.get('alpha', 1.0),
                    'extend': layer.get('extend', 'both')
                }
                
                if 'colors' in layer:
                    kwargs['colors'] = layer['colors']
                else:
                    kwargs['cmap'] = layer.get('cmap', 'viridis')
                    
                cf = ax.contourf(lon_c, lat, data_c, **kwargs)
                plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=50, label=layer.get('cbar_label', label))
                
            elif layer['plot_type'] == 'contour':
                # Line contours
                cs = ax.contour(lon_c, lat, data_c, 
                                transform=ccrs.PlateCarree(), 
                                colors=layer.get('colors', 'black'), 
                                levels=layer.get('levels'),
                                linewidths=layer.get('linewidths', 1.0),
                                linestyles=layer.get('linestyles', 'solid'))
                
                if layer.get('label_contours', False):
                    ax.clabel(cs, inline=True, fontsize=10, fmt=layer.get('fmt', '%1.0f'))
                    
        except Exception as e:
            print(f"  Error plotting layer {layer['name']}: {e}")

    ax.set_title(f"GFS {var_config['title']}\nRegion: {region_name.capitalize()} | Valid: {valid_time_str}")
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Map saved to {output_path}")

def create_map(grib_file, output_path, var_type, region_name):
    """
    Creates a map based on variable type and region.
    """
    if var_type not in VARIABLES:
        print(f"Unknown variable type: {var_type}")
        return

    var_config = VARIABLES[var_type]
    
    if var_config.get('type') == 'composite':
        return create_composite_map(grib_file, output_path, var_config, region_name)
    
    # 1. Open Dataset
    ds = get_dataset(grib_file, var_config)
    
    # Fallback for some variables that might be in different levels/types in different GFS versions
    if ds is None:
        print(f"Warning: Could not find exact match for {var_type}, trying generic open...")
        try:
            ds = xr.open_dataset(grib_file, engine='cfgrib')
        except:
            print("Could not open GRIB file.")
            return

    # 2. Process Data
    data, label = var_config['process_func'](ds)
    if data is None:
        print(f"Error: {label} - Variable not found in file.")
        return

    # 3. Setup Plot
    fig = plt.figure(figsize=(15, 10))
    
    # Use dynamic projection
    projection = get_projection(region_name)
    ax = plt.axes(projection=projection)
    
    # Set extent if region is specified
    if region_name in REGIONS and REGIONS[region_name] is not None:
        ax.set_extent(REGIONS[region_name], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')
    
    # Plot data
    try:
        # Wrap data to avoid white line at meridian
        data_c, lon_c, lat = wrap_data(data)
        
        plot = ax.contourf(
            lon_c, lat, data_c,
            transform=ccrs.PlateCarree(),
            cmap=var_config['cmap'],
            extend='both'
        )
        
        plt.colorbar(plot, ax=ax, label=label, orientation='vertical', pad=0.02, aspect=50)
        
        valid_time = data.valid_time.dt.strftime('%Y-%m-%d %H:%M UTC').item() if 'valid_time' in data.coords else "Unknown"
        ax.set_title(f"GFS {var_config['title']}\nRegion: {region_name.capitalize()} | Valid: {valid_time}")
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Map saved to {output_path}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate weather maps from GFS GRIB2 files.")
    parser.add_argument("--file", type=str, help="Specific GRIB file to process. If not provided, scans gfs_data.")
    parser.add_argument("--var", type=str, default="2m_temp", choices=VARIABLES.keys(), help="Variable to plot.")
    parser.add_argument("--region", type=str, default="global", choices=REGIONS.keys(), help="Region to plot.")
    parser.add_argument("--output", type=str, default="maps_output", help="Output directory.")
    
    args = parser.parse_args()
    
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine file to process
    if args.file:
        if os.path.exists(args.file):
            files = [args.file]
        else:
            print(f"File not found: {args.file}")
            return
    else:
        data_dir = "gfs_data"
        if not os.path.exists(data_dir):
            print(f"Data directory '{data_dir}' not found.")
            return
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("gfs") and not f.endswith(".idx")]
        if not files:
            print("No GFS files found.")
            return
        print(f"No file specified, using first found: {files[0]}")
        files = [files[0]]

    for grib_file in files:
        filename = os.path.basename(grib_file)
        output_filename = f"{args.var}_{args.region}_{filename}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {filename} -> {args.var} ({args.region})...")
        create_map(grib_file, output_path, args.var, args.region)

if __name__ == "__main__":
    main()
