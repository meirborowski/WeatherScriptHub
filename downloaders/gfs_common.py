from datetime import datetime, timedelta, timezone

def get_gfs_url(date, cycle, forecast_hour, resolution='0p25'):
    """
    Constructs the NOMADS URL for a specific GFS file.
    
    Args:
        date (datetime): The date of the model run.
        cycle (int): The cycle hour (0, 6, 12, 18).
        forecast_hour (int): The forecast hour (e.g., 0, 3, 6...).
        resolution (str): The grid resolution ('0p25', '0p50', '1p00').
        
    Returns:
        str: The URL to download the file.
    """
    date_str = date.strftime('%Y%m%d')
    cycle_str = f"{cycle:02d}"
    f_hour_str = f"{forecast_hour:03d}"
    
    # URL pattern for GFS on NOMADS
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
    filename = f"gfs.t{cycle_str}z.pgrb2.{resolution}.f{f_hour_str}"
    url = f"{base_url}/gfs.{date_str}/{cycle_str}/atmos/{filename}"
    
    return url, filename

def get_latest_cycle():
    """
    Determines the most likely available GFS cycle based on current UTC time.
    GFS runs are available roughly 4-5 hours after the cycle time.
    
    Returns:
        tuple: (date, cycle)
    """
    today = datetime.now(timezone.utc)
    current_hour = today.hour
    
    if current_hour < 4:
        # Use previous day's 18z
        date = today - timedelta(days=1)
        cycle = 18
    elif current_hour < 10:
        date = today
        cycle = 0
    elif current_hour < 16:
        date = today
        cycle = 6
    elif current_hour < 22:
        date = today
        cycle = 12
    else:
        date = today
        cycle = 18
        
    return date, cycle
