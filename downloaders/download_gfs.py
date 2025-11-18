import os
import argparse
import requests
from datetime import datetime
from gfs_common import get_gfs_url, get_latest_cycle
from downloader import download_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download GFS GRIB2 files.")
    
    parser.add_argument("--date", type=str, help="Date in YYYYMMDD format. Defaults to latest available.")
    parser.add_argument("--cycle", type=int, choices=[0, 6, 12, 18], help="Cycle hour. Defaults to latest available.")
    parser.add_argument("--max-hour", type=int, default=6, help="Maximum forecast hour to download. Default is 6.")
    parser.add_argument("--step", type=int, default=3, help="Forecast hour step. Default is 3.")
    parser.add_argument("--full", action="store_true", help="Download the full set (0-384 hours). Overrides --max-hour.")
    parser.add_argument("--output", type=str, default="gfs_data", help="Output directory. Default is 'gfs_data'.")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Determine date and cycle
    if args.date and args.cycle is not None:
        try:
            date = datetime.strptime(args.date, "%Y%m%d")
            cycle = args.cycle
        except ValueError:
            print("Error: Date must be in YYYYMMDD format.")
            return
    else:
        # Default to latest if not fully specified
        # (You could enhance this to use provided date + latest cycle, or provided cycle + today, etc.)
        latest_date, latest_cycle = get_latest_cycle()
        date = datetime.strptime(args.date, "%Y%m%d") if args.date else latest_date
        cycle = args.cycle if args.cycle is not None else latest_cycle

    # Determine forecast hours
    if args.full:
        max_h = 384
    else:
        max_h = args.max_hour
        
    forecast_hours = list(range(0, max_h + 1, args.step))
    
    output_dir = args.output
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Targeting GFS Run: {date.strftime('%Y-%m-%d')} Cycle: {cycle:02d}z")
    print(f"Downloading {len(forecast_hours)} files (Hours: 0 to {max_h}, Step: {args.step})...")
    
    for f_hour in forecast_hours:
        url, filename = get_gfs_url(date, cycle, f_hour)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"File already exists: {filename}")
            continue
            
        print(f"Downloading {filename} (Forecast Hour {f_hour})...")
        try:
            download_file(url, output_path)
            print(f"Successfully downloaded {filename}")
        except requests.exceptions.HTTPError as e:
            print(f"Failed to download {filename}: {e}")
            print("The file might not be available yet on the server.")

if __name__ == "__main__":
    main()
