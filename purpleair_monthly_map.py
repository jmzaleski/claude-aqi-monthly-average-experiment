#!/usr/bin/env python3
"""
PurpleAir Monthly Average Map Generator (Daily Worst Method)

Creates a single static map showing monthly average AQI at each sensor.
Uses EPA-style methodology: Average of daily WORST readings.

Methodology:
1. Fetch all available data for the month (highest granularity)
2. For each sensor, for each day → find WORST (maximum) PM2.5 reading
3. Calculate monthly average from those daily worst readings

This is more health-relevant than simple averaging and reveals pollution events.

Usage:
    export PURPLEAIR_API_KEY="your-key-here"
    
    # Generate January 2026 map for Golden
    python3 purpleair_monthly_map.py \
        --month 2026-01 \
        --background golden_map.tif \
        --output jan2026_aqi_golden.png
    
    # Use cached data (skip API calls)
    python3 purpleair_monthly_map.py \
        --month 2026-01 \
        --background golden_map.tif \
        --output jan2026_aqi_golden.png \
        --use-cached

Author: Generated for air quality monitoring
Date: 2025
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from calendar import monthrange
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import time


# AQI calculation constants (US EPA standard for PM2.5)
AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50, "Good", "#00E400"),
    (12.1, 35.4, 51, 100, "Moderate", "#FFFF00"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (55.5, 150.4, 151, 200, "Unhealthy", "#FF0000"),
    (150.5, 250.4, 201, 300, "Very Unhealthy", "#8F3F97"),
    (250.5, 500.4, 301, 500, "Hazardous", "#7E0023"),
]


def calculate_aqi(pm25):
    """Calculate AQI from PM2.5 concentration using EPA formula."""
    if pd.isna(pm25) or pm25 < 0:
        return None, None, "#CCCCCC"
    
    for pm_low, pm_high, aqi_low, aqi_high, category, color in AQI_BREAKPOINTS:
        if pm_low <= pm25 <= pm_high:
            aqi = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low
            return round(aqi), category, color
    
    return 500, "Hazardous", "#7E0023"


def get_sensors(api_key, bbox):
    """Get all sensors in bounding box."""
    headers = {"X-API-Key": api_key}
    nwlat, nwlng, selat, selng = bbox
    
    params = {
        "fields": "name,latitude,longitude",
        "location_type": 0,  # Outdoor sensors only
        "nwlat": nwlat,
        "nwlng": nwlng,
        "selat": selat,
        "selng": selng,
    }
    
    response = requests.get(
        "https://api.purpleair.com/v1/sensors",
        headers=headers,
        params=params,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    data = response.json()
    
    if "data" in data and data["data"]:
        sensors = []
        for row in data["data"]:
            sensors.append({
                'sensor_index': row[0],
                'name': row[1],
                'latitude': row[2],
                'longitude': row[3]
            })
        return pd.DataFrame(sensors)
    
    return None


def fetch_monthly_data(api_key, sensors_df, year, month, output_csv):
    """
    Fetch all available data for entire month (real-time granularity).
    Save to CSV for reuse.
    """
    # Calculate month boundaries
    _, last_day = monthrange(year, month)
    start_time = datetime(year, month, 1, 0, 0, 0)
    end_time = datetime(year, month, last_day, 23, 59, 59)
    
    print(f"\nFetching data for {start_time.strftime('%B %Y')}...")
    print(f"  Time range: {start_time} to {end_time}")
    print(f"  Sensors: {len(sensors_df)}")
    print(f"  Method: All available data (for daily worst calculation)")
    print()
    
    all_data = []
    
    for idx, sensor in sensors_df.iterrows():
        sensor_idx = sensor['sensor_index']
        sensor_name = sensor['name']
        
        print(f"  [{idx+1}/{len(sensors_df)}] {sensor_name} (ID: {sensor_idx})...", end='', flush=True)
        
        headers = {"X-API-Key": api_key}
        params = {
            "start_timestamp": int(start_time.timestamp()),
            "end_timestamp": int(end_time.timestamp()),
            "average": 10,  # Real-time data (most granular)
            "fields": "pm2.5_atm"
        }
        
        try:
            response = requests.get(
                f"https://api.purpleair.com/v1/sensors/{sensor_idx}/history",
                headers=headers,
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and data["data"]:
                    for row in data["data"]:
                        all_data.append({
                            'sensor_index': sensor_idx,
                            'timestamp': datetime.fromtimestamp(row[0]),
                            'pm2.5': row[1]
                        })
                    print(f" {len(data['data'])} points")
                else:
                    print(" no data")
            else:
                print(f" error {response.status_code}")
        
        except Exception as e:
            print(f" failed: {e}")
        
        # Rate limiting - be nice to the API
        time.sleep(0.5)
    
    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved {len(df)} data points to {output_csv}")
        print(f"  File size: {os.path.getsize(output_csv) / 1024 / 1024:.2f} MB")
        return df
    
    return None


def calculate_monthly_averages(sensors_df, historical_df):
    """
    Calculate monthly average using EPA-style methodology:
    - For each day, find WORST (maximum) PM2.5 reading per sensor
    - Average those daily worst readings for the month
    
    This is more health-relevant than simple averaging and reveals pollution events.
    """
    print("\nCalculating monthly averages (daily worst method)...")
    print("  Method: Average of each day's WORST reading")
    
    # Add date column
    historical_df['date'] = historical_df['timestamp'].dt.date
    
    # For each sensor, for each day, get the WORST (max) reading
    daily_worst = historical_df.groupby(['sensor_index', 'date'])['pm2.5'].max().reset_index()
    daily_worst.rename(columns={'pm2.5': 'daily_worst_pm25'}, inplace=True)
    
    # Now calculate monthly stats from those daily worst readings
    monthly_avg = daily_worst.groupby('sensor_index')['daily_worst_pm25'].agg([
        ('avg_pm25', 'mean'),          # This is now: avg of daily worst readings
        ('worst_day_pm25', 'max'),     # Absolute worst day in the month
        ('best_day_pm25', 'min'),      # Best of the daily worst readings
        ('days_with_data', 'count')    # Number of days with data
    ]).reset_index()
    
    # Merge with sensor locations
    result = sensors_df.merge(monthly_avg, on='sensor_index', how='left')
    
    # Calculate AQI from average of daily worst readings
    result['aqi'] = result['avg_pm25'].apply(lambda x: calculate_aqi(x)[0])
    result['category'] = result['avg_pm25'].apply(lambda x: calculate_aqi(x)[1])
    result['color'] = result['avg_pm25'].apply(lambda x: calculate_aqi(x)[2])
    
    # Also get AQI for worst single day (for reference)
    result['worst_day_aqi'] = result['worst_day_pm25'].apply(lambda x: calculate_aqi(x)[0])
    
    valid_count = len(result[result['avg_pm25'].notna()])
    print(f"  Calculated averages for {valid_count} sensors")
    
    if valid_count > 0:
        avg_days = result['days_with_data'].mean()
        print(f"  Average days with data per sensor: {avg_days:.1f}")
    
    return result


def create_monthly_map(sensors_avg_df, bbox, region_name, month_str, output_path, 
                       background_image=None, dpi=300):
    """
    Create a single static map showing monthly average AQI at each sensor.
    
    Args:
        sensors_avg_df: DataFrame with sensor locations and monthly averages
        bbox: Bounding box (nwlat, nwlng, selat, selng)
        region_name: Name of region (e.g., "Golden Region")
        month_str: Month string (e.g., "January 2026")
        output_path: Where to save the image
        background_image: Optional path to georeferenced background image
        dpi: Image resolution
    """
    print(f"\nCreating map...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    
    # Load and display background image
    if background_image is not None:
        try:
            bg_img = Image.open(background_image)
            bg_array = np.array(bg_img)
            
            nwlat, nwlng, selat, selng = bbox
            ax.imshow(bg_array,
                     extent=[nwlng, selng, selat, nwlat],
                     zorder=0,
                     alpha=1.0)
            print(f"  Background image loaded: {background_image}")
        except Exception as e:
            print(f"  Warning: Could not load background image: {e}")
    
    # Filter to sensors with data
    valid_sensors = sensors_avg_df[sensors_avg_df['avg_pm25'].notna()].copy()
    
    if len(valid_sensors) == 0:
        print("  Error: No valid sensor data to plot!")
        return False
    
    # Plot sensors with AQI colors
    for _, row in valid_sensors.iterrows():
        # Larger circles for monthly averages (easier to see)
        ax.scatter(
            row['longitude'],
            row['latitude'],
            c=row['color'],
            s=400,  # Larger than animation frames
            alpha=0.85,
            edgecolors='black',
            linewidths=2.5,
            zorder=5
        )
        
        # Add AQI value as text
        if pd.notna(row['aqi']):
            ax.text(
                row['longitude'],
                row['latitude'],
                f"{int(row['aqi'])}",
                fontsize=10,
                ha='center',
                va='center',
                fontweight='bold',
                color='black',
                zorder=6
            )
    
    # Set map bounds
    nwlat, nwlng, selat, selng = bbox
    margin = 0.02
    ax.set_xlim(nwlng - margin, selng + margin)
    ax.set_ylim(selat - margin, nwlat + margin)
    
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
    
    # Title
    title = f'Monthly Average Air Quality Index (AQI)\n{region_name} - {month_str}\n(Average of Daily Worst Readings - EPA Method)'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend - AQI categories
    legend_elements = [
        mpatches.Patch(color='#00E400', label='Good (0-50)'),
        mpatches.Patch(color='#FFFF00', label='Moderate (51-100)'),
        mpatches.Patch(color='#FF7E00', label='Unhealthy for Sensitive (101-150)'),
        mpatches.Patch(color='#FF0000', label='Unhealthy (151-200)'),
        mpatches.Patch(color='#8F3F97', label='Very Unhealthy (201-300)'),
        mpatches.Patch(color='#7E0023', label='Hazardous (301+)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             framealpha=0.95, title='AQI Categories', title_fontsize=12)
    
    # Statistics box
    valid_aqi = valid_sensors['aqi'].dropna()
    if len(valid_aqi) > 0:
        worst_day_aqi = valid_sensors['worst_day_aqi'].max()
        avg_days = valid_sensors['days_with_data'].mean()
        stats_text = (
            f"Method: Avg of Daily Worst\n"
            f"Sensors: {len(valid_aqi)}\n"
            f"Monthly Avg AQI: {valid_aqi.mean():.1f}\n"
            f"Range: {valid_aqi.min():.0f} - {valid_aqi.max():.0f}\n"
            f"Worst Day AQI: {worst_day_aqi:.0f}\n"
            f"Avg Days/Sensor: {avg_days:.0f}"
        )
        ax.text(
            0.99, 0.01, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='wheat', alpha=0.95),
            family='monospace'
        )
    
    # Save
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Map saved to: {output_path}")
    print(f"  Resolution: {dpi} DPI")
    print("bbox",bbox)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate monthly average AQI map using daily worst readings (EPA-style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methodology:
  This script uses a health-conservative approach:
  1. Fetch all available sensor data for the month  
  2. For each day, find the WORST (maximum) PM2.5 reading
  3. Calculate monthly average from those daily worst readings
  
  This reveals pollution events better than simple averaging and is
  more relevant for health impact assessment.

Examples:
  # Generate January 2026 map for Golden
  python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png --background map.tif
  
  # Use cached data (no API calls)
  python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png --use-cached
        """
    )
    
    parser.add_argument('--month', type=str, required=True,
                       help='Month to analyze (format: YYYY-MM, e.g., 2026-01)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image filename (e.g., jan2026_aqi.png)')
    parser.add_argument('--background', type=str, default=None,
                       help='Path to georeferenced background image (TIF/PNG/JPG)')
    parser.add_argument('--region', type=str, default='golden_town',
                       choices=['golden_town','golden', 'vancouver', 'victoria', 'custom'],
                       help='Predefined region (default: golden)')
    parser.add_argument('--bbox', type=str, default=None,
                       help='Custom bounding box: "nwlat,nwlng,selat,selng"')
    parser.add_argument('--region-name', type=str, default=None,
                       help='Custom region name for title')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output image resolution (default: 150)')
    parser.add_argument('--use-cached', action='store_true',
                       help='Use existing cached data (skip API calls)')
    parser.add_argument('--data-dir', type=str, default='./purpleair_data',
                       help='Directory for cached data (default: ./purpleair_data)')
    
    args = parser.parse_args()
    
    # Check API key (unless using cached data)
    api_key = os.environ.get('PURPLEAIR_API_KEY')
    if not api_key and not args.use_cached:
        print("Error: Set PURPLEAIR_API_KEY environment variable")
        print("Or use --use-cached to use existing data")
        sys.exit(1)
    
    # Parse month
    try:
        year, month = map(int, args.month.split('-'))
        month_name = datetime(year, month, 1).strftime('%B %Y')
    except ValueError:
        print(f"Error: Invalid month format '{args.month}'. Use YYYY-MM (e.g., 2026-01)")
        sys.exit(1)
    
    # Define regions
    REGIONS = {
        #from matz caltopo map https://caltopo.com/m/A02KGUS
        'golden_town': {
            'bbox': (51.31, -117.00, 51.28, -116.94), #from matz caltopo map https://caltopo.com/m/A02KGUS
            'name': 'Golden Region, BC'
        },
        'golden': {
            'bbox': (51.5, -117.5, 51.0, -116.25),
            'name': 'Golden Region, BC'
        },
        'vancouver': {
            'bbox': (49.35, -123.25, 49.00, -122.75),
            'name': 'Vancouver/Surrey Region, BC'
        },
        'victoria': {
            'bbox': (48.55, -123.50, 48.40, -123.30),
            'name': 'Victoria Region, BC'
        }
    }
    
    # Get bounding box and region name
    if args.region == 'custom':
        if not args.bbox:
            print("Error: --bbox required when using --region custom")
            sys.exit(1)
        try:
            bbox_parts = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox_parts) != 4:
                raise ValueError
            BBOX = tuple(bbox_parts)
        except:
            print("Error: --bbox must be 'nwlat,nwlng,selat,selng' (4 numbers)")
            sys.exit(1)
        REGION_NAME = args.region_name or "Custom Region"
    else:
        print("args.region", args.region)
        BBOX = REGIONS[args.region]['bbox']
        REGION_NAME = args.region_name or REGIONS[args.region]['name']
    
    print("=" * 80)
    print("PurpleAir Monthly Average Map Generator")
    print("=" * 80)
    print(f"Region: {REGION_NAME}")
    print(f"Month: {month_name}")
    print(f"Bounding Box: {BBOX}")
    print()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # File paths
    sensors_csv = os.path.join(args.data_dir, f"{args.region}_sensors.csv")
    data_csv = os.path.join(args.data_dir, f"{args.region}_{year}-{month:02d}_rawdata.csv")
    summary_csv = os.path.join(args.data_dir, f"{args.region}_{year}-{month:02d}_daily_worst_avg.csv")
    
    # Step 1: Get sensors
    if args.use_cached and os.path.exists(sensors_csv):
        print(f"Using cached sensors from {sensors_csv}")
        sensors_df = pd.read_csv(sensors_csv)
        print(f"  Loaded {len(sensors_df)} sensors")
    else:
        print("Fetching sensors from PurpleAir API...")
        sensors_df = get_sensors(api_key, BBOX)
        if sensors_df is None or len(sensors_df) == 0:
            print("Error: No sensors found in region!")
            sys.exit(1)
        sensors_df.to_csv(sensors_csv, index=False)
        print(f"  ✓ Found {len(sensors_df)} sensors")
    
    # Step 2: Fetch or load historical data
    if args.use_cached and os.path.exists(data_csv):
        print(f"\nUsing cached data from {data_csv}")
        historical_df = pd.read_csv(data_csv)
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
        print(f"  Loaded {len(historical_df)} data points")
    else:
        historical_df = fetch_monthly_data(api_key, sensors_df, year, month, data_csv)
        if historical_df is None or len(historical_df) == 0:
            print("Error: No historical data retrieved!")
            sys.exit(1)
    
    # Step 3: Calculate monthly averages
    sensors_avg_df = calculate_monthly_averages(sensors_df, historical_df)
    
    # Save summary
    summary_csv = os.path.join(args.data_dir, f"{args.region}_{year}-{month:02d}_averages.csv")
    sensors_avg_df.to_csv(summary_csv, index=False)
    print(f"  ✓ Saved averages to {summary_csv}")
    
    # Step 4: Create map
    success = create_monthly_map(
        sensors_avg_df,
        BBOX,
        REGION_NAME,
        month_name,
        args.output,
        background_image=args.background,
        dpi=args.dpi
    )
    
    if success:
        print()
        print("=" * 80)
        print("✓ Complete!")
        print("=" * 80)
        print(f"Output: {args.output}")
        print(f"Data: {data_csv} (reuse with --use-cached)")
        print(f"Summary: {summary_csv}")
        print()
        print("CSV columns in summary:")
        print("  - avg_pm25: Monthly average of each day's WORST reading")  
        print("  - worst_day_pm25: Absolute worst PM2.5 in the month")
        print("  - best_day_pm25: Best of the daily worst readings")
        print("  - days_with_data: Number of days with sensor data")
        print("  - aqi, worst_day_aqi: Calculated AQI values")
        print()
    else:
        print("Error: Failed to create map")
        sys.exit(1)


if __name__ == "__main__":
    main()
