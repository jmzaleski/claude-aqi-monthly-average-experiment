#!/usr/bin/env python3
"""
PurpleAir Hourly Pattern Grid Map Generator

Creates a grid of maps showing average AQI at each hour of the day.
Reveals time-based patterns like wood stove burning, traffic, etc.

Layout: 6×4 grid showing all 24 hours (12am, 1am, 2am... 11pm)

Usage:
    export PURPLEAIR_API_KEY="your-key-here"
    
    python3 purpleair_hourly_grid.py \
        --month 2026-01 \
        --background golden_map.tif \
        --output jan2026_hourly_grid.png
    
    # Use cached data
    python3 purpleair_hourly_grid.py \
        --month 2026-01 \
        --background golden_map.tif \
        --output jan2026_hourly_grid.png \
        --use-cached
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


# TIMEZONE OFFSET: PurpleAir API returns UTC timestamps
# Golden, BC is in Mountain Time (MST/MDT)
# Set this to the hour offset from UTC to local time
# MST = UTC-7, MDT = UTC-8 (use -7 for most of the year)
TIMEZONE_OFFSET_HOURS = -7  # Adjust to local time from UTC

# DATA FILTERING: Exclude invalid/extreme PM2.5 readings
# Sensor errors and malfunctions can produce unrealistic values  
# PM2.5 >250 would show as AQI 300+ (Hazardous), likely sensor error
MAX_VALID_PM25 = 250.0  # µg/m³ - strict cutoff for sensor errors
MIN_VALID_PM25 = 0.0    # µg/m³ - negative values are sensor errors


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
        "location_type": 0,
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
    """Fetch hourly data for entire month (reuses from main script)."""
    _, last_day = monthrange(year, month)
    start_time = datetime(year, month, 1, 0, 0, 0)
    end_time = datetime(year, month, last_day, 23, 59, 59)
    
    print(f"\nFetching data for {start_time.strftime('%B %Y')}...")
    print(f"  Time range: {start_time} to {end_time}")
    print(f"  Sensors: {len(sensors_df)}")
    print(f"  Method: 10-minute averages (for hourly patterns)")
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
            "average": 10,
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
        
        time.sleep(0.5)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved {len(df)} data points to {output_csv}")
        print(f"  File size: {os.path.getsize(output_csv) / 1024 / 1024:.2f} MB")
        return df
    
    return None


def calculate_hourly_averages(sensors_df, historical_df):
    """
    Calculate average PM2.5 for each hour of the day (0-23).
    
    Returns DataFrame with columns: sensor_index, hour, avg_pm25, aqi, color
    """
    print("\nCalculating hourly averages...")
    print("  Method: Average PM2.5 for each hour across all days in month")
    print(f"  Timezone: Applying {TIMEZONE_OFFSET_HOURS:+d} hour offset from UTC")
    
    # FILTER OUT BAD DATA
    # Exclude PM2.5 values that are sensor errors or malfunctions
    initial_count = len(historical_df)
    historical_df = historical_df[
        (historical_df['pm2.5'].notna()) & 
        (historical_df['pm2.5'] >= MIN_VALID_PM25) & 
        (historical_df['pm2.5'] <= MAX_VALID_PM25)
    ].copy()
    filtered_count = initial_count - len(historical_df)
    
    if filtered_count > 0:
        pct_filtered = (filtered_count / initial_count) * 100
        print(f"  Filtered out {filtered_count} invalid readings ({pct_filtered:.1f}%)")
        print(f"    (PM2.5 <{MIN_VALID_PM25} or >{MAX_VALID_PM25} µg/m³)")
        print(f"  Remaining valid readings: {len(historical_df)}")
    
    # Apply timezone offset to convert UTC to local time
    historical_df['local_timestamp'] = historical_df['timestamp'] + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)
    
    # Extract hour and date from local timestamp
    historical_df['hour'] = historical_df['local_timestamp'].dt.hour
    historical_df['date'] = historical_df['local_timestamp'].dt.date
    
    # VALIDATION: Calculate daily max for comparison
    print("\n  Validation: Comparing hourly averages with daily max method...")
    daily_max = historical_df.groupby(['sensor_index', 'date'])['pm2.5'].max().reset_index()
    daily_max_monthly_avg = daily_max.groupby('sensor_index')['pm2.5'].mean().reset_index()
    daily_max_monthly_avg.rename(columns={'pm2.5': 'monthly_avg_of_daily_max'}, inplace=True)
    
    # PRINT DAILY MAX TABLE
    print("\n" + "="*120)
    print("DAILY MAX TABLE (AQI values - worst reading each day)")
    print("="*120)
    
    # Filter to only dates in the target month (timezone shift can cause dates from adjacent months)
    daily_max['month'] = pd.to_datetime(daily_max['date']).dt.month
    daily_max['year'] = pd.to_datetime(daily_max['date']).dt.year
    
    # Determine the target month/year from the data
    target_month = daily_max['month'].mode()[0]  # Most common month
    target_year = daily_max['year'].mode()[0]    # Most common year
    
    daily_max_filtered = daily_max[
        (daily_max['month'] == target_month) & 
        (daily_max['year'] == target_year)
    ].copy()
    
    print(f"  Filtered to {target_year}-{target_month:02d} (removed dates from adjacent months due to timezone shift)")
    
    # Add day of month column
    daily_max_filtered['day'] = pd.to_datetime(daily_max_filtered['date']).dt.day
    
    # Convert PM2.5 to AQI
    daily_max_filtered['aqi'] = daily_max_filtered['pm2.5'].apply(lambda x: calculate_aqi(x)[0])
    
    # Pivot to create sensor × day table
    daily_pivot = daily_max_filtered.pivot(index='sensor_index', columns='day', values='aqi')
    
    # Add sensor names
    daily_with_names = daily_pivot.merge(sensors_df[['sensor_index', 'name']], 
                                         left_index=True, right_on='sensor_index')
    daily_with_names = daily_with_names.set_index('name')
    daily_with_names = daily_with_names.drop('sensor_index', axis=1)
    
    # Print header
    print(f"{'Sensor':<25}", end='')
    for day in range(1, 32):
        print(f"{day:>4}", end='')
    print()
    print("-" * 120)
    
    # Print each sensor's daily max AQI
    for sensor_name, row in daily_with_names.iterrows():
        name_display = sensor_name[:23]
        print(f"{name_display:<25}", end='')
        
        for day in range(1, 32):
            val = row[day] if day in row.index else None
            if pd.notna(val):
                print(f"{int(val):>4}", end='')
            else:
                print(f"{'--':>4}", end='')
        print()
    
    print("="*120)
    print("NOTE: This shows the WORST reading each day (converted to AQI).")
    print("Compare this to the hourly table below to understand timing patterns.")
    print("="*120)
    print()
    
    # For each sensor, for each hour, calculate average PM2.5
    hourly_avg = historical_df.groupby(['sensor_index', 'hour'])['pm2.5'].mean().reset_index()
    hourly_avg.rename(columns={'pm2.5': 'avg_pm25'}, inplace=True)
    
    # ADDITIONAL FILTERING: Remove hourly averages that are still >MAX_VALID_PM25
    # (in case a particular hour had consistently high but valid readings that average to >250)
    before_filter = len(hourly_avg)
    
    # Debug: show what we're filtering
    bad_averages = hourly_avg[hourly_avg['avg_pm25'] > MAX_VALID_PM25]
    if len(bad_averages) > 0:
        print(f"\n  Filtering out {len(bad_averages)} hourly averages that exceed {MAX_VALID_PM25}:")
        for _, row in bad_averages.iterrows():
            sensor_name = sensors_df[sensors_df['sensor_index'] == row['sensor_index']]['name'].iloc[0]
            print(f"    - {sensor_name}: hour {row['hour']}, PM2.5 = {row['avg_pm25']:.1f}")
    
    # Apply stricter filter - use < instead of <= to be safe
    hourly_avg = hourly_avg[hourly_avg['avg_pm25'] < MAX_VALID_PM25].copy()
    after_filter = len(hourly_avg)
    
    if before_filter > after_filter:
        print(f"  Total filtered: {before_filter - after_filter} hourly averages")
    
    # Calculate AQI from average PM2.5
    hourly_avg['aqi'] = hourly_avg['avg_pm25'].apply(lambda x: calculate_aqi(x)[0])
    hourly_avg['category'] = hourly_avg['avg_pm25'].apply(lambda x: calculate_aqi(x)[1])
    hourly_avg['color'] = hourly_avg['avg_pm25'].apply(lambda x: calculate_aqi(x)[2])
    
    # VALIDATION: For each sensor, find the max hourly average (for comparison with daily max)
    hourly_max_by_sensor = hourly_avg.groupby('sensor_index')['avg_pm25'].max().reset_index()
    hourly_max_by_sensor.rename(columns={'avg_pm25': 'max_hourly_avg'}, inplace=True)
    
    # Compare the two methods
    validation = hourly_max_by_sensor.merge(daily_max_monthly_avg, on='sensor_index')
    validation = validation.merge(sensors_df[['sensor_index', 'name']], on='sensor_index')
    
    print("\n  Sensor-by-sensor validation:")
    print("  " + "="*70)
    print(f"  {'Sensor':<20} {'Max Hourly Avg':>15} {'Monthly Avg Daily Max':>20}")
    print("  " + "-"*70)
    
    all_good = True
    for _, row in validation.iterrows():
        sensor_name = row['name'][:18]  # Truncate long names
        max_hourly = row['max_hourly_avg']
        daily_max_avg = row['monthly_avg_of_daily_max']
        
        # The max hourly average should be close to (but likely less than) the monthly avg of daily maxes
        # They measure different things but should be in the same ballpark
        ratio = max_hourly / daily_max_avg if daily_max_avg > 0 else 0
        status = "✓" if 0.5 <= ratio <= 1.5 else "⚠"
        
        print(f"  {status} {sensor_name:<20} {max_hourly:>15.1f} {daily_max_avg:>20.1f}")
        
        if not (0.3 <= ratio <= 2.0):  # Wider tolerance for assertion
            all_good = False
    
    print("  " + "="*70)
    
    # ASSERTION: The calculations should be in the same ballpark
    # Max hourly average might be less than monthly avg of daily max (which includes worst days)
    # But they should be reasonably close
    if not all_good:
        print("\n  ⚠ WARNING: Large discrepancies detected between hourly and daily calculations!")
        print("  This might indicate:")
        print("    - Timezone offset is wrong (peaks appearing at wrong hours)")
        print("    - Data quality issues")
        print("    - Sensor reporting irregularities")
        print("\n  Check TIMEZONE_OFFSET_HOURS constant and verify sensor data.")
    else:
        print("\n  ✓ Validation passed: Hourly and daily calculations are consistent")
    
    # Calculate AQI from average PM2.5
    hourly_avg['aqi'] = hourly_avg['avg_pm25'].apply(lambda x: calculate_aqi(x)[0])
    hourly_avg['category'] = hourly_avg['avg_pm25'].apply(lambda x: calculate_aqi(x)[1])
    hourly_avg['color'] = hourly_avg['avg_pm25'].apply(lambda x: calculate_aqi(x)[2])
    
    # Merge with sensor locations
    result = hourly_avg.merge(sensors_df, on='sensor_index', how='left')
    
    # PRINT DETAILED HOURLY TABLE FOR DEBUGGING
    print(f"\n  Calculated hourly averages for {len(sensors_df)} sensors × 24 hours")
    print("\n" + "="*120)
    print("HOURLY AVERAGE TABLE (AQI values)")
    print("="*120)
    
    # Safety check: Remove any remaining 500s before displaying
    hourly_avg_display = hourly_avg[hourly_avg['aqi'] < 500].copy()
    
    # Pivot to create sensor × hour table (using AQI values)
    pivot_table = hourly_avg_display.pivot(index='sensor_index', columns='hour', values='aqi')
    
    # Add sensor names
    pivot_with_names = pivot_table.merge(sensors_df[['sensor_index', 'name']], 
                                         left_index=True, right_on='sensor_index')
    pivot_with_names = pivot_with_names.set_index('name')
    pivot_with_names = pivot_with_names.drop('sensor_index', axis=1)
    
    # Print header
    print(f"{'Sensor':<25}", end='')
    for hour in range(24):
        if hour == 0:
            label = "12a"
        elif hour < 12:
            label = f"{hour}a"
        elif hour == 12:
            label = "12p"
        else:
            label = f"{hour-12}p"
        print(f"{label:>6}", end='')
    print()
    print("-" * 120)
    
    # Print each sensor's hourly AQI averages
    for sensor_name, row in pivot_with_names.iterrows():
        # Truncate sensor name
        name_display = sensor_name[:23]
        print(f"{name_display:<25}", end='')
        
        for hour in range(24):
            val = row[hour] if hour in row.index else None
            if pd.notna(val):
                # AQI is already an integer, just print it
                print(f"{int(val):>6}", end='')
            else:
                print(f"{'--':>6}", end='')
        print()
    
    print("="*120)
    print("\nNOTE: Compare these hourly AQI values to your monthly average AQI.")
    print("If monthly average (e.g., 156) is MUCH higher than any hourly value,")
    print("it means daily peaks occur at DIFFERENT hours on different days.")
    print("="*120)
    print()
    
    return result


def create_single_hour_map(ax, sensors_hourly_df, hour, bbox, background_array):
    """
    Create one small map for a specific hour.
    
    Args:
        ax: Matplotlib axis to draw on
        sensors_hourly_df: DataFrame filtered to one specific hour
        hour: Hour of day (0-23)
        bbox: Bounding box tuple
        background_array: Background image array (or None)
    """
    nwlat, nwlng, selat, selng = bbox
    
    # Set axis limits
    ax.set_xlim(nwlng, selng)
    ax.set_ylim(selat, nwlat)
    
    # Display background
    if background_array is not None:
        ax.imshow(background_array,
                 extent=[nwlng, selng, selat, nwlat],
                 origin='upper',
                 aspect='auto',
                 interpolation='none',
                 zorder=0,
                 alpha=1.0)
    
    # Plot sensors
    valid_sensors = sensors_hourly_df[sensors_hourly_df['avg_pm25'].notna()].copy()
    
    for _, row in valid_sensors.iterrows():
        ax.scatter(row['longitude'], row['latitude'],
                  c=row['color'], s=200, alpha=0.85,
                  edgecolors='black', linewidths=1.5, zorder=5)
        
        if pd.notna(row['aqi']):
            ax.text(row['longitude'], row['latitude'], f"{int(row['aqi'])}",
                   fontsize=7, ha='center', va='center',
                   fontweight='bold', color='black', zorder=6)
    
    # Title for this hour
    if hour == 0:
        hour_label = "12am"
    elif hour < 12:
        hour_label = f"{hour}am"
    elif hour == 12:
        hour_label = "12pm"
    else:
        hour_label = f"{hour-12}pm"
    
    ax.set_title(hour_label, fontsize=11, fontweight='bold')
    
    # Minimal labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Stats box
    valid_aqi = valid_sensors['aqi'].dropna()
    if len(valid_aqi) > 0:
        avg_aqi = valid_aqi.mean()
        stats = f"Avg: {avg_aqi:.0f}"
        ax.text(0.95, 0.05, stats, transform=ax.transAxes,
               fontsize=7, ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))


def create_hourly_grid(sensors_df, hourly_df, bbox, region_name, month_str,
                       output_path, background_image=None, dpi=150):
    """
    Create 6×4 grid of maps showing all 24 hours of the day.
    
    Layout: 4 columns × 6 rows = 24 hours
    """
    print(f"\nCreating 24-hour grid map...")
    
    # Unpack bounding box
    nwlat, nwlng, selat, selng = bbox
    
    # Load background image if provided
    bg_array = None
    if background_image is not None:
        try:
            bg_img = Image.open(background_image)
            bg_array = np.array(bg_img)
            print(f"  Background image loaded")
        except Exception as e:
            print(f"  Warning: Could not load background image: {e}")
    
    # All 24 hours
    hours_to_show = list(range(24))  # 0, 1, 2, ..., 23
    
    # Create figure with 4 columns × 6 rows
    fig = plt.figure(figsize=(16, 24))
    
    # Main title
    fig.suptitle(f'24-Hour Air Quality Patterns - {region_name} - {month_str}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Create 6×4 grid (4 columns, 6 rows)
    for idx, hour in enumerate(hours_to_show):
        # Position in grid (6 rows, 4 columns)
        row = idx // 4
        col = idx % 4
        
        # Create subplot
        ax = plt.subplot(6, 4, idx + 1)
        
        # Filter data for this hour
        hour_data = hourly_df[hourly_df['hour'] == hour].copy()
        
        # Create map for this hour
        create_single_hour_map(ax, hour_data, hour, bbox, bg_array)
    
    # Add legend (shared for all maps)
    legend_elements = [
        mpatches.Patch(color='#00E400', label='Good (0-50)'),
        mpatches.Patch(color='#FFFF00', label='Moderate (51-100)'),
        mpatches.Patch(color='#FF7E00', label='Unhealthy for Sensitive (101-150)'),
        mpatches.Patch(color='#FF0000', label='Unhealthy (151-200)'),
        mpatches.Patch(color='#8F3F97', label='Very Unhealthy (201-300)'),
        mpatches.Patch(color='#7E0023', label='Hazardous (301+)'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
              fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.005))
    
    # Adjust layout for 24 maps
    plt.tight_layout(rect=[0, 0.015, 1, 0.99])
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"  ✓ Grid map saved to: {output_path}")
    print(f"  Resolution: {dpi} DPI")
    print(f"  Layout: 4 columns × 6 rows = 24 hours")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate hourly pattern grid map for PurpleAir sensors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 purpleair_hourly_grid.py --month 2026-01 --output jan_hourly.png --background map.tif
  python3 purpleair_hourly_grid.py --month 2026-01 --output jan_hourly.png --use-cached
        """
    )
    
    parser.add_argument('--month', type=str, required=True,
                       help='Month to analyze (format: YYYY-MM)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image filename')
    parser.add_argument('--background', type=str, default=None,
                       help='Path to georeferenced background image')
    parser.add_argument('--region', type=str, default='golden',
                       choices=['golden', 'vancouver', 'victoria', 'custom'],
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
                       help='Directory for cached data')
    parser.add_argument('--data-prefix', type=str, default=None,
                       help='Prefix for data filenames (default: same as --region, e.g., "golden_town")')
    
    args = parser.parse_args()
    
    # Check API key
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
        print(f"Error: Invalid month format '{args.month}'. Use YYYY-MM")
        sys.exit(1)
    
    # Define regions
    REGIONS = {
        'golden': {
            'bbox': (51.31484, -117.00112, 51.28285, -116.94258),
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
            print("Error: --bbox must be 'nwlat,nwlng,selat,selng'")
            sys.exit(1)
        REGION_NAME = args.region_name or "Custom Region"
    else:
        BBOX = REGIONS[args.region]['bbox']
        REGION_NAME = args.region_name or REGIONS[args.region]['name']
    
    print("=" * 80)
    print("PurpleAir Hourly Pattern Grid Generator")
    print("=" * 80)
    print(f"Region: {REGION_NAME}")
    print(f"Month: {month_name}")
    print(f"Layout: 6 rows × 4 columns = 24 hours (complete day)")
    print()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # File paths - use data-prefix if specified, otherwise use region name
    data_prefix = args.data_prefix or args.region
    sensors_csv = os.path.join(args.data_dir, f"{data_prefix}_sensors.csv")
    data_csv = os.path.join(args.data_dir, f"{data_prefix}_{year}-{month:02d}_rawdata.csv")
    
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
    
    # Step 3: Calculate hourly averages
    hourly_df = calculate_hourly_averages(sensors_df, historical_df)
    
    # Step 4: Create grid map
    success = create_hourly_grid(
        sensors_df,
        hourly_df,
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
        print()
        print("Interpretation:")
        print("  - Morning peaks (6am-9am): Wood stoves starting, morning traffic")
        print("  - Evening peaks (6pm-9pm): Evening heating, cooking, traffic")
        print("  - Overnight lows (12am-6am): Minimal activity")
        print("  - Compare sensor locations to see residential vs industrial patterns")
        print()
    else:
        print("Error: Failed to create map")
        sys.exit(1)


if __name__ == "__main__":
    main()
