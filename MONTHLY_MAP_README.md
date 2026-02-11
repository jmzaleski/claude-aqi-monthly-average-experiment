# PurpleAir Monthly Average Map Generator

Generate a single static map showing monthly average air quality for your region.

Perfect for:
- Website display
- Monthly reports
- Comparing air quality across months
- Historical analysis

## Quick Start

```bash
# 1. Set your API key
export PURPLEAIR_API_KEY="your-key-here"

# 2. Generate January 2026 map for Golden
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output golden_jan2026.png \
    --background golden_map.tif

# 3. View the result
open golden_jan2026.png
```

## Usage Examples

### Basic - No Background
```bash
python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png
```

### With Background Image
```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output golden_jan2026.png
```

### Use Cached Data (No API Calls)
Once you've fetched data for a month, reuse it:
```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output golden_jan2026_v2.png \
    --use-cached
```

### Different Region
```bash
# Vancouver
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --region vancouver \
    --background vancouver_map.tif \
    --output vancouver_jan2026.png

# Victoria
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --region victoria \
    --output victoria_jan2026.png
```

### Custom Region
```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --region custom \
    --bbox "51.5,-117.5,51.0,-116.25" \
    --region-name "My Custom Area" \
    --output custom_jan2026.png
```

### High Resolution for Print
```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output golden_jan2026_hires.png \
    --dpi 300
```

## Command-Line Options

```
--month YYYY-MM          Month to analyze (e.g., 2026-01)
--output FILENAME        Output image file (e.g., jan2026.png)
--background IMAGE       Background map image (TIF/PNG/JPG)
--region NAME            Predefined region: golden, vancouver, victoria
--bbox "lat,lng,lat,lng" Custom bounding box
--region-name "Name"     Custom name for title
--dpi NUMBER            Image resolution (default: 150)
--use-cached            Use existing data, skip API calls
--data-dir PATH         Where to store cached data
```

## Workflow for Webmaster

### First Time (Fetch Data)
```bash
# Set API key once
export PURPLEAIR_API_KEY="your-key-here"

# Fetch January 2026 data
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output website/jan2026_aqi.png
```

This will:
1. Fetch all sensors in Golden region
2. Get hourly data for all of January 2026
3. Calculate monthly averages
4. Create the map
5. Save data to `./purpleair_data/` for reuse

### Subsequent Updates (Use Cached Data)
If you want to regenerate the image (different DPI, updated background, etc.):

```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map_updated.tif \
    --output website/jan2026_aqi_v2.png \
    --use-cached
```

No API calls needed! Uses saved data.

### Monthly Updates
```bash
# February 2026
python3 purpleair_monthly_map.py \
    --month 2026-02 \
    --background golden_map.tif \
    --output website/feb2026_aqi.png

# March 2026
python3 purpleair_monthly_map.py \
    --month 2026-03 \
    --background golden_map.tif \
    --output website/mar2026_aqi.png
```

## Output Files

### Image
- High-quality PNG suitable for web
- Default 150 DPI (increase with --dpi for print)
- Shows each sensor as colored circle with AQI number
- Includes legend, statistics, title

### Data Files (in ./purpleair_data/)
- `golden_sensors.csv` - Sensor locations (reusable)
- `golden_2026-01_data.csv` - Hourly data for the month
- `golden_2026-01_averages.csv` - Monthly averages per sensor

### CSV Format (averages)
```
sensor_index,name,latitude,longitude,avg_pm25,min_pm25,max_pm25,count,aqi,category,color
12345,Golden Sensor 1,51.3,-116.9,8.5,2.1,18.3,672,36,Good,#00E400
```

Perfect for:
- Importing to spreadsheet
- Further analysis
- Website data tables

## Understanding the Map

### Colors (Same as PurpleAir Website)
- 🟢 **Green (0-50)**: Good - Air quality is satisfactory
- 🟡 **Yellow (51-100)**: Moderate - Acceptable for most
- 🟠 **Orange (101-150)**: Unhealthy for Sensitive Groups
- 🔴 **Red (151-200)**: Unhealthy
- 🟣 **Purple (201-300)**: Very Unhealthy
- 🔴 **Maroon (301+)**: Hazardous

### Numbers on Circles
The AQI value (0-500) based on monthly average PM2.5

### Statistics Box
- Number of sensors with valid data
- Regional average AQI
- Range (min to max)
- Average data points per sensor

## Tips

### For Website Display
```bash
# Generate web-optimized version
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output jan2026_web.png \
    --dpi 96
```

### For Reports/Print
```bash
# High-res for printing
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output jan2026_print.png \
    --dpi 300
```

### Comparing Months
Generate multiple months and display side-by-side:
```bash
for month in 2025-12 2026-01 2026-02; do
    python3 purpleair_monthly_map.py \
        --month $month \
        --background golden_map.tif \
        --output comparison_${month}.png \
        --use-cached
done
```

## Troubleshooting

### "No sensors found"
- Check your bounding box coordinates
- Visit https://map.purpleair.com/ to verify sensors exist
- Try widening the bounding box

### "No historical data"
- Check the month is valid (can't fetch future data)
- Some sensors may be offline for that month
- Verify API key is set correctly

### Background image doesn't align
- Make sure your background image covers the exact bounding box
- See BACKGROUND_IMAGE_GUIDE.md for creating aligned images
- Test without background first to verify sensor positions

### Rate limiting errors
- The script includes 0.5s delays between sensor queries
- If you hit limits, wait a few minutes and use --use-cached

## API Usage

### Points Used
For Golden region (~10 sensors) for 1 month:
- Fetch sensors: ~1 point
- Fetch 1 month history × 10 sensors: ~10 points
- **Total: ~11 points per month**

With 1 million free points, you can generate maps for ~90,000 months! 😊

### Cached Data
Once fetched, data is saved. Regenerating the map with different settings (DPI, background) uses **0 points**.

## Requirements

```bash
pip install requests pandas numpy matplotlib Pillow
```

See requirements-core.txt for exact versions.

## Integration with Website

### Manual Upload
1. Generate map: `python3 purpleair_monthly_map.py ...`
2. Upload `jan2026_aqi.png` to website
3. Update HTML: `<img src="jan2026_aqi.png" alt="January 2026 Air Quality">`

### Automated Monthly
Create a cron job or scheduled task:
```bash
# /etc/cron.monthly/generate_aqi_map.sh
#!/bin/bash
MONTH=$(date +%Y-%m)
python3 /path/to/purpleair_monthly_map.py \
    --month $MONTH \
    --background /path/to/golden_map.tif \
    --output /var/www/html/current_aqi.png
```

## Support

For questions about:
- **API**: https://community.purpleair.com/
- **Script usage**: See this README
- **Background images**: See BACKGROUND_IMAGE_GUIDE.md
