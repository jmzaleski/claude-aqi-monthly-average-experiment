# Hourly Pattern Grid Map - Usage Guide

## What It Does

Creates a **4×2 grid of maps** showing how air quality changes throughout the day:

```
┌─────────┬─────────┬─────────┬─────────┐
│  12am   │   3am   │   6am   │   9am   │
│ (night) │ (dawn)  │ (morn)  │ (mid)   │
├─────────┼─────────┼─────────┼─────────┤
│  12pm   │   3pm   │   6pm   │   9pm   │
│ (noon)  │ (after) │ (eve)   │ (night) │
└─────────┴─────────┴─────────┴─────────┘
```

Each map shows **average AQI for that hour** across all days in the month.

## Quick Start

```bash
# Set API key
export PURPLEAIR_API_KEY="your-key-here"

# Generate hourly grid for January 2026
python3 purpleair_hourly_grid.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output jan2026_hourly_patterns.png

# Use cached data (no API calls)
python3 purpleair_hourly_grid.py \
    --month 2026-01 \
    --background golden_map.tif \
    --output jan2026_hourly_patterns.png \
    --use-cached
```

## What You'll See

### Expected Patterns:

**Wood Stove Burning:**
- **6am-9am**: Red sensors appear (people wake up, start fires)
- **6pm-9pm**: More red sensors (coming home, evening heating)
- **12am-6am**: Green everywhere (fires die out)

**Traffic Patterns:**
- **7am-9am**: Sensors near roads turn orange/red
- **5pm-7pm**: Another spike near roads

**Residential vs Industrial:**
- **Residential areas**: Peak at 7am and 7pm
- **Industrial areas**: Constant all day (if present)

### Your Specific Case (Golden):
You hypothesized wood stoves - you should see:
- **Your house (sensor 156)**: Green at 3am, red at 7pm
- **Downtown**: Possibly moderate all day
- **Clear geographic patterns**: Residential neighborhoods vs main roads

## Data Reuse

The script uses the **same cached data** as `purpleair_monthly_map.py`:
- Sensors: `purpleair_data/golden_sensors.csv`
- Raw data: `purpleair_data/golden_2026-01_rawdata.csv`

If you already fetched data for the monthly map, just add `--use-cached` = instant!

## Output

**Single PNG file** with:
- 8 maps showing key hours throughout the day
- Each map has same geographic area and sensors
- Colors show AQI for that specific hour
- Shared legend at bottom
- Stats for each hour

Perfect for:
- Website display
- Identifying pollution sources
- Understanding daily patterns
- Comparing weekday vs weekend (future enhancement)

## Command-Line Options

```bash
--month YYYY-MM          # Month to analyze (required)
--output FILENAME        # Output PNG file (required)
--background IMAGE       # Georeferenced map image
--region NAME            # golden, vancouver, victoria (default: golden)
--dpi NUMBER            # Resolution (default: 150)
--use-cached            # Use existing data files
--data-dir PATH         # Where cached data is stored
```

## File Size

Expect output to be **~500KB - 2MB** depending on:
- DPI setting (150 = web, 300 = print)
- Background image complexity
- Number of sensors

## Interpretation Tips

1. **Look for red vertical patterns** = that sensor is always bad
2. **Look for red horizontal patterns** = that hour is bad everywhere
3. **Compare morning (6am-9am) vs evening (6pm-9pm)** = different sources?
4. **Check overnight (12am-6am)** = baseline air quality
5. **Your house location** = does it match your wood stove schedule?

## Next Steps

After running, you might want:
- Weekday vs weekend comparison (different script)
- Specific hour deep-dive (filter to just 7pm)
- 24-hour animation (all hours cycling)
- Heatmap version (sensors × hours table)

Let me know what insights you find!
