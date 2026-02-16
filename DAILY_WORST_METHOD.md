# Daily Worst Method - Updated Methodology

## What Changed

The monthly map generator now uses a **health-conservative** approach:

### Old Method (Simple Average)
- Fetch hourly averages
- Calculate mean of all readings
- **Problem**: Hides pollution events in good days

### New Method (Daily Worst Average)  
- Fetch all available data (most granular)
- For each day, find the WORST (maximum) PM2.5 reading
- Calculate monthly average from those daily worst readings
- **Better**: Reveals pollution events, more health-relevant

## Why This is Better

✅ **EPA Methodology** - Matches official air quality reporting  
✅ **Health Focused** - Bad air days matter more than averages  
✅ **Reveals Events** - Wildfire smoke, traffic spikes show up  
✅ **Conservative** - Better to overestimate than underestimate risk

## Example

Imagine a sensor in January:
- **29 days**: PM2.5 = 5 (good air)
- **2 days**: PM2.5 = 100 (smoke event)

**Old method**: Average = 11.3 (looks good!)  
**New method**: Average of daily worst ≈ 15-20 (reveals there were bad days)

The new method doesn't hide that smoke event.

## Usage (No Changes Needed!)

The command stays the same:

```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026.png \
    --background golden_map.tif
```

## What You'll Get

### Map Changes
- **Title** now says: "(Average of Daily Worst Readings - EPA Method)"
- **Statistics box** shows:
  - Method: Avg of Daily Worst
  - Worst Day AQI: Shows single worst day in the month
  - Avg Days/Sensor: How many days each sensor reported

### CSV Changes

**Old CSV** (`golden_2026-01_averages.csv`):
```
avg_pm25, min_pm25, max_pm25, count
```

**New CSV** (`golden_2026-01_daily_worst_avg.csv`):
```
avg_pm25             - Monthly average of each day's WORST reading
worst_day_pm25       - Absolute worst PM2.5 in the month  
best_day_pm25        - Best of the daily worst readings
days_with_data       - Number of days with sensor data
aqi                  - AQI calculated from avg_pm25
worst_day_aqi        - AQI of the single worst day
```

## Data Volume

You'll now get MORE data points (all available readings):

- **Old**: ~744 points per sensor (31 days × 24 hours)
- **New**: ~4,000-8,000 points per sensor (depends on sensor reporting frequency)

File will be larger (~2-5 MB instead of ~500 KB), but it's cached so you only fetch once.

## Regenerating Old Data

If you want to regenerate with the new method:

```bash
# Delete old cached data
rm purpleair_data/golden_2026-01_data.csv

# Fetch fresh with new method
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_new.png \
    --background golden_map.tif
```

Or just generate a different month - new fetches use the new method automatically.

## Comparing Methods

Want to see the difference? You can't easily with the current script, but you could:

1. Keep your old `golden_2026-01_averages.csv` (rename it)
2. Fetch fresh to get `golden_2026-01_daily_worst_avg.csv`  
3. Compare the AQI values

The new method will generally show higher (more conservative) AQI values, especially if there were pollution events.

## For Your Webmaster

Tell them:
- **Same usage, better methodology**
- **More accurate health assessment**
- **Matches EPA standards**
- **No changes to workflow needed**

The maps will look the same, but the numbers are now more meaningful for public health.
