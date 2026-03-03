# purpleair_aqi_charts.py

Reads a PurpleAir cached raw data CSV and generates two self-contained interactive HTML files:

- **Heatmap** — sensors × hour-of-day grid, coloured with official US EPA AQI colours
- **Scatter** — every raw reading plotted over time, with an hourly-median line overlay per sensor

No API key or internet access required — works entirely from the local cache produced by the download scripts.

---

## Requirements

```
pip install pandas numpy
```

---

## Usage

```
python3 purpleair_aqi_charts.py --month YYYY-MM [options]
```

---

## Arguments

### `--month YYYY-MM` *(required)*

Month to process.

```bash
--month 2026-01
```

---

### `--data-dir PATH`

Directory containing the cached CSV files.

**Default:** `./purpleair_data`

```bash
--data-dir ./purpleair_data
--data-dir /home/user/air_quality/data
```

---

### `--data-prefix PREFIX`

Filename prefix used when locating the cache file. The script looks for:

```
{data-dir}/{data-prefix}_{month}_rawdata.csv
```

**Default:** `golden_town`

```bash
--data-prefix golden_town        # → purpleair_data/golden_town_2026-01_rawdata.csv
--data-prefix vancouver_region   # → purpleair_data/vancouver_region_2026-01_rawdata.csv
```

---

### `--exclude SENSOR_ID [SENSOR_ID ...]`

One or more sensor IDs to drop before processing. Useful for permanently broken or miscalibrated sensors. The excluded IDs are noted in the chart subtitles.

**Default:** none

```bash
--exclude 127469
--exclude 127469 99999 88888
```

---

### `--heatmap-out PATH`

Output path for the heatmap HTML file.

**Default:** `{data-prefix}_{month}_heatmap.html` in the current directory

```bash
--heatmap-out jan2026_heatmap.html
--heatmap-out /var/www/html/golden_heatmap.html
```

---

### `--scatter-out PATH`

Output path for the scatter HTML file.

**Default:** `{data-prefix}_{month}_scatter.html` in the current directory

```bash
--scatter-out jan2026_scatter.html
--scatter-out /var/www/html/golden_scatter.html
```

---

## Cache file format

The input CSV must have at minimum these three columns:

| Column | Type | Description |
|---|---|---|
| `sensor_index` | integer | PurpleAir sensor ID |
| `timestamp` | datetime string | Reading timestamp, any pandas-parseable format |
| `pm2.5` | float | PM2.5 concentration in µg/m³ |

This matches the `*_rawdata.csv` files produced by `purpleair_monthly_map.py` and `purpleair_hourly_grid.py`.

---

## Output files

Both outputs are fully self-contained HTML files — no server needed, open directly in any browser.

### Heatmap

- Grid of sensors (rows) × hours of day 0–23 (columns)
- Each cell shows the AQI value averaged across all days in the month for that sensor/hour combination
- Coloured with official EPA AQI palette: green → yellow → orange → red → purple
- Hover any cell for a tooltip with sensor, time, AQI value, and category name

### Scatter

- One dot per raw reading, coloured by sensor
- Solid line showing the hourly median per sensor
- AQI category bands shaded in the background
- Interactive: scroll or pinch to zoom on the time axis, drag to pan
- Click sensor names in the legend to show/hide individual sensors
- "Hide scatter" button to show only the median lines

---

## Examples

```bash
# Minimal — uses all defaults
python3 purpleair_aqi_charts.py --month 2026-01

# Typical Golden Town setup
python3 purpleair_aqi_charts.py \
    --month 2026-01 \
    --data-dir ./purpleair_data \
    --data-prefix golden_town \
    --exclude 127469

# Custom output locations
python3 purpleair_aqi_charts.py \
    --month 2026-01 \
    --data-prefix golden_town \
    --heatmap-out /var/www/html/jan_heatmap.html \
    --scatter-out /var/www/html/jan_scatter.html

# Multiple excluded sensors
python3 purpleair_aqi_charts.py \
    --month 2026-02 \
    --data-prefix golden_town \
    --exclude 127469 55555
```
