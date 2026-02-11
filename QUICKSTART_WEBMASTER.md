# Quick Start for Golden BC Webmaster
## Monthly AQI Map Generator

This tool creates a single static image showing the monthly average air quality for the Golden region, perfect for your website.

## 📥 Files You Need

Download these 4 files:

1. **[purpleair_monthly_map.py](file:///mnt/user-data/outputs/purpleair_monthly_map.py)** - Main script
2. **[generate_monthly_map.sh](file:///mnt/user-data/outputs/generate_monthly_map.sh)** - Easy-to-use wrapper
3. **[MONTHLY_MAP_README.md](file:///mnt/user-data/outputs/MONTHLY_MAP_README.md)** - Complete documentation
4. **[EXAMPLES.sh](file:///mnt/user-data/outputs/EXAMPLES.sh)** - Copy-paste commands

## 🚀 First Time Setup (5 minutes)

### 1. Install Python packages
```bash
pip3 install requests pandas numpy matplotlib Pillow
```

Or use the requirements file from earlier:
```bash
pip3 install -r requirements-core.txt
```

### 2. Get PurpleAir API Key
1. Visit https://develop.purpleair.com/
2. Sign up (free, uses Google account)
3. Create an API key (1 million free points)
4. Set it:
```bash
export PURPLEAIR_API_KEY="your-key-here"
```

### 3. Make scripts executable
```bash
chmod +x purpleair_monthly_map.py generate_monthly_map.sh
```

## 📍 Generate January 2026 Map

### Option 1: Interactive (Easiest)
```bash
bash generate_monthly_map.sh
```
Then answer the prompts:
- Month: `2026-01`
- Output: `jan2026_golden.png`
- Background: `golden_map.tif` (or press Enter to skip)

### Option 2: One Command
```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden.png \
    --background golden_map.tif
```

### Option 3: Without Background
```bash
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden.png
```

## 🗺️ Background Image (Optional but Recommended)

If you have a georeferenced map of Golden:
- Use `--background your_map.tif`
- Map should cover coordinates: (51.5°, -117.5°) to (51.0°, -116.25°)
- See BACKGROUND_IMAGE_GUIDE.md for creating one from CalTopo

## 📊 What You Get

### Image Output
- **File**: High-quality PNG (e.g., `jan2026_golden.png`)
- **Size**: ~200-500 KB (depending on DPI)
- **Content**: 
  - Color-coded sensors (green=good air, red=bad air)
  - AQI numbers on each sensor
  - Legend, statistics, title
  - Your background map (if provided)

### Data Files (in `./purpleair_data/`)
- `golden_sensors.csv` - Sensor locations (reuse for all months)
- `golden_2026-01_data.csv` - Raw hourly data (reuse for regenerating)
- `golden_2026-01_averages.csv` - Monthly averages (for spreadsheets)

## 🔄 Monthly Updates

For each new month, just run:
```bash
python3 purpleair_monthly_map.py \
    --month 2026-02 \
    --output feb2026_golden.png \
    --background golden_map.tif
```

## 💾 Reusing Data (Zero API Calls)

Once you've fetched data for a month, regenerate the image anytime with different settings:

```bash
# Change DPI
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_hires.png \
    --background golden_map.tif \
    --dpi 300 \
    --use-cached

# Change background
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_newbg.png \
    --background new_map.tif \
    --use-cached
```

No API points used! Instant regeneration.

## 🌐 Website Integration

### Simple HTML
```html
<div class="air-quality">
  <h2>January 2026 Air Quality</h2>
  <img src="jan2026_golden.png" 
       alt="Golden BC Monthly Air Quality" 
       style="max-width: 100%; height: auto;">
  <p>Monthly average AQI from PurpleAir sensors</p>
</div>
```

### Upload to Server
```bash
# Via SCP
scp jan2026_golden.png user@yoursite.com:/var/www/html/

# Via FTP
# Use your FTP client to upload jan2026_golden.png
```

## 🎨 Understanding the Colors

The script uses the same color scale as PurpleAir's website:

- 🟢 **Green (0-50)**: Good - Safe air quality
- 🟡 **Yellow (51-100)**: Moderate - Generally acceptable
- 🟠 **Orange (101-150)**: Unhealthy for sensitive groups
- 🔴 **Red (151-200)**: Unhealthy for everyone
- 🟣 **Purple (201-300)**: Very unhealthy
- 🔴 **Maroon (301+)**: Hazardous

Each sensor shows:
- **Circle color**: AQI category
- **Number**: Actual AQI value (0-500)

## 📐 Region Coverage

The Golden region is defined as:
- **Northwest corner**: 51.5°N, 117.5°W
- **Southeast corner**: 51.0°N, 116.25°W

This covers Golden and surrounding area. Sensors within this box are included.

## 💡 Tips

### For Website (Fast Loading)
```bash
--dpi 96
```

### For Print/Reports
```bash
--dpi 300
```

### Compare Multiple Months
```bash
# Generate last 3 months
for month in 2025-11 2025-12 2026-01; do
    python3 purpleair_monthly_map.py \
        --month $month \
        --output golden_${month}.png \
        --background golden_map.tif
done
```

## ❓ Troubleshooting

### "No sensors found"
- Check https://map.purpleair.com/ to verify sensors exist near Golden
- Sensors may be temporarily offline

### "No data for month"
- Can't fetch future months (only past/current)
- Some sensors may not have data for older months

### Background doesn't align
- Verify your image covers exact coordinates (51.5,-117.5) to (51.0,-116.25)
- Try without background first: `--output test.png` (no --background)

### Rate limiting
- Script includes 0.5s delays between requests
- If rate limited, wait a few minutes
- Use `--use-cached` to regenerate without API calls

## 📊 API Usage

For Golden region (~10 sensors):
- **First fetch**: ~11 API points per month
- **Regenerations**: 0 points (use `--use-cached`)
- **Your quota**: 1 million points = ~90,000 months of data

You'll never run out!

## 📞 Support

- **Full documentation**: MONTHLY_MAP_README.md
- **Example commands**: EXAMPLES.sh
- **PurpleAir API**: https://community.purpleair.com/
- **Background images**: BACKGROUND_IMAGE_GUIDE.md

## ⚡ Quick Reference

```bash
# Standard monthly map
python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png --background map.tif

# Web-optimized
python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png --dpi 96

# High-res print
python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png --dpi 300

# Regenerate (no API)
python3 purpleair_monthly_map.py --month 2026-01 --output jan2026.png --use-cached
```

That's it! You're ready to generate monthly AQI maps for your website. 🎉
