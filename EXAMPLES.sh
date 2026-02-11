#!/bin/bash
# Example Commands for Golden BC Webmaster
# Copy and paste these to generate your monthly AQI maps

# =============================================================================
# SETUP (One Time)
# =============================================================================

# 1. Get your PurpleAir API key from https://develop.purpleair.com/
#    Then set it as an environment variable:
export PURPLEAIR_API_KEY="your-key-here"

# 2. Make scripts executable
chmod +x purpleair_monthly_map.py generate_monthly_map.sh

#

# =============================================================================
# GENERATE JANUARY 2026 MAP FOR GOLDEN
# =============================================================================

# Option 1: Interactive (easiest)
bash generate_monthly_map.sh
# Then answer the prompts:
#   Month: 2026-01
#   Output: jan2026_golden.png
#   Background: golden_map.tif  (or press Enter to skip)

# Option 2: Command line (fastest)
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden.png \
    --background golden_map.tif


# =============================================================================
# GENERATE WITHOUT BACKGROUND IMAGE
# =============================================================================

python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden_no_bg.png


# =============================================================================
# GENERATE FOR DIFFERENT MONTHS
# =============================================================================

# December 2025
python3 purpleair_monthly_map.py \
    --month 2025-12 \
    --output dec2025_golden.png \
    --background golden_map.tif

# February 2026
python3 purpleair_monthly_map.py \
    --month 2026-02 \
    --output feb2026_golden.png \
    --background golden_map.tif


# =============================================================================
# REUSE CACHED DATA (No API calls, instant)
# =============================================================================

# Once you've fetched data for a month, you can regenerate the image
# with different settings without using any API points:

python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden_hires.png \
    --background golden_map.tif \
    --dpi 300 \
    --use-cached


# =============================================================================
# HIGH RESOLUTION FOR PRINT
# =============================================================================

python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden_print.png \
    --background golden_map.tif \
    --dpi 300


# =============================================================================
# WEB-OPTIMIZED VERSION
# =============================================================================

python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_golden_web.png \
    --background golden_map.tif \
    --dpi 96


# =============================================================================
# BATCH GENERATE MULTIPLE MONTHS
# =============================================================================

# Generate last 3 months
for month in 2025-11 2025-12 2026-01; do
    echo "Generating $month..."
    python3 purpleair_monthly_map.py \
        --month $month \
        --output golden_${month}.png \
        --background golden_map.tif
done


# =============================================================================
# AUTOMATED MONTHLY UPDATE
# =============================================================================

# Generate current month (for cron job)
CURRENT_MONTH=$(date +%Y-%m)
python3 purpleair_monthly_map.py \
    --month $CURRENT_MONTH \
    --output website/current_aqi.png \
    --background golden_map.tif


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If you get "No sensors found":
# 1. Visit https://map.purpleair.com/
# 2. Check that sensors exist near Golden, BC
# 3. The bounding box is: (51.5, -117.5) to (51.0, -116.25)

# If you get API errors:
# 1. Check your API key: echo $PURPLEAIR_API_KEY
# 2. Wait a few minutes if you hit rate limits
# 3. Use --use-cached to regenerate without API calls

# If background doesn't align:
# 1. Make sure your image covers coordinates (51.5,-117.5) to (51.0,-116.25)
# 2. Try without background first to verify sensor positions
# 3. See BACKGROUND_IMAGE_GUIDE.md for creating aligned maps


# =============================================================================
# VIEW OUTPUT
# =============================================================================

# macOS
open jan2026_golden.png

# Linux
xdg-open jan2026_golden.png

# Or just open in your web browser or image viewer


# =============================================================================
# WHAT YOU'LL GET
# =============================================================================

# 1. PNG image with:
#    - Color-coded sensors (green=good, red=bad)
#    - AQI numbers on each sensor
#    - Legend showing AQI categories
#    - Statistics (average, min, max)
#    - Your background map (if provided)

# 2. CSV files in ./purpleair_data/:
#    - golden_sensors.csv (sensor locations - reusable)
#    - golden_2026-01_data.csv (hourly data - reusable)
#    - golden_2026-01_averages.csv (monthly averages - for spreadsheets)


# =============================================================================
# INTEGRATION WITH YOUR WEBSITE
# =============================================================================

# 1. Generate the map
python3 purpleair_monthly_map.py \
    --month 2026-01 \
    --output jan2026_aqi.png \
    --background golden_map.tif \
    --dpi 96

# 2. Upload to your web server
scp jan2026_aqi.png user@yoursite.com:/var/www/html/images/

# 3. Add to your webpage
cat > aqi.html << 'EOF'
<div class="aqi-map">
  <h2>January 2026 Air Quality</h2>
  <img src="images/jan2026_aqi.png" 
       alt="Golden BC Air Quality - January 2026"
       style="max-width: 100%; height: auto;">
  <p>Monthly average AQI from PurpleAir sensors in the Golden region.</p>
</div>
EOF


# =============================================================================
# API USAGE
# =============================================================================

# For Golden region (approximately 10 sensors):
# - Fetch 1 month of data: ~11 API points
# - With 1 million free points: ~90,000 months of data!
# - Using --use-cached: 0 points (unlimited regenerations)
