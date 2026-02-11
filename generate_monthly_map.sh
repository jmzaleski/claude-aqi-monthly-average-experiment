#!/bin/bash
# Convenience script for generating PurpleAir monthly maps
# Makes it easy to generate maps without remembering all the options

set -e

# Configuration
REGION="golden"
BACKGROUND_IMAGE="purple-air-backgound-caltopo.jpeg"  # Set to your background image path, or leave empty
DATA_DIR="./purpleair_data"
DPI=150

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "PurpleAir Monthly Map Generator"
echo "======================================================================"
echo ""

# Check for API key
if [ -z "$PURPLEAIR_API_KEY" ]; then
    echo -e "${YELLOW}Warning: PURPLEAIR_API_KEY not set${NC}"
    echo "Set it with: export PURPLEAIR_API_KEY='your-key-here'"
    echo ""
    read -p "Continue anyway (use cached data)? (y/n) [n]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
    USE_CACHED="--use-cached"
else
    echo -e "${GREEN}✓ API key found${NC}"
    USE_CACHED=""
fi

echo ""

# Get month
read -p "Enter month (YYYY-MM) [$(date +%Y-%m)]: " MONTH
MONTH=${MONTH:-$(date +%Y-%m)}

# Validate month format
if [[ ! "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
    echo -e "${RED}Error: Invalid month format. Use YYYY-MM (e.g., 2026-01)${NC}"
    exit 1
fi

# Get output filename
DEFAULT_OUTPUT="${REGION}_${MONTH}_aqi.png"
read -p "Output filename [$DEFAULT_OUTPUT]: " OUTPUT
OUTPUT=${OUTPUT:-$DEFAULT_OUTPUT}

# Ask about background image
if [ -z "$BACKGROUND_IMAGE" ]; then
    read -p "Background image path (or press Enter to skip): " BG_INPUT
    if [ -n "$BG_INPUT" ]; then
        BACKGROUND_IMAGE="$BG_INPUT"
    fi
fi

# Build command
CMD="python3 purpleair_monthly_map.py"
CMD="$CMD --month $MONTH"
CMD="$CMD --output $OUTPUT"
CMD="$CMD --region $REGION"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --dpi $DPI"

if [ -n "$BACKGROUND_IMAGE" ]; then
    CMD="$CMD --background $BACKGROUND_IMAGE"
fi

if [ -n "$USE_CACHED" ]; then
    CMD="$CMD $USE_CACHED"
fi

echo ""
echo "======================================================================"
echo "Configuration:"
echo "======================================================================"
echo "  Region: $REGION"
echo "  Month: $MONTH"
echo "  Output: $OUTPUT"
echo "  Background: ${BACKGROUND_IMAGE:-none}"
echo "  DPI: $DPI"
echo "  Data directory: $DATA_DIR"
if [ -n "$USE_CACHED" ]; then
    echo "  Mode: Using cached data"
fi
echo ""
echo "Command:"
echo "  $CMD"
echo ""

read -p "Proceed? (y/n) [y]: " PROCEED
PROCEED=${PROCEED:-y}

if [[ ! "$PROCEED" =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "======================================================================"
echo "Running..."
echo "======================================================================"
echo ""

# Run the command
$CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}✓ Success!${NC}"
    echo "======================================================================"
    echo "  Output: $OUTPUT"
    
    # Show file size
    if [ -f "$OUTPUT" ]; then
        SIZE=$(du -h "$OUTPUT" | cut -f1)
        echo "  Size: $SIZE"
    fi
    
    # Try to open it (macOS)
    if command -v open &> /dev/null; then
        read -p "Open image now? (y/n) [y]: " OPEN
        OPEN=${OPEN:-y}
        if [[ "$OPEN" =~ ^[Yy]$ ]]; then
            open "$OUTPUT"
        fi
    fi
else
    echo ""
    echo -e "${RED}Error: Command failed${NC}"
    exit 1
fi
