#!/bin/bash

#run it the first time with this null and after that don't need to hit the purpleair API
USE_CACHED="--use-cached"

MON_STR=feb
MONTH=2026-2
OUTPUT=$MON_STR"2026_hourly_patterns_avg.png"

python3 purpleair_hourly_grid.py $USE_CACHED --trim-outliers --use-cached --month $MONTH --output $OUTPUT  --background purple-air-backgound-townsite.jpeg

read -p "hit return to open $OUTPUT > " JUNK
case $JUNK in
	n*)exit
	   ;;
esac
open $OUTPUT

