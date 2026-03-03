#!/bin/bash

#run it the first time with this null and after that don't need to hit the purpleair API
USE_CACHED="--use-cached"

MON_STR=feb
MONTH=2026-2
OUTPUT=$MON_STR"2026_charts.png"

set -x
python3  purpleair_aqi_charts.py \
		 --data-dir ./purpleair_data \
		 --month $MONTH \
		 --data-prefix golden_town \
		 --exclude 127469 \
		 --heatmap-out ./$MON_STR-heatmap.html \
		 --scatter-out ./$MON_STR-scatter.html


# --background purple-air-backgound-townsite.jpeg

read -p "hit return to open $OUTPUT > " JUNK
case $JUNK in
	n*)exit
	   ;;
esac
open $OUTPUT
