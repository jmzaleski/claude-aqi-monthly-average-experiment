#!/bin/bash

#run it the first time with this null and after that don't need to hit the purpleair API
USE_CACHED="--use-cached"

MON_STR=feb
MONTH=2026-2
HEATMAP_OUT=$MON_STR"2026_heatmap.html"
SCATTER_OUT=$MON_STR"2026_scatter.html"

set -x
python3  purpleair_aqi_charts.py \
		 --data-dir ./purpleair_data \
		 --month $MONTH \
		 --data-prefix golden_town \
		 --exclude 127469 \
		 --heatmap-out $HEATMAP_OUT \
		 --scatter-out $SCATTER_OUT


# --background purple-air-backgound-townsite.jpeg

read -p "hit return to open $HEATMAP_OUT $SCATTER_OUT > " JUNK
case $JUNK in
	n*)exit
	   ;;
esac
open $HEATMAP_OUT && open $SCATTER_OUT
