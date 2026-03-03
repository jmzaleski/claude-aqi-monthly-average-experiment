#!/bin/bash

#run it the first time with this null and after that don't need to hit the purpleair API
USE_CACHED="--use-cached"

MON_STR=feb
MONTH=2026-2
OUTPUT=$MON_STR"2026_monthly.png"

python3  purpleair_monthly_map.py $USE_CACHED --month $MONTH --output $OUTPUT --background purple-air-backgound-townsite.jpeg

read -p "hit return to open $OUTPUT > " JUNK
case $JUNK in
	n*)exit
	   ;;
esac
open $OUTPUT
