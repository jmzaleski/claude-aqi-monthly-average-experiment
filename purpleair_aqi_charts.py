#!/usr/bin/env python3
"""
purpleair_aqi_charts.py
=======================
Reads PurpleAir cached raw data and generates two interactive HTML charts:

  1. AQI Heatmap  — sensors × hour-of-day, coloured with official US EPA AQI colours
  2. AQI Scatter  — every raw reading over time, with hourly-median line overlay

Usage
-----
  python3 purpleair_aqi_charts.py --month 2026-01

  # Custom data dir / prefix (matches your existing cached files)
  python3 purpleair_aqi_charts.py --month 2026-01 \
      --data-dir ./purpleair_data \
      --data-prefix golden_town

  # Exclude broken sensors
  python3 purpleair_aqi_charts.py --month 2026-01 \
      --exclude 127469 99999

  # Custom output paths
  python3 purpleair_aqi_charts.py --month 2026-01 \
      --heatmap-out golden_heatmap.html \
      --scatter-out golden_scatter.html

Cache file expected
-------------------
  {data_dir}/{data_prefix}_{YYYY-MM}_rawdata.csv

  Required columns: sensor_index, timestamp, pm2.5
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# US EPA PM2.5 → AQI conversion
# ---------------------------------------------------------------------------
PM25_BREAKPOINTS = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

AQI_CATEGORIES = [
    {"lo": 0,   "hi": 50,  "label": "Good",                         "bg": "#00e400", "fg": "#1a1a1a"},
    {"lo": 51,  "hi": 100, "label": "Moderate",                     "bg": "#ffff00", "fg": "#1a1a1a"},
    {"lo": 101, "hi": 150, "label": "Unhealthy for Sensitive Groups","bg": "#ff7e00", "fg": "#ffffff"},
    {"lo": 151, "hi": 200, "label": "Unhealthy",                    "bg": "#ff0000", "fg": "#ffffff"},
    {"lo": 201, "hi": 300, "label": "Very Unhealthy",               "bg": "#8f3f97", "fg": "#ffffff"},
    {"lo": 301, "hi": 500, "label": "Hazardous",                    "bg": "#7e0023", "fg": "#ffffff"},
]


def pm25_to_aqi(pm: float) -> int | None:
    if pm is None or (isinstance(pm, float) and np.isnan(pm)):
        return None
    pm = round(float(pm), 1)
    for bp_lo, bp_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if bp_lo <= pm <= bp_hi:
            return round((aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (pm - bp_lo) + aqi_lo)
    return 500 if pm > 500.4 else 0


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------
SCATTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#e91e63",
]


def build_heatmap_html(df: pd.DataFrame, month_str: str, excluded: list[int]) -> str:
    """Build the sensor × hour AQI heatmap HTML."""

    sensors = sorted(df["sensor_index"].unique())
    color_map = {sid: SCATTER_COLORS[i % len(SCATTER_COLORS)]
                 for i, sid in enumerate(sensors)}

    # Compute hourly averages per sensor, convert to AQI
    heatmap = {}
    for sid in sensors:
        heatmap[str(sid)] = {}
        sdf = df[df["sensor_index"] == sid]
        for h in range(24):
            vals = sdf[sdf["hour"] == h]["pm2.5"].dropna()
            avg_pm = float(vals.mean()) if len(vals) > 0 else None
            heatmap[str(sid)][h] = pm25_to_aqi(avg_pm)

    sensor_strs = [str(s) for s in sensors]
    data_js = json.dumps({
        "sensors": sensor_strs,
        "data": heatmap,
        "categories": AQI_CATEGORIES,
        "sensorColors": {str(s): color_map[s] for s in sensors},
    })

    excluded_note = ""
    if excluded:
        excluded_note = f" &nbsp;|&nbsp; Excluded (faulty): {', '.join(str(e) for e in excluded)}"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AQI Heatmap — {month_str}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:"Segoe UI",Arial,sans-serif; background:#eef2f7; padding:24px; }}
  h1 {{ text-align:center; color:#2c3e50; font-size:20px; margin-bottom:5px; }}
  .subtitle {{ text-align:center; color:#7f8c8d; font-size:12px; margin-bottom:24px; }}
  .card {{ background:#fff; border-radius:14px; padding:22px 28px 18px;
           box-shadow:0 2px 14px rgba(0,0,0,0.08); max-width:1000px; margin:0 auto 20px; }}
  .heatmap-wrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; min-width:700px; }}
  th {{ font-size:10px; color:#aaa; font-weight:600; padding:0 0 8px; text-align:center; }}
  th.row-label {{ text-align:right; padding-right:14px; min-width:88px;
                  color:#555; font-size:12px; }}
  td {{ width:36px; height:40px; border-radius:4px; text-align:center;
        vertical-align:middle; font-size:10.5px; font-weight:700; cursor:default;
        transition:transform .1s,box-shadow .1s; }}
  td:hover {{ transform:scale(1.2); box-shadow:0 2px 10px rgba(0,0,0,.25);
              z-index:10; position:relative; }}
  td.sensor-label {{ text-align:right; padding-right:14px; font-size:12px;
                     font-weight:600; color:#34495e; background:none !important;
                     transform:none !important; box-shadow:none !important;
                     width:auto; white-space:nowrap; }}
  .legend-grid {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .legend-item {{ display:flex; align-items:center; gap:8px; background:#f8f9fa;
                  border-radius:8px; padding:7px 12px; flex:1; min-width:155px; }}
  .swatch {{ width:28px; height:28px; border-radius:6px; flex-shrink:0;
             display:flex; align-items:center; justify-content:center;
             font-size:9px; font-weight:700; }}
  .legend-text .range {{ font-size:12px; font-weight:700; color:#2c3e50; }}
  .legend-text .lbl {{ font-size:11px; color:#888; }}
  .legend-title {{ font-size:13px; font-weight:600; color:#555; margin-bottom:10px; }}
  .tooltip-box {{ position:fixed; background:#2c3e50; color:#fff; padding:8px 13px;
                  border-radius:8px; font-size:12px; pointer-events:none; opacity:0;
                  transition:opacity .15s; z-index:1000; white-space:nowrap;
                  box-shadow:0 4px 12px rgba(0,0,0,.25); }}
</style>
</head>
<body>
<h1>🌫️ AQI by Sensor &amp; Hour of Day</h1>
<p class="subtitle">Average PM2.5 converted to US AQI &nbsp;|&nbsp; {month_str}{excluded_note}</p>
<div class="card">
  <div class="heatmap-wrap"><table id="heatmap"></table></div>
</div>
<div class="card">
  <div class="legend-title">US AQI Categories (PM2.5)</div>
  <div class="legend-grid" id="legend"></div>
</div>
<div class="tooltip-box" id="tooltip"></div>
<script>
const D = {data_js};
const hourLabels = ['12a','1a','2a','3a','4a','5a','6a','7a','8a','9a','10a','11a',
                    '12p','1p','2p','3p','4p','5p','6p','7p','8p','9p','10p','11p'];
const periods = [{{label:'Night',span:6}},{{label:'Morning',span:6}},
                 {{label:'Afternoon',span:6}},{{label:'Evening',span:6}}];

function aqiCat(v) {{
  for (const c of D.categories) if (v <= c.hi) return c;
  return D.categories[D.categories.length-1];
}}

const tbl = document.getElementById('heatmap');
let html = '<thead><tr><th class="row-label"></th>';
periods.forEach(p => {{
  html += `<th colspan="${{p.span}}" style="border-bottom:2px solid #ecf0f1;color:#bbb;`+
          `font-size:10px;letter-spacing:.5px;text-transform:uppercase;padding-bottom:6px">`+
          `${{p.label}}</th>`;
}});
html += '</tr><tr><th class="row-label">Sensor</th>';
for (let h=0;h<24;h++) html += `<th style="font-size:10px;color:#ccc;font-weight:500;min-width:33px">${{hourLabels[h]}}</th>`;
html += '</tr></thead><tbody>';
D.sensors.forEach(sid => {{
  html += `<tr><td class="sensor-label">${{sid}}</td>`;
  for (let h=0;h<24;h++) {{
    const v = D.data[sid][h];
    if (v === null) {{ html += '<td style="background:#f0f0f0;color:#ccc">—</td>'; continue; }}
    const c = aqiCat(v);
    html += `<td style="background:${{c.bg}};color:${{c.fg}}" `+
            `data-sensor="${{sid}}" data-hour="${{h}}" data-aqi="${{v}}" data-cat="${{c.label}}">${{v}}</td>`;
  }}
  html += '</tr>';
}});
html += '</tbody>';
tbl.innerHTML = html;

// Legend
const legEl = document.getElementById('legend');
D.categories.forEach(c => {{
  legEl.innerHTML += `<div class="legend-item">
    <div class="swatch" style="background:${{c.bg}};color:${{c.fg}}">${{c.lo}}</div>
    <div class="legend-text">
      <div class="range">${{c.lo}}–${{c.hi}}</div>
      <div class="lbl">${{c.label}}</div>
    </div></div>`;
}});

// Tooltip
const tip = document.getElementById('tooltip');
tbl.addEventListener('mouseover', e => {{
  const td = e.target.closest('td[data-sensor]'); if (!td) {{ tip.style.opacity=0; return; }}
  const h = parseInt(td.dataset.hour);
  const period = h<6?'Night':h<12?'Morning':h<18?'Afternoon':'Evening';
  tip.textContent = `Sensor ${{td.dataset.sensor}} · ${{hourLabels[h]}} (${{period}}) · AQI ${{td.dataset.aqi}} — ${{td.dataset.cat}}`;
  tip.style.opacity = 1;
}});
tbl.addEventListener('mousemove', e => {{
  tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-36)+'px';
}});
tbl.addEventListener('mouseleave', () => tip.style.opacity=0);
</script>
</body>
</html>"""
    return html


def build_scatter_html(df: pd.DataFrame, month_str: str, excluded: list[int]) -> str:
    """Build the raw-scatter + hourly-median line chart HTML."""

    sensors = sorted(df["sensor_index"].unique())
    color_map = {sid: SCATTER_COLORS[i % len(SCATTER_COLORS)]
                 for i, sid in enumerate(sensors)}

    scatter_data = {}
    line_data = {}

    for sid in sensors:
        sdf = df[df["sensor_index"] == sid].copy()
        if len(sdf) > 800:
            sdf_sample = sdf.sample(800, random_state=42).sort_values("timestamp")
        else:
            sdf_sample = sdf.sort_values("timestamp")

        scatter_data[str(sid)] = {
            "x": sdf_sample["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "y": sdf_sample["aqi"].tolist(),
        }

        sdf2 = sdf.copy()
        sdf2["hour_bucket"] = sdf2["timestamp"].dt.floor("1h")
        grp = (sdf2.groupby("hour_bucket")["aqi"]
               .agg(median="median",
                    p25=lambda x: x.quantile(0.25),
                    p75=lambda x: x.quantile(0.75))
               .reset_index()
               .sort_values("hour_bucket"))

        line_data[str(sid)] = {
            "x":      grp["hour_bucket"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "median": grp["median"].round(1).tolist(),
            "p25":    grp["p25"].round(1).tolist(),
            "p75":    grp["p75"].round(1).tolist(),
        }

    sensor_strs = [str(s) for s in sensors]
    data_js = json.dumps({
        "sensors": sensor_strs,
        "scatter": scatter_data,
        "lines": line_data,
        "colors": {str(s): color_map[s] for s in sensors},
    })

    excluded_note = ""
    if excluded:
        excluded_note = f" | Excluded: {', '.join(str(e) for e in excluded)}"

    # JS block written as a raw string to avoid f-string / brace conflicts
    js_block = r"""
const datasets = [];
D.sensors.forEach(sid => {
  const c = D.colors[sid];
  datasets.push({
    type:"scatter", label:sid+"_s",
    data: D.scatter[sid].x.map((x,i) => ({x, y:D.scatter[sid].y[i]})),
    backgroundColor:c+"2e", pointRadius:2.5, pointHoverRadius:5,
    _sensor:sid, _kind:"scatter"
  });
  datasets.push({
    type:"line", label:sid,
    data: D.lines[sid].x.map((x,i) => ({x, y:D.lines[sid].median[i]})),
    borderColor:c, borderWidth:2.5, pointRadius:0, tension:0.3, fill:false,
    _sensor:sid, _kind:"line"
  });
});

const bandsPlugin = {
  id:"bands",
  beforeDraw(chart) {
    const {ctx,chartArea:{top,bottom,left,right},scales:{y}} = chart;
    if (!y) return;
    [{y0:0,y1:50,col:"rgba(0,228,0,0.07)"},
     {y0:51,y1:100,col:"rgba(255,255,0,0.07)"},
     {y0:101,y1:150,col:"rgba(255,126,0,0.09)"},
     {y0:151,y1:200,col:"rgba(255,0,0,0.09)"},
     {y0:201,y1:300,col:"rgba(143,63,151,0.09)"}
    ].forEach(b => {
      const yT=Math.max(y.getPixelForValue(b.y1),top),
            yB=Math.min(y.getPixelForValue(b.y0),bottom);
      if (yB<=yT) return;
      ctx.save(); ctx.fillStyle=b.col; ctx.fillRect(left,yT,right-left,yB-yT); ctx.restore();
    });
  }
};

const chart = new Chart(document.getElementById("chart").getContext("2d"), {
  data: {datasets},
  options: {
    animation: false, responsive: true,
    interaction: {mode:"nearest",intersect:false,axis:"x"},
    scales: {
      x: {
        type:"time",
        time:{unit:"day",displayFormats:{day:"MMM d"},tooltipFormat:"MMM d HH:mm"},
        ticks:{color:"#888",maxTicksLimit:16}, grid:{color:"#2e3340"}
      },
      y: {
        min:0, max:200,
        title:{display:true,text:"AQI",color:"#888"},
        ticks:{color:"#888"}, grid:{color:"#2e3340"}
      }
    },
    plugins: {
      legend: {display:false},
      tooltip: {callbacks: {
        label: i => "Sensor "+i.dataset._sensor+" ("+(i.dataset._kind==="scatter"?"raw":"median")+"): AQI "+Math.round(i.parsed.y)
      }},
      zoom: {
        zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:"x"},
        pan:{enabled:true,mode:"x"}
      }
    }
  },
  plugins:[bandsPlugin]
});

const legEl = document.getElementById("legend");
const hidden = new Set();
D.sensors.forEach(sid => {
  const el = document.createElement("div");
  el.className = "legend-item";
  el.innerHTML = "<div class='legend-dot' style='background:"+D.colors[sid]+"'></div>"+sid;
  el.onclick = () => {
    hidden.has(sid) ? hidden.delete(sid) : hidden.add(sid);
    el.classList.toggle("hidden");
    chart.data.datasets.forEach(d => { if (d._sensor===sid) d.hidden=hidden.has(sid); });
    chart.update();
  };
  legEl.appendChild(el);
});

document.getElementById("btnZoom").onclick = () => chart.resetZoom();
let sv = true;
document.getElementById("btnScatter").onclick = function() {
  sv = !sv;
  chart.data.datasets.forEach(d => { if (d._kind==="scatter") d.hidden=!sv; });
  this.textContent = sv ? "Hide scatter" : "Show scatter";
  chart.update();
};
"""

    css = """* {box-sizing:border-box;margin:0;padding:0;}
body {font-family:"Segoe UI",Arial,sans-serif;background:#1a1d23;color:#e0e0e0;padding:24px;}
h1 {text-align:center;font-size:19px;color:#f0f0f0;margin-bottom:5px;}
.subtitle {text-align:center;color:#888;font-size:12px;margin-bottom:18px;}
.card {background:#22262e;border-radius:14px;padding:20px 24px;box-shadow:0 4px 20px rgba(0,0,0,.4);max-width:1200px;margin:0 auto 16px;}
canvas {width:100%!important;}
.controls {display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:16px;}
.legend {display:flex;flex-wrap:wrap;gap:8px;}
.legend-item {display:flex;align-items:center;gap:6px;cursor:pointer;padding:4px 10px;border-radius:20px;background:#2c3140;font-size:12px;user-select:none;transition:opacity .2s;}
.legend-item.hidden {opacity:.35;}
.legend-dot {width:12px;height:12px;border-radius:50%;flex-shrink:0;}
.btn {background:#3a3f4e;border:none;color:#ccc;padding:6px 14px;border-radius:8px;cursor:pointer;font-size:12px;}
.btn:hover {background:#4a5060;}
.aqi-bands {font-size:11px;color:#666;margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;justify-content:center;}
.band {display:flex;align-items:center;gap:4px;}
.band-sw {width:24px;height:10px;border-radius:2px;}"""

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>AQI Scatter \u2014 {month_str}</title>\n"
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>\n'
        "<style>\n" + css + "\n</style>\n</head>\n<body>\n"
        f"<h1>\U0001f32b\ufe0f AQI Over Time \u2014 {month_str}</h1>\n"
        f'<p class="subtitle">Faint dots = raw readings &nbsp;|&nbsp; Solid line = hourly median &nbsp;|&nbsp; '
        f'Scroll/pinch to zoom &nbsp;|&nbsp; Drag to pan &nbsp;|&nbsp; '
        f"Click legend to toggle sensors{excluded_note}</p>\n"
        '<div class="card">\n'
        '  <div class="controls">\n'
        '    <div class="legend" id="legend"></div>\n'
        '    <button class="btn" id="btnZoom">Reset zoom</button>\n'
        '    <button class="btn" id="btnScatter">Hide scatter</button>\n'
        '  </div>\n'
        '  <canvas id="chart" height="420"></canvas>\n'
        '  <div class="aqi-bands">\n'
        '    <div class="band"><div class="band-sw" style="background:#00e400"></div>Good (0\u201350)</div>\n'
        '    <div class="band"><div class="band-sw" style="background:#ffff00"></div>Moderate (51\u2013100)</div>\n'
        '    <div class="band"><div class="band-sw" style="background:#ff7e00"></div>Sensitive (101\u2013150)</div>\n'
        '    <div class="band"><div class="band-sw" style="background:#ff0000"></div>Unhealthy (151\u2013200)</div>\n'
        '    <div class="band"><div class="band-sw" style="background:#8f3f97"></div>Very Unhealthy (201+)</div>\n'
        '  </div>\n</div>\n'
        "<script>\nconst D = "
        + data_js
        + ";\n"
        + js_block
        + "\n</script>\n</body>\n</html>"
    )
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate AQI heatmap and scatter HTML from PurpleAir cached data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic — reads ./purpleair_data/golden_town_2026-01_rawdata.csv
  python3 purpleair_aqi_charts.py --month 2026-01

  # Explicit paths
  python3 purpleair_aqi_charts.py --month 2026-01 \\
      --data-dir ./purpleair_data \\
      --data-prefix golden_town

  # Exclude faulty sensors
  python3 purpleair_aqi_charts.py --month 2026-01 --exclude 127469

  # Custom output filenames
  python3 purpleair_aqi_charts.py --month 2026-01 \\
      --heatmap-out jan_heatmap.html \\
      --scatter-out jan_scatter.html
        """,
    )
    parser.add_argument("--month", required=True,
                        help="Month to process, format YYYY-MM  e.g. 2026-01")
    parser.add_argument("--data-dir", default="./purpleair_data",
                        help="Directory containing cached CSV files (default: ./purpleair_data)")
    parser.add_argument("--data-prefix", default="golden_town",
                        help="Filename prefix matching your cache files (default: golden_town)")
    parser.add_argument("--exclude", nargs="*", type=int, default=[],
                        metavar="SENSOR_ID",
                        help="Sensor IDs to exclude (e.g. --exclude 127469)")
    parser.add_argument("--heatmap-out", default=None,
                        help="Output path for heatmap HTML (default: {prefix}_{month}_heatmap.html)")
    parser.add_argument("--scatter-out", default=None,
                        help="Output path for scatter HTML (default: {prefix}_{month}_scatter.html)")
    args = parser.parse_args()

    # --- validate month ---
    try:
        year, month = map(int, args.month.split("-"))
        month_str = datetime(year, month, 1).strftime("%B %Y")
    except ValueError:
        print(f"Error: invalid month '{args.month}'. Use YYYY-MM.", file=sys.stderr)
        sys.exit(1)

    # --- locate cache file ---
    rawdata_csv = os.path.join(
        args.data_dir, f"{args.data_prefix}_{args.month}_rawdata.csv"
    )
    if not os.path.exists(rawdata_csv):
        print(f"Error: cache file not found:\n  {rawdata_csv}", file=sys.stderr)
        print("Check --data-dir and --data-prefix match your file layout.", file=sys.stderr)
        sys.exit(1)

    # --- default output filenames ---
    heatmap_out = args.heatmap_out or f"{args.data_prefix}_{args.month}_heatmap.html"
    scatter_out = args.scatter_out or f"{args.data_prefix}_{args.month}_scatter.html"

    print("=" * 60)
    print("PurpleAir AQI Charts Generator")
    print("=" * 60)
    print(f"  Month       : {month_str}")
    print(f"  Source CSV  : {rawdata_csv}")
    if args.exclude:
        print(f"  Excluded    : {args.exclude}")
    print(f"  Heatmap out : {heatmap_out}")
    print(f"  Scatter out : {scatter_out}")
    print()

    # --- load data ---
    print("Loading data...", end=" ", flush=True)
    df = pd.read_csv(rawdata_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    print(f"{len(df):,} rows, {df['sensor_index'].nunique()} sensors")

    # --- exclude bad sensors ---
    if args.exclude:
        before = df["sensor_index"].nunique()
        df = df[~df["sensor_index"].isin(args.exclude)]
        after = df["sensor_index"].nunique()
        print(f"  Excluded {before - after} sensor(s) → {after} remaining")

    # --- convert PM2.5 → AQI ---
    print("Converting PM2.5 → AQI...", end=" ", flush=True)
    df["aqi"] = df["pm2.5"].apply(pm25_to_aqi)
    df = df.dropna(subset=["aqi"])
    df["aqi"] = df["aqi"].astype(int)
    print("done")

    # --- build and write heatmap ---
    print("Building heatmap...", end=" ", flush=True)
    heatmap_html = build_heatmap_html(df, month_str, args.exclude)
    with open(heatmap_out, "w", encoding="utf-8") as f:
        f.write(heatmap_html)
    print(f"written → {heatmap_out}")

    # --- build and write scatter ---
    print("Building scatter...", end=" ", flush=True)
    scatter_html = build_scatter_html(df, month_str, args.exclude)
    with open(scatter_out, "w", encoding="utf-8") as f:
        f.write(scatter_html)
    print(f"written → {scatter_out}")

    print()
    print("Done!")
    print(f"  Open {heatmap_out} in your browser for the heatmap.")
    print(f"  Open {scatter_out} in your browser for the scatter.")


if __name__ == "__main__":
    main()
