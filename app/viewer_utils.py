"""
3-D molecular viewer HTML builders for the RitS Streamlit app.

Pure functions — no Streamlit imports.  GIF-specific snippets are
composed in from gif_utils so the two concerns stay cleanly separated.
"""

import json

import numpy as np
import py3Dmol

from gif_utils import gif_controls_html, gif_css, gif_download_js, gif_script_tag


# ---------------------------------------------------------------------------
# MOL-block builder
# ---------------------------------------------------------------------------

def build_mol_block(symbols, coords, bonds: list) -> str:
    """Build a V2000 MOL block from atom symbols, coordinates and bond list."""
    n_atoms = len(symbols)
    n_bonds = len(bonds)
    lines = [
        "",          # mol name (blank)
        "     RitS",
        "",
        f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for sym, c in zip(symbols, coords):
        lines.append(
            f"{c[0]:10.4f}{c[1]:10.4f}{c[2]:10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for i, j in bonds:
        lines.append(f"{i+1:3d}{j+1:3d}  1  0  0  0  0")
    lines.append("M  END")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-structure viewer
# ---------------------------------------------------------------------------

def render_ts_3d(symbols, coords, bonds: list, height: int = 450, width: int = 600) -> str:
    """Return a py3Dmol HTML snippet for a single TS structure."""
    mol_block = build_mol_block(symbols, coords, bonds)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer._make_html()


# ---------------------------------------------------------------------------
# IRC trajectory viewer (with GIF download)
# ---------------------------------------------------------------------------

def build_trajectory_html(mol_blocks: list, width: int = 900, height: int = 500) -> str:
    """Build a self-contained HTML page with all IRC frames, a JS slider,
    and a client-side GIF download button (from gif_utils).

    All frame data lives in the browser — scrubbing is instant with no
    server round-trips.
    """
    n   = len(mol_blocks)
    mid = n // 2
    frames_json = json.dumps(mol_blocks)

    return f"""\
<!DOCTYPE html>
<html>
<head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
{gif_script_tag()}
<style>
  body {{ margin:0; padding:0; font-family: sans-serif; }}
  #viewer {{ width:{width}px; height:{height}px; position:relative; }}
  .controls {{
    padding:6px 0; display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  }}
  .controls button {{
    padding:6px 16px; border:1px solid #ccc; border-radius:6px;
    background:#f8f8f8; cursor:pointer; font-size:14px;
  }}
  .controls button:hover {{ background:#e8e8e8; }}
  .controls button:disabled {{ opacity:0.5; cursor:not-allowed; }}
  #slider {{
    flex:1; min-width:200px; height:6px;
    -webkit-appearance:none; appearance:none;
    background:#ddd; border-radius:3px; outline:none;
  }}
  #slider::-webkit-slider-thumb {{
    -webkit-appearance:none; appearance:none;
    width:18px; height:18px; border-radius:50%;
    background:#4a90d9; cursor:pointer;
  }}
  #slider::-moz-range-thumb {{
    width:18px; height:18px; border-radius:50%;
    background:#4a90d9; cursor:pointer; border:none;
  }}
  #frame-label {{ font-size:14px; color:#555; min-width:100px; }}
  #play-btn {{ font-size:18px; width:40px; text-align:center; }}
{gif_css()}
</style>
</head>
<body>
<div id="viewer"></div>
<div class="controls">
  <button onclick="goTo(0)">Reactant</button>
  <button onclick="goTo({mid})">TS</button>
  <button onclick="goTo({n-1})">Product</button>
  <button id="play-btn" onclick="togglePlay()">&#9654;</button>
  <input id="slider" type="range" min="0" max="{n-1}" value="{mid}">
  <span id="frame-label">Frame {mid}/{n-1}</span>
</div>
{gif_controls_html()}
<script>
var frames = {frames_json};
var viewer = $3Dmol.createViewer("viewer", {{
  backgroundColor:      "white",
  preserveDrawingBuffer: true
}});
var current   = {mid};
var playing   = false;
var playTimer = null;

function showFrame(i) {{
  current = i;
  viewer.removeAllModels();
  viewer.addModel(frames[i], "mol");
  viewer.setStyle({{}}, {{stick:{{radius:0.15}}, sphere:{{scale:0.25}}}});
  viewer.render();
  document.getElementById("slider").value = i;
  document.getElementById("frame-label").textContent = "Frame " + i + "/{n-1}";
}}

function goTo(i) {{ showFrame(i); }}

document.getElementById("slider").addEventListener("input", function() {{
  showFrame(parseInt(this.value));
}});

function togglePlay() {{
  playing = !playing;
  document.getElementById("play-btn").innerHTML = playing ? "&#9724;" : "&#9654;";
  if (playing) {{
    playTimer = setInterval(function() {{
      var next = (current + 1) % {n};
      showFrame(next);
    }}, 80);
  }} else {{
    clearInterval(playTimer);
  }}
}}

{gif_download_js(n, n - 1)}

showFrame({mid});
viewer.zoomTo();
</script>
</body>
</html>"""
