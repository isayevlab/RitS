import json
import random
from pathlib import Path

import py3Dmol
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    EXAMPLE_REACTIONS,
    IRC_THRESH_OPTIONS,
    IRC_TYPE_OPTIONS,
    TSOPT_TYPE_OPTIONS,
    coords_to_xyz_string,
    frame_to_xyz_string,
    get_bond_topology_from_smarts,
    load_ts_model,
    multi_xyz_string,
    parse_multiframe_xyz,
    render_reaction_svg,
    run_irc_for_xyz,
    sample_transition_states,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RitS", page_icon="\U0001F920", layout="wide")
st.title("\U0001F920 Right into the Saddle")
st.markdown("Stereochemistry-Aware Generation of Molecular Transition States")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Sampling")
num_steps = st.sidebar.slider("Sampling steps", min_value=5, max_value=100, value=25)
n_samples = st.sidebar.slider("Number of samples", min_value=1, max_value=50, value=5)
charge = st.sidebar.number_input("Molecular charge", value=0, step=1)
add_stereo = st.sidebar.checkbox("Add stereo information", value=True)

st.sidebar.markdown("---")
st.sidebar.header("IRC (post-processing)")
irc_thresh = st.sidebar.selectbox("Convergence threshold", IRC_THRESH_OPTIONS, index=1)
irc_type = st.sidebar.selectbox("IRC method", IRC_TYPE_OPTIONS, index=0)
tsopt_type = st.sidebar.selectbox("TS optimiser", TSOPT_TYPE_OPTIONS, index=0)
irc_max_cycles = st.sidebar.number_input("Max TS-opt cycles", value=1000, min_value=50, max_value=5000, step=50)
irc_hessian_recalc = st.sidebar.number_input("Hessian recalc interval", value=5, min_value=1, max_value=50, step=1)
irc_mult = st.sidebar.number_input("Spin multiplicity", value=1, min_value=1, max_value=7, step=2)

if st.sidebar.button("Clear model cache"):
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared.")

# ---------------------------------------------------------------------------
# Model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_model():
    return load_ts_model()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

example_names = list(EXAMPLE_REACTIONS.keys())
if "default_example" not in st.session_state:
    st.session_state.default_example = random.choice(example_names)

st.header("Reaction input")
input_col1, input_col2 = st.columns([1, 1])

with input_col1:
    mode = st.radio("Input mode", ["Example", "Custom SMARTS"], horizontal=True)

with input_col2:
    if mode == "Example":
        selected = st.selectbox(
            "Choose reaction",
            example_names,
            index=example_names.index(st.session_state.default_example),
        )
        reaction_smarts = EXAMPLE_REACTIONS[selected]
    else:
        reaction_smarts = st.text_area(
            "Reaction SMARTS (R>>P with atom maps)",
            value=EXAMPLE_REACTIONS[st.session_state.default_example],
            height=120,
        )

generate_btn = st.button("Generate transition states", type="primary")

st.header("2D Reaction")
if reaction_smarts and ">>" in reaction_smarts:
    svg = render_reaction_svg(reaction_smarts.strip())
    if svg:
        svg_clean = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
        svg_responsive = (
            '<div style="width:100%;">'
            + svg_clean.replace("width='1600px'", "width='100%'").replace("height='500px'", "")
            + "</div>"
        )
        st.markdown(svg_responsive, unsafe_allow_html=True)
    else:
        st.warning("Could not render 2D reaction (check SMARTS syntax)")
else:
    st.info("Enter a valid reaction SMARTS to see the 2D preview")

# ---------------------------------------------------------------------------
# Helper: 3D viewer with bonds from common topology
# ---------------------------------------------------------------------------

def _build_mol_block(symbols, coords, bonds):
    """Build a V2000 MOL block from symbols, coords and bond list."""
    n_atoms = len(symbols)
    n_bonds = len(bonds)
    lines = []
    lines.append("")  # mol name
    lines.append("     RitS")
    lines.append("")
    lines.append(f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000")
    for sym, c in zip(symbols, coords):
        lines.append(f"{c[0]:10.4f}{c[1]:10.4f}{c[2]:10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    for i, j in bonds:
        lines.append(f"{i+1:3d}{j+1:3d}  1  0  0  0  0")
    lines.append("M  END")
    return "\n".join(lines)


def render_ts_3d(symbols, coords, bonds, height=450, width=600):
    """Render a TS structure in py3Dmol using common-bond topology."""
    mol_block = _build_mol_block(symbols, coords, bonds)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer._make_html()


def _build_trajectory_html(mol_blocks, width=900, height=500):
    """Build a self-contained HTML page with all IRC frames and a JS slider.

    All data lives in the browser -- scrubbing is instant with no server
    round-trips.
    """
    n = len(mol_blocks)
    mid = n // 2
    frames_json = json.dumps(mol_blocks)
    return f"""<!DOCTYPE html>
<html>
<head>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ margin:0; padding:0; font-family: sans-serif; }}
  #viewer {{ width:{width}px; height:{height}px; position:relative; }}
  .controls {{
    padding:10px 0; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  }}
  .controls button {{
    padding:6px 16px; border:1px solid #ccc; border-radius:6px;
    background:#f8f8f8; cursor:pointer; font-size:14px;
  }}
  .controls button:hover {{ background:#e8e8e8; }}
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
<script>
var frames = {frames_json};
var viewer = $3Dmol.createViewer("viewer", {{backgroundColor:"white"}});
var current = {mid};
var playing = false;
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

showFrame({mid});
viewer.zoomTo();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

if generate_btn and reaction_smarts and ">>" in reaction_smarts:
    smarts = reaction_smarts.strip()

    with st.spinner("Loading model..."):
        model, cfg, device = _load_model()

    with st.spinner(f"Sampling {n_samples} transition state(s) ({num_steps} steps)..."):
        try:
            samples = sample_transition_states(
                smarts, model, cfg, device,
                n_samples=n_samples,
                num_steps=num_steps,
                charge=charge,
                add_stereo=add_stereo,
            )
        except Exception as e:
            st.error(f"Sampling failed: {e}")
            samples = []

    if samples:
        st.success(f"Generated {len(samples)} transition state(s)")
        st.session_state["ts_samples"] = samples
        st.session_state["ts_smarts"] = smarts
    else:
        st.warning("No samples generated")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "ts_samples" in st.session_state:
    samples = st.session_state["ts_samples"]
    smarts = st.session_state["ts_smarts"]

    _, common_bonds = get_bond_topology_from_smarts(smarts)

    st.markdown("---")
    viewer_col, info_col = st.columns([2, 1])

    with viewer_col:
        st.header("3D Transition State")
        idx = st.slider("Sample", 0, len(samples) - 1, 0) if len(samples) > 1 else 0
        symbols, coords = samples[idx]
        html = render_ts_3d(symbols, coords, common_bonds)
        components.html(html, height=470, width=650, scrolling=False)

    with info_col:
        st.header("Info")
        st.metric("Atoms", len(samples[0][0]))
        st.metric("Samples", len(samples))
        st.metric("Common bonds", len(common_bonds))

        st.subheader("Download")
        xyz_all = multi_xyz_string(samples)
        st.download_button(
            label="Download all samples (XYZ)",
            data=xyz_all,
            file_name="rits_samples.xyz",
            mime="chemical/x-xyz",
        )
        xyz_single = coords_to_xyz_string(*samples[idx])
        st.download_button(
            label=f"Download sample {idx} (XYZ)",
            data=xyz_single,
            file_name=f"rits_sample_{idx}.xyz",
            mime="chemical/x-xyz",
        )

    # -------------------------------------------------------------------
    # IRC section
    # -------------------------------------------------------------------

    st.markdown("---")
    st.header("IRC (Intrinsic Reaction Coordinate)")

    irc_sample_idx = st.selectbox(
        "Run IRC on sample",
        list(range(len(samples))),
        format_func=lambda i: f"Sample {i}",
    )

    if st.button("Run IRC", type="secondary"):
        syms, crds = samples[irc_sample_idx]
        xyz_str = coords_to_xyz_string(syms, crds)
        with st.spinner("Running pysisyphus IRC (this may take several minutes)..."):
            ok, result_text, run_dir = run_irc_for_xyz(
                xyz_str,
                charge=charge,
                mult=irc_mult,
                thresh=irc_thresh,
                max_cycles=irc_max_cycles,
                hessian_recalc=irc_hessian_recalc,
                tsopt_type=tsopt_type,
                irc_type=irc_type,
            )
        if ok:
            st.success("IRC completed successfully")
            st.session_state["irc_trajectory"] = result_text
            st.session_state["irc_smarts"] = smarts
        else:
            st.error(f"IRC failed: {result_text}")

    if "irc_trajectory" in st.session_state:
        st.subheader("IRC Trajectory Viewer")
        trj_text = st.session_state["irc_trajectory"]
        irc_smarts = st.session_state["irc_smarts"]
        frames = parse_multiframe_xyz(trj_text)

        if frames:
            _, irc_bonds = get_bond_topology_from_smarts(irc_smarts)
            mol_blocks = []
            for n_at, comment, atom_lines in frames:
                syms, crds = [], []
                for line in atom_lines:
                    parts = line.split()
                    syms.append(parts[0])
                    crds.append([float(parts[1]), float(parts[2]), float(parts[3])])
                import numpy as np
                mol_blocks.append(_build_mol_block(syms, np.array(crds), irc_bonds))

            html = _build_trajectory_html(mol_blocks)
            components.html(html, height=600, width=920, scrolling=False)

            st.download_button(
                label="Download IRC trajectory (XYZ)",
                data=trj_text,
                file_name="irc_trajectory.xyz",
                mime="chemical/x-xyz",
            )
        else:
            st.warning("Could not parse IRC trajectory frames")

