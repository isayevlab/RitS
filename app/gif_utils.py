"""
Client-side GIF export helpers for the IRC trajectory viewer.

Each function returns a plain string (HTML / CSS / JavaScript snippet) that
is composed into the self-contained viewer iframe by viewer_utils.py.
Nothing here imports Streamlit or any heavy dependency.
"""

# ---------------------------------------------------------------------------
# CDN constants
# ---------------------------------------------------------------------------

GIF_JS_CDN     = "https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js"
GIF_WORKER_CDN = "https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js"


# ---------------------------------------------------------------------------
# HTML / CSS snippets
# ---------------------------------------------------------------------------

def gif_script_tag() -> str:
    """<script> tag that loads gif.js from CDN."""
    return f'<script src="{GIF_JS_CDN}"></script>'


def gif_css() -> str:
    """CSS rules for the GIF speed selector, download button and status label."""
    return """
  #gif-fps {
    padding:4px 8px; border:1px solid #ccc; border-radius:6px;
    font-size:13px; background:#f8f8f8; cursor:pointer;
  }
  #gif-status { font-size:13px; color:#666; min-width:160px; }
  #gif-btn { background:#e8f4e8 !important; border-color:#8bc98b !important; }
  #gif-btn:hover:not(:disabled) { background:#d0ebd0 !important; }"""


def gif_controls_html() -> str:
    """HTML row with the FPS selector, download button and status span."""
    return """\
<div class="controls" style="border-top:1px solid #eee; padding-top:8px; margin-top:2px;">
  <label style="font-size:13px; color:#555;">GIF speed:</label>
  <select id="gif-fps">
    <option value="5">5 fps</option>
    <option value="10" selected>10 fps</option>
    <option value="15">15 fps</option>
    <option value="25">25 fps</option>
  </select>
  <button id="gif-btn" onclick="downloadGif()">&#11015; Download GIF (current view)</button>
  <span id="gif-status"></span>
</div>"""


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

def gif_download_js(n_frames: int, n_minus_1: int) -> str:
    """Return the ``downloadGif()`` JavaScript function as a string.

    The returned string contains literal JS braces and is safe to embed
    directly inside a parent f-string (Python does not re-process string
    values inserted via ``{expr}``).

    Parameters
    ----------
    n_frames:
        Total number of trajectory frames (used in loop bound and status).
    n_minus_1:
        ``n_frames - 1`` (used in frame-label strings).
    """
    return f"""\
async function downloadGif() {{
  var fpsEl  = document.getElementById("gif-fps");
  var fps    = parseInt(fpsEl ? fpsEl.value : "10");
  var delay  = Math.round(1000 / fps);
  var status = document.getElementById("gif-status");
  var btn    = document.getElementById("gif-btn");

  btn.disabled = true;
  if (playing) {{ togglePlay(); }}
  status.textContent = "Loading encoder\u2026";

  try {{
    // Fetch gif.js worker via a Blob URL to sidestep CORS restrictions
    var resp = await fetch("{GIF_WORKER_CDN}");
    var workerText = await resp.text();
    var workerBlob = new Blob([workerText], {{type: "text/javascript"}});
    var workerURL  = URL.createObjectURL(workerBlob);

    // Snapshot the camera angle the user has set
    var savedView    = viewer.getView();
    var savedCurrent = current;

    var gif = new GIF({{
      workers:      2,
      quality:      8,
      workerScript: workerURL
    }});

    gif.on("progress", function(p) {{
      status.textContent = "Encoding: " + Math.round(p * 100) + "%";
    }});

    gif.on("finished", function(blob) {{
      // Trigger browser download
      var url = URL.createObjectURL(blob);
      var a   = document.createElement("a");
      a.href  = url;
      a.download = "irc_trajectory.gif";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      URL.revokeObjectURL(workerURL);

      // Restore the viewer to the frame the user was on
      viewer.removeAllModels();
      viewer.addModel(frames[savedCurrent], "mol");
      viewer.setStyle({{}}, {{stick:{{radius:0.15}}, sphere:{{scale:0.25}}}});
      viewer.setView(savedView);
      viewer.render();
      current = savedCurrent;
      document.getElementById("slider").value = savedCurrent;
      document.getElementById("frame-label").textContent =
        "Frame " + savedCurrent + "/{n_minus_1}";

      status.textContent = "Done \u2713";
      btn.disabled = false;
      setTimeout(function() {{ status.textContent = ""; }}, 3000);
    }});

    // Render every frame at the saved view angle and capture it as a PNG
    function captureNextFrame(i) {{
      if (i >= {n_frames}) {{
        status.textContent = "Encoding GIF\u2026";
        gif.render();
        return;
      }}
      status.textContent = "Capturing " + (i + 1) + "\u202f/\u202f{n_frames}";

      viewer.removeAllModels();
      viewer.addModel(frames[i], "mol");
      viewer.setStyle({{}}, {{stick:{{radius:0.15}}, sphere:{{scale:0.25}}}});
      viewer.setView(savedView);
      viewer.render();

      // Double-RAF: guarantees WebGL has finished writing before we read pixels
      requestAnimationFrame(function() {{
        requestAnimationFrame(function() {{
          var canvas = document.querySelector("#viewer canvas");
          if (!canvas) {{ captureNextFrame(i + 1); return; }}
          var dataURL = canvas.toDataURL("image/png");
          var img = new Image();
          img.onload = function() {{
            gif.addFrame(img, {{delay: delay}});
            captureNextFrame(i + 1);
          }};
          img.src = dataURL;
        }});
      }});
    }}

    captureNextFrame(0);

  }} catch(e) {{
    status.textContent = "Error: " + e.message;
    btn.disabled = false;
  }}
}}"""
