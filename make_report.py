"""
Build report.ipynb from results/ (run run_experiments.py first).

The notebook is written directly as JSON with the figures already embedded as cell
outputs, so it displays fully on GitHub without being executed and needs no Jupyter
install to build. Each code cell can also be re-run in JupyterLab.
"""
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RES = ROOT / "results"
R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
P = R["params"]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def fig(name, caption=None):
    """A code cell that displays a saved figure, with the image embedded as its output."""
    b64 = base64.b64encode((RES / name).read_bytes()).decode()
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "source": f'from IPython.display import Image\nImage("results/{name}")',
        "outputs": [{"output_type": "display_data", "metadata": {},
                     "data": {"image/png": b64, "text/plain": ["<IPython.core.display.Image object>"]}}],
    })
    if caption:
        md(f"*{caption}*")


def code(src, out_text):
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "source": src.strip("\n"),
        "outputs": [{"output_type": "stream", "name": "stdout", "text": out_text}],
    })


kT = 8.617333e-5 * P["T"]
q1, q2, q3, q4, q5 = R["q1"], R["q2"], R["q3"], R["q4"], R["q5"]

# ----------------------------------------------------------------------------- title + model
r_bind = np.exp(P['E_bind'] / kT)
r_edge = np.exp((P['eps_S'] - P['eps_Mo']) / kT)
md(f"""
# Kinetic Monte Carlo model of MoS₂ flake growth

Code: `kmc/` (model), `run_experiments.py` (all results), `make_report.py` (this report). Repository README explains how to run it.

## 1. Model

**Lattice and states.** {P['L']}×{P['L']} triangular lattice of Mo sites (periodic, run stops near the border). Mo never moves;
a site is *oxide* or *MoS₂* and only oxide → MoS₂ is allowed, so the flake grows as a conversion front from **one
converted site in the centre**. Each oxide site has {P['S_max']} S slots. At an oxide site touching the flake
(*n* = number of MoS₂ neighbours) three events are possible:
S **adsorption** $k_{{ads}}P\\,(S_{{max}}-s)$; S **desorption** $s\\,\\nu e^{{-(E_{{des}}+nE_{{bind}})/kT}}$ (harder with more MoS₂
neighbours); **conversion**, only when $s={P['S_req']}$, at $\\nu e^{{-(E_{{conv}}-n_\\Delta\\varepsilon_S-n_\\nabla\\varepsilon_{{Mo}})/kT}}$.

**Two edge types, no imposed shape.** The 6 unit triangles around a site alternate Δ/∇. In 1H-MoS₂ the S columns sit over
Δ only. A site joining an **S-edge** closes a Δ triangle (bridging S already present, $\\varepsilon_S$). A site joining a
**Mo-edge** closes a ∇ triangle ($\\varepsilon_{{Mo}}$). Opposite-facing zigzag edges are always of opposite type, so S- and
Mo-edges alternate around the flake. The code only counts triangles and never refers to a shape.

**kMC loop (BKL/Gillespie).** Per-site rate sums live in a binary sum tree. Each step picks one event with probability ∝ rate
(O(log N)), advances $\\Delta t=-\\ln u/R_{{tot}}$ and updates only the affected sites. There is no fixed time step.
Sulfur on sites away from the flake is independent of everything, so it is drawn from its steady state
(binomial, $\\theta_0$) when a site first touches the flake. This is exact, and it saves millions of irrelevant events.
Validation: the measured mean time to the first conversion at 1 Pa is 18.3 s against 18.4 s analytic (40 runs).

| parameter | value | why |
|---|---|---|
| $T$, $\\nu$ | {P['T']:.0f} K, 10¹³ s⁻¹ | CVD growth temperature (650–850 °C); lattice attempt frequency |
| $k_{{ads}}$ | {P['k_ads']:.0f} Pa⁻¹s⁻¹ per slot | Hertz–Knudsen S₂ flux on an 8.6 Å² site ≈ 900 Pa⁻¹s⁻¹, sticking ≈ 0.1 |
| $E_{{des}}$ | {P['E_des']} eV | weakly held S on oxide (≈ 65 µs); puts the growth threshold at a realistic 0.1–1 Pa |
| $E_{{bind}}$ | {P['E_bind']} eV / neighbour | ≈ one weak extra Mo–S bond; each neighbour keeps S {r_bind:.0f}× longer |
| $E_{{conv}}$ | {P['E_conv']} eV | bare sulfidation ≈ 1 s⁻¹: slow unless the site is embedded in the flake |
| $\\varepsilon_S$ / $\\varepsilon_{{Mo}}$ | {P['eps_S']} / {P['eps_Mo']} eV per triangle | S-edges join {r_edge:.1f}× faster, so Mo-edges bound the flake (S-rich CVD) |
| growth window | {P['t_max']:.0f} s | growth slower than this counts as "stopped" |
""")

# ----------------------------------------------------------------------------- Q1
md(f"""
## Q1. High pressure ({q1['P']:g} Pa): a triangle grows

The flake reaches the border (≈{np.mean(q1['final_size']):.0f} sites) in {np.mean(q1['time_to_border_s']):.1f} s. It is a clear triangle:
3-fold asymmetry $A_3$ = {np.mean(q1['A3']):.2f} (0 for a hexagon, ≈ 0.25 for a triangle), and {q1['edge_sites_S_vs_Mo'][1]}/{sum(q1['edge_sites_S_vs_Mo'])}
flat-edge sites lie on Mo-edges. After a short incubation of the lone nucleus, the side $\\sqrt{{2N}}$ grows **linearly** in time
(≈ {np.mean(q1['side_velocity_sites_per_s']):.0f} sites/s, so $N\\propto t^2$). Each slow Mo-edge advances at constant speed by
nucleating a new row and zipping it along through kinks. The colour bands (earlier outlines) show the shape is triangular throughout.
""")
fig("q1_triangle_growth.png")
md("![growth at high pressure](results/q1_growth_high_pressure.gif)")

# ----------------------------------------------------------------------------- Q2
eq = [r for r in q2 if r["case"].startswith("equal")][0]
md(f"""
## Q2. Equal edge rates → hexagon

With $k_S=k_{{Mo}}$ the flake is a **hexagon**: $A_3$ = {eq['A3_mean']:.2f} and {eq['Mo_edge_fraction']:.0%} of facets are Mo-edges.
A crystal ends up bounded by its **slowest** edges (kinetic Wulff construction). A fast edge outruns its neighbours, shortens
and vanishes into a corner, while slow edges lengthen. With S-edges fast, the 3 Mo-edges survive and form a triangle.
With equal rates all 6 edges survive and form a hexagon. Shrinking the difference gives truncated triangles. Reversing it gives
a triangle rotated by 180° and bounded by S-edges. The triangle therefore comes purely from the edge-rate asymmetry.
""")
fig("q2_edge_rates.png")

# ----------------------------------------------------------------------------- Q3
P_all = [r["P"] for r in q3 if r["fraction_grown"] == 1.0]
P_none = [r["P"] for r in q3 if r["fraction_grown"] == 0.0]
P_on, P_off = min(P_all), max(P_none)
frac = ", ".join(f"{r['P']:g}: {r['fraction_grown']:.0%}" for r in q3 if P_off <= r["P"] <= 0.6)
md(f"""
## Q3. Lowering the pressure: where growth stops

With one nucleus, {len(q3[0]['sizes'])} runs per pressure: every nucleus grows down to **{P_on:g} Pa**. The shape is unchanged,
but the edge speed falls from {q3[0]['velocity_mean']:.0f} to ≈ 5 sites/s. Below that, growth becomes a lottery
(fraction grown, P in Pa: {frac}). **At {P_off:g} Pa and below no nucleus grows.** Single-nucleus growth therefore stops at
**≈ {P_off:g}–{P_on:g} Pa**. Because conversion is irreversible, "stop" means the rate drops far below 1/(growth window). The threshold
is still sharp: the nucleus rate scales as $P^2$ (two S are needed), so a 5× longer window moves it only by √5.
""")
fig("q3_pressure_sweep.png")

# ----------------------------------------------------------------------------- Q4
cfg = {c["n"]: c for c in q4["edge_configs"]}
c1 = q4["edge_configs"][0]
P_st = q5["P"]
stall = q4["event_counts"][str(P_st)]
md(f"""
## Q4. Why it stops, and why small flakes are hit hardest

**Desorption wins at the edge.** A site converts only while it holds both S atoms, so each attempt is a race between conversion
and an S atom leaving. Next to a lone nucleus ($n$ = 1) an S atom leaves ≈ {c1['k_des_per_S']:.0f} s⁻¹, while conversion runs at
{c1['k_conv']:.1f} s⁻¹. A full site converts first only {100*c1['p_convert_before_S_leaves']:.2f}% of the time, and at low P it is
rarely full ($\\theta^2\\ll1$). The effective rate $k_{{conv}}\\theta(n)^2$ (left) falls as $P^2$ and is the first to drop below
1/(growth window) at poorly bound sites. In the stalled runs at {P_st:g} Pa (middle), sites around the nucleus lost
{stall['desorptions_by_n'][0]:,} S atoms and converted **0** times.

**Size matters** because every MoS₂ neighbour raises the desorption barrier by {P['E_bind']} eV: S stays {r_bind:.0f}× longer and
the chance of holding 2 S rises ≈ {r_bind**2:.0f}×. A lone nucleus offers only $n=1$ sites. A faceted flake offers $n=2$
(row start, P(convert) = {cfg[2]['p_convert_before_S_leaves']:.2f}) and $n=3$ kinks ({cfg[3]['p_convert_before_S_leaves']:.2f}), and a
bigger flake has more places to start a row. Starting from larger seeds (right), the stopping pressure falls from ≈ {P_off:g} Pa
(1 site) to ≈ 0.01 Pa (7–37 sites) and ≈ 0.005 Pa (217 sites). This is a **critical nucleus size**, a kinetic analogue of
Gibbs–Thomson: sharply curved, poorly coordinated edges lose their sulfur before they can convert.
""")
fig("q4_why_it_stops.png")

# ----------------------------------------------------------------------------- Q5
pairs = {r["separation"]: r["fraction_grown"] for r in q5["pairs"]}
single_grew = sum(s >= 50 for s in q5["single_sizes"])
md(f"""
## Q5. A second nucleus at {P_st:g} Pa

Alone, a nucleus never grows here ({single_grew}/{len(q5['single_sizes'])}). With a second nucleus 1 or 2 hops away, **{pairs[1]:.0%} / {pairs[2]:.0%}**
of pairs grow into a single full triangle. At 3, 4 or 6 hops, **{pairs[3]:.0%}** do. At 2 hops the nuclei share an oxide neighbour (black ring) with
$n$ = 2: its S stays {r_bind:.0f}× longer, so it is full ≈ 200× more often than any site around a lone nucleus. It converts and **bridges**
the nuclei, and the new cluster offers $n$ = 2–3 sites, so growth runs away. At 1 hop the nuclei already form a dimer. At ≥ 3 hops they
share no neighbour and each stalls alone. Nearby nuclei **cooperate**: together they exceed the critical size of Q4. In real CVD this favours
growth where nuclei are dense (steps, seeds, defects) and the coalescence of close nuclei into one flake.
""")
fig("q5_two_nuclei.png")
md("![two nuclei](results/q5_two_nuclei.gif)")

# ----------------------------------------------------------------------------- improvements
md("""
## With more time I would add

**Reversibility** (MoS₂ → oxide, MoO₃₋ₓ volatilisation), for a true thermodynamic critical nucleus. **Sulfur transport**: surface
diffusion and S₂ depletion above the flake, with *P* coupled to the chamber-scale CVD model. **DFT/NEB barriers** instead of
hand-picked values, plus temperature sweeps. **Mo/S-ratio effects** that switch the slow edge (truncated triangles). **A faster engine**
(Numba/C++, 10⁶ sites) for many nuclei, size distributions and grain boundaries between merging misoriented flakes.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
(ROOT / "report.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote report.ipynb with", len(cells), "cells")


# ----------------------------------------------------------------------------- HTML copy
def export_html():
    """
    report.html: the same report as one self-contained file (figures and GIFs embedded),
    for opening in a browser or printing to PDF. Markdown and LaTeX are rendered in the
    browser by marked.js and KaTeX, so viewing it needs an internet connection.
    """
    import re

    def gif_inline(text):
        def sub(m):
            data = base64.b64encode((ROOT / m.group(2)).read_bytes()).decode()
            return f'<img alt="{m.group(1)}" src="data:image/gif;base64,{data}" class="gif">'
        return re.sub(r"!\[([^\]]*)\]\((results/[^)]+\.gif)\)", sub, text)

    blocks = []
    for c in cells:
        if c["cell_type"] == "markdown":
            blocks.append({"md": gif_inline(c["source"])})
        else:
            for o in c.get("outputs", []):
                if "data" in o and "image/png" in o["data"]:
                    blocks.append({"img": o["data"]["image/png"]})
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoS2 kMC Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
 body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:0 auto;padding:24px 16px;
      line-height:1.5;color:#222;background:#fff}
 h1{font-size:1.7em;border-bottom:2px solid #333;padding-bottom:6px}
 h2{font-size:1.3em;margin-top:1.6em;border-bottom:1px solid #ccc}
 table{border-collapse:collapse;margin:10px 0;font-size:.92em}
 th,td{border:1px solid #bbb;padding:3px 8px;text-align:left}
 th{background:#f1f1f1}
 img{max-width:100%}
 img.gif{width:340px;display:block;margin:8px auto}
 .fig{text-align:center;margin:8px 0}
 @page{size:A4;margin:11mm 13mm}
 @media print{body{max-width:none;font-size:8.6pt;line-height:1.32;padding:0}
   h1{font-size:14pt;margin:0 0 4px;padding-bottom:2px} h2{font-size:10.5pt;margin:7px 0 2px;page-break-after:avoid}
   p{margin:3px 0} table{font-size:7.8pt;margin:3px 0} th,td{padding:1px 5px}
   img.gif{display:none} .fig{page-break-inside:avoid;margin:2px 0} .fig img{max-height:45mm;max-width:100%}
   .katex{font-size:1.02em}}
</style></head><body><div id="out"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const blocks = JSON.parse(document.getElementById('data').textContent);
const out = document.getElementById('out');
for (const b of blocks) {
  const div = document.createElement('div');
  if (b.md !== undefined) {
    // protect $...$ / $$...$$ from the markdown parser, then restore for KaTeX
    const math = [];
    let src = b.md.replace(/\\$\\$[\\s\\S]+?\\$\\$|\\$[^$\\n]+?\\$/g, m => { math.push(m); return '@@M' + (math.length - 1) + '@@'; });
    let h = marked.parse(src);
    h = h.replace(/@@M(\\d+)@@/g, (_, i) => math[+i].replace(/&/g,'&amp;').replace(/</g,'&lt;'));
    div.innerHTML = h;
  } else {
    div.className = 'fig';
    div.innerHTML = '<img src="data:image/png;base64,' + b.img + '">';
  }
  out.appendChild(div);
}
renderMathInElement(out, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});
</script></body></html>"""
    html = html.replace("__DATA__", json.dumps(blocks).replace("</", "<\\/"))
    (ROOT / "report.html").write_text(html, encoding="utf-8")
    print("wrote report.html")


export_html()
