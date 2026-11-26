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

# ----------------------------------------------------------------------------- title
md(f"""
# Kinetic Monte Carlo model of MoS₂ flake growth

**Task:** CVD conversion of MoO₃ to MoS₂ on a triangular Mo lattice, simulated with
kinetic Monte Carlo. Code: `kmc/` (model), `run_experiments.py` (all results),
`make_report.py` (this notebook).

## 1. The model

* **Lattice.** {P['L']}×{P['L']} triangular lattice of Mo sites (6 neighbours each). Mo never moves;
  each site is either *oxide* or *MoS₂*, and only oxide → MoS₂ is allowed. The flake grows
  as a conversion front starting from **one converted site in the centre**.
* **Sulfur.** Every oxide site has {P['S_max']} S slots (MoS₂ has 2 S per Mo).
* **Three events** at an oxide site that touches the flake (*n* = its number of MoS₂ neighbours):

| event | rate |
|---|---|
| S adsorption | $k_{{ads}}\\,P\\,(S_{{max}}-s)$, proportional to pressure |
| S desorption | $s\\,\\nu\\,e^{{-(E_{{des}}+n\\,E_{{bind}})/kT}}$, harder with more MoS₂ neighbours |
| conversion (only if $s = {P['S_req']}$) | $\\nu\\,e^{{-(E_{{conv}}-n_\\Delta\\varepsilon_S-n_\\nabla\\varepsilon_{{Mo}})/kT}}$ |

* **Two edge types without forcing a shape.** The 6 unit triangles around a site alternate
  between Δ and ∇ orientation. In 1H-MoS₂ the S columns sit over the Δ triangles only. A site joining
  an **S-terminated edge** closes a Δ triangle, so its bridging S is already there, which makes
  conversion easier ($\\varepsilon_S$). A site joining a **Mo-terminated edge** closes a ∇ triangle
  ($\\varepsilon_{{Mo}}$). Edges facing opposite directions are always of opposite type, so the
  six possible zigzag edges alternate S, Mo, S, Mo… The code only counts triangles and never mentions a
  shape.
* **kMC loop (BKL/Gillespie).** All event rates are kept in a binary sum tree. Each step picks one
  event with probability ∝ its rate (O(log N)), advances time by $\\Delta t=-\\ln u/R_{{tot}}$,
  and updates only the affected sites. There is no fixed time step.
* **Exact shortcut.** Sulfur far from the flake never affects anything, so it is not simulated. A site gets its
  steady-state sulfur (binomial with $\\theta_0$) the moment it first touches the flake. An independent process
  that starts in steady state stays there, so this is exact.

### Parameters and why

| parameter | value | reasoning |
|---|---|---|
| $T$ | {P['T']:.0f} K | typical CVD temperature (650–850 °C) |
| $\\nu$ | {P['nu']:.0e} s⁻¹ | lattice-vibration attempt frequency |
| $k_{{ads}}$ | {P['k_ads']:.0f} Pa⁻¹s⁻¹ per slot | Hertz–Knudsen S₂ flux on one 8.6 Å² site ≈ 900 Pa⁻¹s⁻¹, sticking ≈ 0.1 |
| $E_{{des}}$ | {P['E_des']} eV | weakly held S on oxide (≈ 65 µs); puts the growth threshold at 0.1–1 Pa |
| $E_{{bind}}$ | {P['E_bind']} eV / neighbour | ≈ one extra weak Mo–S bond; each neighbour keeps S {np.exp(P['E_bind']/kT):.0f}× longer |
| $E_{{conv}}$ | {P['E_conv']} eV | bare sulfidation ≈ 1 s⁻¹, slow unless the site is embedded in the flake |
| $\\varepsilon_S,\\ \\varepsilon_{{Mo}}$ | {P['eps_S']}, {P['eps_Mo']} eV / triangle | S-edges join {np.exp((P['eps_S']-P['eps_Mo'])/kT):.1f}× faster, so Mo-edges bound the flake, as seen in S-rich CVD |
| growth window | {P['t_max']:.0f} s | how long we watch. Growth slower than this counts as "stopped" |
""")

# ----------------------------------------------------------------------------- Q1
t_mean = np.mean(q1["time_to_border_s"])
md(f"""
## Q1. High sulfur pressure ({q1['P']:g} Pa): a triangle grows

The flake reaches the lattice border (≈{np.mean(q1['final_size']):.0f} Mo sites) after
{t_mean:.2f} s on average ({len(q1['final_size'])} runs). It is a clear triangle: the 3-fold asymmetry
$A_3$ = {np.mean(q1['A3']):.2f} (0 for a hexagon or circle, ≈ 0.25 for a triangle), and
{q1['edge_sites_S_vs_Mo'][1]} of {sum(q1['edge_sites_S_vs_Mo'])} flat-edge sites are on Mo-edges.

**Size against time.** After a short wait for the lone nucleus to make its first conversions, the
equivalent side $\\sqrt{{2N}}$ grows **linearly** with time
(≈ {np.mean(q1['side_velocity_sites_per_s']):.0f} sites/s), i.e. $N \\propto t^2$. Each Mo-edge advances
at constant speed: a new row has to nucleate on the slow Mo-edge, and then it zips along quickly through kinks.
The colour bands mark the flake outline at earlier times. It is triangular from very early on.
""")
fig("q1_triangle_growth.png")
md("![growth at high pressure](results/q1_growth_high_pressure.gif)")

# ----------------------------------------------------------------------------- Q2
rows = "\n".join(f"| {r['case']} | {r['rate_ratio_S_over_Mo']:.2f} | {r['A3_mean']:.2f} ± {r['A3_std']:.2f} | {r['Mo_edge_fraction']:.0%} |" for r in q2)
md(f"""
## Q2. Equal edge rates: the flake becomes a hexagon

| case | $k_S/k_{{Mo}}$ | $A_3$ | Mo-edge share |
|---|---|---|---|
{rows}

**Why.** A growing crystal ends up bounded by its **slowest** edges (the kinetic Wulff
construction). A fast edge outruns its neighbours, gets shorter and disappears into a corner. The slow edges
are left behind and get longer. With S-edges fast, the three Mo-edges survive and form a triangle. With equal rates
there is no winner: all six zigzag edges survive and the flake is a **hexagon** ($A_3 \\approx 0$, 50/50 edges).
Reversing the rates gives a triangle again, but rotated by 180° and bounded by S-edges. Triangles in experiment
therefore indicate that one edge termination grows much faster than the other, which depends on the S:Mo ratio.
""")
fig("q2_edge_rates.png")

# ----------------------------------------------------------------------------- Q3
P_all = [r["P"] for r in q3 if r["fraction_grown"] == 1.0]
P_none = [r["P"] for r in q3 if r["fraction_grown"] == 0.0]
P_on, P_off = min(P_all), max(P_none)
rows = "\n".join(f"| {r['P']:g} | {r['fraction_grown']:.0%} | {r['median_size']:.0f} | {r['velocity_mean']:.1f} |" for r in q3)
md(f"""
## Q3. Lowering the sulfur pressure: where does growth stop?

One nucleus, {len(q3[0]['sizes'])} runs per pressure, {P['t_max']:.0f} s growth window:

| P (Pa) | nuclei that grew | median final size | edge speed (sites/s) |
|---|---|---|---|
{rows}

* Down to **{P_on:g} Pa** every nucleus grows into a full triangle. The growth is slower (the edge speed falls
  steeply with P) but the shape is unchanged.
* Between {P_on:g} and {P_off:g} Pa it becomes a **lottery**. Some nuclei never get going, but those that do
  grow all the way.
* At **{P_off:g} Pa and below**, no single nucleus grows: the flake stays at 1 site.

So growth stops at **P ≈ {P_off:g}–{P_on:g} Pa** for a single nucleus. Because conversion is irreversible in this
model, "stop" means the rate falls far below 1/(growth window). The transition is sharp anyway: the rate at the
nucleus scales as $P^2$ (two S needed), so a 5× longer window would only move the threshold by √5.
""")
fig("q3_pressure_sweep.png")

# ----------------------------------------------------------------------------- Q4
cfg = q4["edge_configs"]
rows = "\n".join(f"| {c['config']} | {c['k_des_per_S']:.3g} | {c['P_half_coverage_Pa']:.3g} | {c['k_conv']:.3g} | {c['p_convert_before_S_leaves']:.1e} |" for c in cfg)
ev = q4["event_counts"]
P_st = q5["P"]                     # the "stall" pressure used in Q4b and Q5
stall = ev[str(P_st)]
cs = q4["critical_size"]
md(f"""
## Q4. Why it stops, and why small flakes are hit hardest

**Desorption wins.** An edge site can only convert while it holds both S atoms. That turns it into a race between
*conversion* and *an S atom leaving*:

| edge site | $k_{{des}}$ per S (s⁻¹) | $P_{{1/2}}$ for half-filled slots (Pa) | $k_{{conv}}$ (s⁻¹) | P(convert before an S leaves) |
|---|---|---|---|---|
{rows}

At a site touching the flake with only **one** neighbour, an S atom leaves ≈ 800 times per second, while
conversion happens about once per second. At low P the site is almost never full ($\\theta^2 \\ll 1$),
and when it is full, an S atom leaves first > 99.9% of the time. The effective attachment rate
$k_{{conv}}\\theta(n)^2$ (left panel) falls as $P^2$ and drops below 1/(growth window) first for the
weakly bound sites. In the stalled runs at {P_st:g} Pa (middle panel), the edge sites around the lone
nucleus lost {stall['desorptions_by_n'][0]} S atoms and converted **0** times.

**Why size matters.** Each MoS₂ neighbour raises the desorption barrier by {P['E_bind']} eV, so S stays
{np.exp(P['E_bind']/kT):.0f}× longer per neighbour, and the chance of holding 2 S rises by ≈ {np.exp(2*P['E_bind']/kT):.0f}×.
* A **lone nucleus** offers only $n=1$ sites, the worst case.
* A flake with straight edges offers $n=2$ sites (to start a new row) and $n=3$ kinks (to finish it).
* A larger flake also has longer edges, so there are more places to start a row.

The right panel starts from hexagonal seeds of increasing size. The pressure at which growth stops falls
from ≈{P_off:g}–{P_on:g} Pa for a single site to ≈ 0.01–0.02 Pa for 7–37 sites, ≈ 0.007 Pa for 91 sites and
≈ 0.005 Pa for 217 sites. Bigger nuclei keep growing at lower pressure. This is a
**critical nucleus size**: the kinetic version of the Gibbs–Thomson effect. Sharply curved, poorly coordinated
edges lose their sulfur before they can convert.
""")
fig("q4_why_it_stops.png")

# ----------------------------------------------------------------------------- Q5
rows = "\n".join(f"| {r['separation']} | {r['fraction_grown']:.0%} | {int(np.median(r['sizes']))} |" for r in q5["pairs"])
single_grew = sum(s >= 50 for s in q5["single_sizes"])
md(f"""
## Q5. A second nucleus close to the first, at {q5['P']:g} Pa

At {q5['P']:g} Pa a lone nucleus never grows ({single_grew}/{len(q5['single_sizes'])} runs).
Adding a second nucleus at distance *d* (lattice hops):

| d | pairs that grew | median final size |
|---|---|---|
{rows}

**What happens.** At *d* = 2 the two nuclei share an oxide neighbour (black ring in the figure). That site has
$n = 2$, so its S stays {np.exp(P['E_bind']/kT):.0f}× longer and it is full ≈ 200× more often than any site around
a lone nucleus. It converts within tens of seconds and **bridges** the two nuclei. The new 3-site cluster
offers more $n=2$ and $n=3$ sites, so the next conversions are even faster and the pair grows into
**one single triangle**. At *d* = 1 the two nuclei already form a dimer, which works the same way. At *d* ≥ 3 the
nuclei share no neighbour, so each one behaves as a lone nucleus and both stall.

**Why it matters.** A flake that is too small to grow alone becomes supercritical when its edge
sites get extra MoS₂ neighbours from another flake. Neighbouring nuclei cooperate, and this is the
same physics as Q4. In real CVD this favours growth where nuclei are dense (substrate steps, seeds,
defects) and leads to coalescence of nearby nuclei into one larger flake.
""")
fig("q5_two_nuclei.png")
md("![two nuclei](results/q5_two_nuclei.gif)")

# ----------------------------------------------------------------------------- improvements
md("""
## What I would improve with more time

1. **Reversibility.** Let MoS₂ edge sites lose S and revert (or let Mo etch away as volatile MoO₃₋ₓ). Then a
   true critical nucleus and an equilibrium shape exist, instead of a "slow enough" kinetic stop.
2. **Sulfur transport.** Model S diffusion on the surface and depletion of S₂ above the flake (boundary layer).
   Couple *P* to the chamber-scale CVD model, so each region of the substrate sees its own local pressure.
3. **Barriers from DFT.** Replace the hand-picked $E$ and $\\varepsilon$ with DFT/NEB values for S₂ adsorption,
   desorption and Mo-edge versus S-edge sulfidation, and add temperature sweeps (Arrhenius plots of growth speed).
4. **Mo supply.** Here the Mo is already on the substrate. Real growth also involves MoO₃₋ₓ vapour and
   Mo/S ratio effects, which switch the slow edge and give truncated triangles and hexagons.
5. **Scale and speed.** Use a Numba or C++ engine and bigger lattices (10⁶ sites). Run many nuclei to get size
   distributions and nucleation density versus P, and to see grain boundaries when misoriented flakes meet.
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
 @media print{body{max-width:none;font-size:10.5pt} img.gif{display:none} h2{page-break-after:avoid}
              .fig{page-break-inside:avoid}}
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
