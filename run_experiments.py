"""
Run every experiment for questions 1-5 and save figures, GIFs and numbers to results/.

    python run_experiments.py            # full run (~5-10 min)
    python run_experiments.py --quick    # fewer repeats, for a fast check (~1-2 min)
    python run_experiments.py --only q1 q5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmc import Params, Simulation, TriangularLattice
from kmc.analysis import threefold_asymmetry, edge_type_fraction, equivalent_side, growth_velocity
from kmc.engine import EVENT_NAMES
from kmc.viz import draw_flake, crop_box, growth_movie

OUT = Path(__file__).parent / "results"
BASE = Params()
LAT = TriangularLattice(BASE.L)
CENTER = LAT.center()
P_HIGH = 10.0     # "high" sulfur pressure, Pa
P_STALL = 0.1     # a pressure where a lone nucleus does not grow (found in Q3)
GROWN = 50        # a run counts as "grew" if the flake passes this many sites


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   saved", name)


def run(params, nuclei=None, seed=0):
    return Simulation(params, nuclei=nuclei, seed=seed, lattice=LAT).run()


# =============================================================================
def q1(R, reps):
    """Q1: growth at high sulfur pressure -> triangle; size vs time."""
    print("Q1: high pressure growth")
    p = BASE.with_(P=P_HIGH)
    sims = [run(p, seed=s) for s in range(reps)]
    main = sims[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    sc = draw_flake(axes[0], LAT, main.conv_time, crop=crop_box(LAT, main.conv_time))
    fig.colorbar(sc, ax=axes[0], location="left", shrink=0.8,
                 label="fraction of final flake when this site converted")
    axes[0].set_title(f"P = {P_HIGH:g} Pa: final flake ({main.n_conv} sites)")
    ax = axes[1]
    for k, s in enumerate(sims):
        ax.plot(s.history_t, equivalent_side(s.history_n), lw=1.2, alpha=0.8, label=f"run {k}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"equivalent triangle side  $\sqrt{2N}$  (lattice units)")
    ax.set_title("Flake size against time")
    ax2 = ax.twinx()
    ax2.plot(main.history_t, main.history_n, "k:", lw=1)
    ax2.set_ylabel("N (MoS$_2$ sites), dotted = run 0")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "q1_triangle_growth.png")

    growth_movie(LAT, main.conv_time, OUT / "q1_growth_high_pressure.gif",
                 title=f"MoS$_2$ growth, P = {P_HIGH:g} Pa", crop=crop_box(LAT, main.conv_time))
    print("   saved q1_growth_high_pressure.gif")

    nS, nMo = edge_type_fraction(LAT, main.mask())
    R["q1"] = {
        "P": P_HIGH,
        "final_size": [s.n_conv for s in sims],
        "time_to_border_s": [round(s.t, 3) for s in sims],
        "side_velocity_sites_per_s": [round(growth_velocity(s.history_t, s.history_n), 2) for s in sims],
        "A3": [round(threefold_asymmetry(LAT, s.mask()), 3) for s in sims],
        "edge_sites_S_vs_Mo": [nS, nMo],
        "events": [s.n_events for s in sims],
    }


# =============================================================================
def q2(R, reps):
    """Q2: what if both edge types convert at the same rate?"""
    print("Q2: edge anisotropy")
    mean = (BASE.eps_S + BASE.eps_Mo) / 2
    cases = [
        ("S-edge fast (default)", BASE.eps_S, BASE.eps_Mo),
        ("smaller difference", mean + 0.0375, mean - 0.0375),
        ("equal edge rates", mean, mean),
        ("Mo-edge fast (reversed)", BASE.eps_Mo, BASE.eps_S),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    out = []
    for ax, (label, eS, eMo) in zip(axes, cases):
        p = BASE.with_(P=P_HIGH, eps_S=eS, eps_Mo=eMo)
        sims = [run(p, seed=s) for s in range(reps)]
        A3 = [threefold_asymmetry(LAT, s.mask()) for s in sims]
        edges = np.array([edge_type_fraction(LAT, s.mask()) for s in sims])
        fMo = edges[:, 1].sum() / max(edges.sum(), 1)
        ratio = np.exp((eS - eMo) / BASE.kT)
        draw_flake(ax, LAT, sims[0].conv_time, crop=crop_box(LAT, sims[0].conv_time))
        ax.set_title(f"{label}\n$k_S/k_{{Mo}}$ = {ratio:.2f}   A3 = {np.mean(A3):.2f}\n"
                     f"Mo-edge share of facets = {fMo:.0%}", fontsize=10)
        out.append({"case": label, "eps_S": eS, "eps_Mo": eMo, "rate_ratio_S_over_Mo": round(ratio, 3),
                    "A3_mean": round(float(np.mean(A3)), 3), "A3_std": round(float(np.std(A3)), 3),
                    "Mo_edge_fraction": round(float(fMo), 3)})
    save(fig, "q2_edge_rates.png")
    R["q2"] = out

    p = BASE.with_(P=P_HIGH, eps_S=mean, eps_Mo=mean)
    s = run(p, seed=0)
    growth_movie(LAT, s.conv_time, OUT / "q2_equal_edges.gif",
                 title="equal edge rates", crop=crop_box(LAT, s.conv_time))


# =============================================================================
def q3(R, reps):
    """Q3: lower the sulfur pressure step by step."""
    print("Q3: pressure sweep")
    pressures = [10, 3, 1, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    rows = []
    for P in pressures:
        t0 = time.time()
        sims = [run(BASE.with_(P=P), seed=s) for s in range(reps)]
        sizes = [s.n_conv for s in sims]
        grown = [s for s in sims if s.n_conv >= GROWN]
        v = [growth_velocity(s.history_t, s.history_n) for s in grown]
        # time the nucleus needed to make its first conversion (incubation)
        first = [s.history_t[1] if len(s.history_t) > 1 else np.nan for s in sims]
        rows.append({
            "P": P, "sizes": sizes,
            "fraction_grown": len(grown) / reps,
            "median_size": float(np.median(sizes)),
            "velocity_mean": float(np.mean(v)) if v else 0.0,
            "first_conversion_s": [None if np.isnan(f) else round(f, 3) for f in first],
        })
        print(f"   P={P:<5g} grew {len(grown)}/{reps}  median size {np.median(sizes):.0f}  ({time.time()-t0:.1f}s)")
    R["q3"] = rows

    Ps = np.array([r["P"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for r in rows:
        ax.scatter([r["P"]] * len(r["sizes"]), r["sizes"], s=14, alpha=0.5, color="C0")
    ax.plot(Ps, [r["median_size"] for r in rows], "C0-", lw=1.5, label="median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sulfur pressure P (Pa)"); ax.set_ylabel(f"flake size after {BASE.t_max:g} s (sites)")
    ax.set_title("Final size (each dot = one run)")
    ax.legend()
    ax = axes[1]
    ax.plot(Ps, [r["fraction_grown"] for r in rows], "o-", color="C3")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("sulfur pressure P (Pa)"); ax.set_ylabel("fraction of nuclei that grew")
    ax.set_title("Does a single nucleus grow?")
    ax = axes[2]
    sel = [r for r in rows if r["velocity_mean"] > 0]
    ax.plot([r["P"] for r in sel], [r["velocity_mean"] for r in sel], "s-", color="C2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sulfur pressure P (Pa)"); ax.set_ylabel("edge advance rate (sites / s)")
    ax.set_title("Growth speed once growing")
    save(fig, "q3_pressure_sweep.png")


# =============================================================================
def q4(R, reps):
    """Q4: why growth stops -- edge competition and the role of flake size."""
    print("Q4: why growth stops")
    p = BASE
    # (a) effective conversion rate of each edge configuration vs pressure.
    #     A site converts only while holding 2 S; with fast S exchange this
    #     happens with probability theta(n)^2, so k_eff = k_conv * theta(n)^2.
    configs = [  # label, n neighbours, n_delta, n_nabla
        ("corner / lone nucleus (n=1)", 1, 0, 0),
        ("Mo-edge facet (n=2)", 2, 0, 1),
        ("S-edge facet (n=2)", 2, 1, 0),
        ("kink (n=3)", 3, 1, 1),
    ]
    Pgrid = np.logspace(-2.5, 1.5, 200)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    table = []
    for label, n, nd, nn in configs:
        k = p.conv_rate(nd, nn)
        d = p.des_rate(n)
        keff = k * np.array([p.coverage(n, P) for P in Pgrid]) ** 2
        ax.plot(Pgrid, keff, lw=2, label=label)
        table.append({"config": label, "n": n, "k_conv": k, "k_des_per_S": d,
                      "P_half_coverage_Pa": d / p.k_ads,
                      "p_convert_before_S_leaves": k / (k + 2 * d)})
    ax.axhline(1 / p.t_max, color="k", ls="--", lw=1)
    ax.text(Pgrid[0] * 1.2, 1.4 / p.t_max, "1 / growth window", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sulfur pressure P (Pa)"); ax.set_ylabel(r"$k_{eff}=k_{conv}\,\theta(n)^2$  (1/s)")
    ax.set_title("Effective conversion rate per edge site")
    ax.legend(fontsize=8)

    # (b) measured: how many S atoms leave for every successful conversion,
    #     split by how many MoS2 neighbours the edge site has.
    ax = axes[1]
    width = 0.27
    measured = {}
    ns = np.arange(1, 5)
    for k, P in enumerate([P_HIGH, 0.3, P_STALL]):
        sims = [run(BASE.with_(P=P), seed=sd) for sd in range(reps)]
        c = sum(s.counts for s in sims)
        des, conv = c[1, 1:5], c[2, 1:5]
        ratio = np.where(conv > 0, des / np.maximum(conv, 1), np.nan)
        measured[str(P)] = {"desorptions_by_n": des.tolist(), "conversions_by_n": conv.tolist(),
                            "final_sizes": [s.n_conv for s in sims]}
        x = ns + (k - 1) * width
        ax.bar(x, np.nan_to_num(ratio), width, color=f"C{k}",
               label=f"P = {P:g} Pa (median size {int(np.median([s.n_conv for s in sims]))})")
        for xi, r, dd, cc in zip(x, ratio, des, conv):
            if cc == 0 and dd > 0:   # S left many times but nothing ever converted
                ax.text(xi, 1.2, f"{dd} : 0", rotation=90, ha="center", va="bottom", fontsize=7, color=f"C{k}")
    ax.set_xticks(ns); ax.set_xticklabels([f"n = {n}" for n in ns])
    ax.set_yscale("log"); ax.set_ylim(1, None)
    ax.set_xlabel("MoS$_2$ neighbours of the edge site")
    ax.set_ylabel("S desorptions per successful conversion")
    ax.set_title("Which event wins at the edge?")
    ax.legend(fontsize=8)

    # (c) critical nucleus size: start from hexagonal seeds of different radius
    ax = axes[2]
    radii = [0, 1, 2, 3, 5, 8]
    Ps = [0.3, 0.1, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003]
    grid = np.zeros((len(radii), len(Ps)))
    for a, r in enumerate(radii):
        seed_sites = LAT.hex_cluster(CENTER, r)
        for b, P in enumerate(Ps):
            ok = 0
            for sd in range(reps):
                s = run(BASE.with_(P=P), nuclei=seed_sites, seed=sd)
                ok += s.n_conv >= len(seed_sites) + 100
            grid[a, b] = ok / reps
        print(f"   seed radius {r} ({len(seed_sites)} sites): {grid[a]}")
    im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(Ps))); ax.set_xticklabels([f"{P:g}" for P in Ps])
    ax.set_yticks(range(len(radii)))
    ax.set_yticklabels([f"r={r} ({len(LAT.hex_cluster(CENTER, r))})" for r in radii])
    ax.set_xlabel("sulfur pressure P (Pa)"); ax.set_ylabel("starting nucleus (radius, sites)")
    ax.set_title("Fraction of runs that keep growing")
    for a in range(len(radii)):
        for b in range(len(Ps)):
            ax.text(b, a, f"{grid[a, b]:.1f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "q4_why_it_stops.png")
    R["q4"] = {"edge_configs": table, "event_counts": measured,
               "critical_size": {"radii": radii, "sites": [len(LAT.hex_cluster(CENTER, r)) for r in radii],
                                 "P": Ps, "fraction_grown": grid.tolist()}}


# =============================================================================
def q5(R, reps):
    """Q5: a second nucleus next to the first, at a pressure where one alone stalls."""
    print("Q5: two nuclei")
    p = BASE.with_(P=P_STALL)
    seps = [1, 2, 3, 4, 6]
    rows = []
    for d in seps:
        other = LAT.site_at_offset(CENTER, 0, d)
        sims = [run(p, nuclei=[CENTER, other], seed=s) for s in range(reps)]
        g = sum(s.n_conv >= GROWN for s in sims)
        rows.append({"separation": d, "fraction_grown": g / reps, "sizes": [s.n_conv for s in sims]})
        print(f"   separation {d}: grew {g}/{reps}")
    single = [run(p, seed=s).n_conv for s in range(reps)]
    R["q5"] = {"P": P_STALL, "pairs": rows, "single_sizes": single}

    # snapshots + movie for separation 2 (the two nuclei share one neighbouring site)
    other = LAT.site_at_offset(CENTER, 0, 2)
    shared = set(LAT.nbr[CENTER]) & set(LAT.nbr[other])
    # Show a typical run: in ~95% of runs the shared site converts first (we checked
    # this over 60 seeds); pick the first seed that does so.
    for sd in range(50):
        s = run(p, nuclei=[CENTER, other], seed=sd)
        if np.argsort(s.conv_time)[2] in shared and s.n_conv >= GROWN:
            break
    far = run(p, nuclei=[CENTER, LAT.site_at_offset(CENTER, 0, 4)], seed=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.bar([str(r["separation"]) for r in rows], [r["fraction_grown"] for r in rows], color="C4")
    ax.axhline(sum(x >= GROWN for x in single) / reps, color="k", ls="--", label="single nucleus")
    ax.set_xlabel("distance between nuclei (lattice hops)"); ax.set_ylabel("fraction that grew")
    ax.set_title(f"P = {P_STALL:g} Pa (lone nucleus: {sum(x >= GROWN for x in single)}/{reps} grew)")
    ax.legend()

    # zoom on the first conversions: nuclei in red, then numbered in the order they converted
    ax = axes[1]
    cx, cy = LAT.x[CENTER] + 1, LAT.y[CENTER]
    zoom = ((cx - 5, cx + 5), (cy - 5, cy + 5))
    draw_flake(ax, LAT, s.conv_time, t=-1, crop=zoom)   # oxide background only
    order = np.argsort(s.conv_time)
    nuclei = [CENTER, other]
    later = [i for i in order[:14] if i not in nuclei and np.isfinite(s.conv_time[i])]
    ax.scatter(LAT.x[nuclei], LAT.y[nuclei], s=420, marker="h", c="#c0392b", linewidths=0)
    ax.scatter(LAT.x[later], LAT.y[later], s=420, marker="h", c="#2e86c1", linewidths=0)
    for k, i in enumerate(later, 1):
        ax.text(LAT.x[i], LAT.y[i], str(k), ha="center", va="center", color="w", fontsize=8, weight="bold")
    shared = set(LAT.nbr[CENTER]) & set(LAT.nbr[other])
    ax.scatter(LAT.x[list(shared)], LAT.y[list(shared)], s=520, marker="h", facecolors="none",
               edgecolors="k", linewidths=1.5)
    ax.set_title(f"red = nuclei, black ring = shared neighbour (n = 2)\n"
                 f"numbers = order of conversion; #1 at t = {s.conv_time[later[0]]:.1f} s", fontsize=10)
    draw_flake(axes[2], LAT, s.conv_time, crop=crop_box(LAT, s.conv_time))
    axes[2].set_title(f"separation 2: final flake ({s.n_conv} sites)\nseparation 4: stays at {far.n_conv} sites")
    save(fig, "q5_two_nuclei.png")
    growth_movie(LAT, s.conv_time, OUT / "q5_two_nuclei.gif",
                 title=f"two nuclei 2 hops apart, P = {P_STALL:g} Pa", crop=crop_box(LAT, s.conv_time))
    print("   saved q5_two_nuclei.gif")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer repeats")
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. q1 q3")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / "results.json"
    R = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    R["params"] = BASE.as_dict()
    reps = {"q1": 3, "q2": 3, "q3": 10, "q4": 5, "q5": 10}
    if a.quick:
        reps = {"q1": 2, "q2": 1, "q3": 4, "q4": 2, "q5": 4}
    t0 = time.time()
    for name, fn in [("q1", q1), ("q2", q2), ("q3", q3), ("q4", q4), ("q5", q5)]:
        if a.only and name not in a.only:
            continue
        fn(R, reps[name])
        path.write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
