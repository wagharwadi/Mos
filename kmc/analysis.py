"""Measurements on a flake: size, shape (triangle vs hexagon) and which edges bound it."""
import numpy as np

from .lattice import triangle_types


def boundary_sites(lat, conv):
    """MoS2 sites with at least one oxide neighbour."""
    return [i for i in np.flatnonzero(conv) if not all(conv[j] for j in lat.nbr[i])]


def threefold_asymmetry(lat, conv):
    """
    How triangular the flake is, from 0 to ~0.3.

    Take the boundary as r(phi) around the centroid and measure its 3-fold
    Fourier component:   A3 = |sum r e^{3 i phi}| / sum r.
    A hexagon or a circle has no 3-fold component (A3 ~ 0); a triangle does
    (A3 ~ 0.2-0.25, because its corners stick out twice as far as its edge midpoints).
    """
    conv = np.asarray(conv, dtype=bool)
    if conv.sum() < 7:
        return 0.0
    cx, cy = lat.x[conv].mean(), lat.y[conv].mean()
    b = boundary_sites(lat, conv)
    dx, dy = lat.x[b] - cx, lat.y[b] - cy
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    return float(abs(np.sum(r * np.exp(3j * phi))) / np.sum(r))


def edge_type_fraction(lat, conv):
    """
    Look at the oxide sites sitting on flat parts of the edge and classify them:
      closes only Delta triangles -> sits on an S-terminated edge
      closes only nabla triangles -> sits on a Mo-terminated edge
    Returns (n_S_edge, n_Mo_edge). A Mo-edge triangle gives mostly Mo; a hexagon ~50/50.
    """
    conv = np.asarray(conv, dtype=bool)
    nS = nMo = 0
    for i in range(lat.N):
        if conv[i]:
            continue
        nb = lat.nbr[i]
        if not any(conv[j] for j in nb):
            continue
        nd, nn = triangle_types(conv, nb)
        if nd and not nn:
            nS += 1
        elif nn and not nd:
            nMo += 1
    return nS, nMo


def equivalent_side(n_sites):
    """Side length (lattice units) of an equilateral triangle with n_sites sites: n ~ s^2/2."""
    return np.sqrt(2.0 * np.asarray(n_sites, dtype=float))


def growth_velocity(times, sizes, frac=(0.2, 0.8)):
    """Slope of equivalent side length vs time over the middle part of the run (sites/s)."""
    times = np.asarray(times)
    side = equivalent_side(sizes)
    if len(side) < 10:
        return 0.0
    lo, hi = np.quantile(side, frac)
    sel = (side >= lo) & (side <= hi)
    if sel.sum() < 5 or np.ptp(times[sel]) == 0:
        return 0.0
    return float(np.polyfit(times[sel], side[sel], 1)[0])
