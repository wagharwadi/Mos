"""
The kinetic Monte Carlo engine (BKL / Gillespie "direct method").

Each step of the loop:

  1. Every possible event has a rate. Their sum is R.
  2. Choose ONE event, with probability (its rate) / R.
     Implementation: each site keeps the sum of its own event rates in a
     binary "sum tree". We walk down the tree with a random number, which
     finds the right site in log2(N) ~ 14 steps rather than scanning 10,000 sites.
  3. Advance the clock by dt = -ln(u) / R, with u uniform in (0, 1].
     When many events are possible, R is large and dt is small, and vice versa.
     There is no fixed time step.
  4. Carry out the event, then update the rates of the few sites it affected.

Which sites do we simulate?
  Only oxide sites that TOUCH the flake (at least one MoS2 neighbour) can
  convert, so only their sulfur matters. Sulfur on a site far from the flake
  just flips between the gas and the surface on its own and never interacts
  with anything. We start every site in that steady state (each of the S_max
  slots filled with probability theta0). The moment a site first touches the
  flake, we draw its sulfur from that same distribution. This is exact, not an
  approximation: an independent process started in steady state stays in
  steady state. It also means we skip millions of pointless far-away events.
"""
import math
import random

import numpy as np

from .lattice import TriangularLattice, triangle_types
from .params import Params


class SumTree:
    """A binary tree whose leaves are per-site rates; each parent holds the sum of its children."""

    def __init__(self, n):
        size = 1
        while size < n:
            size *= 2
        self.size = size
        self.t = [0.0] * (2 * size)

    def set(self, i, value):
        t = self.t
        p = i + self.size
        t[p] = value
        p >>= 1
        while p:
            t[p] = t[2 * p] + t[2 * p + 1]
            p >>= 1

    def total(self):
        return self.t[1]

    def find(self, u):
        """Return the leaf index i where the running sum of rates crosses u."""
        t = self.t
        p = 1
        size = self.size
        while p < size:
            left = t[2 * p]
            if u < left:
                p = 2 * p
            else:
                u -= left
                p = 2 * p + 1
        return p - size


# Event codes, used for the bookkeeping counters
ADS, DES, CONV = 0, 1, 2
EVENT_NAMES = ("adsorption", "desorption", "conversion")


class Simulation:
    def __init__(self, params: Params, nuclei=None, seed=None, lattice=None):
        self.p = params
        self.lat = lattice or TriangularLattice(params.L)
        self.rng = random.Random(seed)
        N = self.lat.N

        self.conv = [False] * N          # True once a site is MoS2
        self.s = [0] * N                 # S atoms on each tracked oxide site
        self.ncon = [0] * N              # number of MoS2 neighbours
        self.rates = [(0.0, 0.0, 0.0)] * N   # (ads, des, conv) for each site
        self.tree = SumTree(N)
        self.conv_time = np.full(N, np.inf)  # when each site converted (for movies)

        # Lookup tables so the inner loop never calls exp()
        p = params
        self.a = p.k_ads * p.P                                    # per empty slot
        self.d = [p.des_rate(n) for n in range(7)]               # per S atom, by n
        self.c = [[p.conv_rate(i, j) for j in range(4)] for i in range(4)]
        self.theta0 = p.coverage(0)                               # far-field S coverage

        # Bookkeeping: counts[event][n] = how often each event fired at sites with n neighbours
        self.counts = np.zeros((3, 7), dtype=np.int64)
        self.t = 0.0
        self.n_events = 0
        self.n_conv = 0
        self.history_t = [0.0]           # flake size N(t), recorded at each conversion
        self.history_n = [0]
        self.stop_reason = None

        for site in (nuclei if nuclei is not None else [self.lat.center()]):
            self._convert(site)
        self.history_n[0] = self.n_conv

    # ------------------------------------------------------------------ helpers
    def _draw_s(self):
        """Sulfur on a site meeting the flake for the first time (far-field steady state)."""
        r = self.rng.random
        th = self.theta0
        return sum(1 for _ in range(self.p.S_max) if r() < th)

    def _refresh(self, i):
        """Recompute the three event rates of site i and store their sum in the tree."""
        if self.conv[i] or self.ncon[i] == 0:
            if self.rates[i] != (0.0, 0.0, 0.0):
                self.rates[i] = (0.0, 0.0, 0.0)
                self.tree.set(i, 0.0)
            return
        s = self.s[i]
        ra = (self.p.S_max - s) * self.a
        rd = s * self.d[self.ncon[i]]
        if s >= self.p.S_req:
            nd, nn = triangle_types(self.conv, self.lat.nbr[i])
            rc = self.c[min(nd, 3)][min(nn, 3)]
        else:
            rc = 0.0
        self.rates[i] = (ra, rd, rc)
        self.tree.set(i, ra + rd + rc)

    def _convert(self, i):
        self.conv[i] = True
        self.conv_time[i] = self.t
        self.n_conv += 1
        self._refresh(i)  # its own rates drop to zero
        for j in self.lat.nbr[i]:
            if self.conv[j]:
                continue
            if self.ncon[j] == 0:            # j just started touching the flake
                self.s[j] = self._draw_s()
            self.ncon[j] += 1
            self._refresh(j)
        if self.lat.border[i]:
            self.stop_reason = "reached border"

    # ---------------------------------------------------------------- main loop
    def run(self, t_max=None, max_events=None, callback=None):
        t_max = self.p.t_max if t_max is None else t_max
        max_events = self.p.max_events if max_events is None else max_events
        rnd = self.rng.random
        tree = self.tree
        rates = self.rates
        s = self.s
        ncon = self.ncon
        counts = self.counts
        log = math.log

        while self.stop_reason is None:
            R = tree.total()
            if R <= 0.0:
                self.stop_reason = "no events possible"
                break
            # (3) random time step
            dt = -log(1.0 - rnd()) / R
            if self.t + dt > t_max:
                self.t = t_max
                self.stop_reason = "time limit"
                break
            self.t += dt

            # (2) choose the site, then the event at that site
            i = tree.find(rnd() * R)
            ra, rd, rc = rates[i]
            u = rnd() * (ra + rd + rc)
            n = ncon[i]
            if u < ra:
                s[i] += 1
                counts[ADS, n] += 1
                self._refresh(i)
            elif u < ra + rd:
                s[i] -= 1
                counts[DES, n] += 1
                self._refresh(i)
            else:
                counts[CONV, n] += 1
                self._convert(i)
                self.history_t.append(self.t)
                self.history_n.append(self.n_conv)
                if callback is not None:
                    callback(self)

            self.n_events += 1
            if self.n_events >= max_events:
                self.stop_reason = "event limit"
        return self

    # ----------------------------------------------------------------- results
    def mask(self, t=None):
        """Boolean array of converted sites (at time t, if given)."""
        if t is None:
            return np.array(self.conv)
        return self.conv_time <= t

    def summary(self):
        return {
            "P": self.p.P,
            "size": self.n_conv,
            "time": self.t,
            "events": self.n_events,
            "stop": self.stop_reason,
        }
