"""
Model parameters, and why each value was chosen.

All rates use the Arrhenius form   rate = nu * exp(-E / kT).
In words: every +0.086 eV of barrier at 1000 K makes the event e (~2.7)
times rarer.

The three events at an oxide site touching the flake are:

  1. S ADSORPTION    rate = k_ads * P * (empty S slots)
  2. S DESORPTION    rate = (S atoms present) * nu * exp(-(E_des + n * E_bind) / kT)
                     where n is the number of converted (MoS2) neighbours.
                     Each MoS2 neighbour binds S more strongly, so S leaves more slowly.
  3. CONVERSION      MoOx -> MoS2, allowed only once the site holds S_req S atoms:
                     rate = nu * exp(-(E_conv - n_delta*eps_S - n_nabla*eps_Mo) / kT)
                     n_delta and n_nabla count the unit triangles the new site
                     would close with the flake (see lattice.triangle_types).
"""
from dataclasses import dataclass, asdict, replace
import numpy as np

KB = 8.617333e-5  # Boltzmann constant, eV/K


@dataclass(frozen=True)
class Params:
    # --- conditions -------------------------------------------------------
    T: float = 1000.0          # K. CVD growth of MoS2 typically runs at 650-850 C (~920-1120 K).
    P: float = 10.0            # Pa, sulfur partial pressure (the knob we turn).

    # --- sulfur supply ----------------------------------------------------
    # Hertz-Knudsen flux of S2 at 1000 K onto one MoS2 site (8.6 A^2) is about
    # 900 hits per second per Pa. A sticking coefficient of ~0.1 gives ~100 /Pa/s.
    k_ads: float = 100.0       # adsorption rate per empty S slot, per Pa, per s
    S_max: int = 2             # S slots per Mo site (MoS2 has 2 S per Mo)
    S_req: int = 2             # S atoms needed before a site can convert

    # --- sulfur desorption ------------------------------------------------
    nu: float = 1e13           # attempt frequency, s^-1 (typical lattice vibration)
    E_des: float = 1.75        # eV. S on bare oxide is weakly held: it stays only ~65 us
                               # at 1000 K. This value puts the growth/no-growth threshold
                               # near 0.1-1 Pa, a realistic S partial pressure in CVD.
    E_bind: float = 0.25       # eV per converted neighbour (roughly one extra weak S-Mo
                               # bond). Each MoS2 neighbour slows desorption ~18x, so an
                               # edge site holds S far better than an isolated one. This
                               # is what makes small flakes (few neighbours) fragile.

    # --- conversion (sulfidation of the Mo site) -----------------------------
    E_conv: float = 2.60       # eV, base barrier (no triangle closed): ~1 s^-1
    eps_S: float = 0.30        # eV of barrier lowering per Delta triangle closed (S-edge).
                               # The bridging S already sits there, so joining is easy.
    eps_Mo: float = 0.15       # eV per nabla triangle closed (Mo-edge); no S pre-placed.
                               # The 0.15 eV difference makes S-edges ~5.7x faster.
    E_min: float = 0.5         # floor so a barrier can never go unphysically low

    # --- run control --------------------------------------------------------
    L: int = 100               # lattice is L x L sites
    t_max: float = 200.0       # s, length of the growth window ("CVD run time")
    max_events: int = 20_000_000

    @property
    def kT(self):
        return KB * self.T

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)

    # Handy derived quantities used in the analysis -----------------------------
    def des_rate(self, n):
        """Desorption rate of ONE S atom at a site with n MoS2 neighbours."""
        return self.nu * np.exp(-(self.E_des + n * self.E_bind) / self.kT)

    def conv_rate(self, n_delta, n_nabla):
        E = max(self.E_min, self.E_conv - n_delta * self.eps_S - n_nabla * self.eps_Mo)
        return self.nu * np.exp(-E / self.kT)

    def coverage(self, n, P=None):
        """Fraction of S slots filled at steady state (Langmuir): theta = aP / (aP + d)."""
        P = self.P if P is None else P
        a = self.k_ads * P
        return a / (a + self.des_rate(n))
