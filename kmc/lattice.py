"""
Triangular lattice of Mo sites.

Picture a sheet of tightly packed marbles: every site has 6 neighbours,
60 degrees apart. We store the sheet as an L x L array using "offset rows":
odd rows are shifted half a spacing to the right, which turns a square array
into a triangular lattice.

    row 2:  o   o   o   o
    row 1:    o   o   o   o      <- shifted by 1/2
    row 0:  o   o   o   o

Neighbour directions are always listed in the same counter-clockwise order:

    k = 0 : 0 deg   (right)
    k = 1 : 60 deg  (up-right)
    k = 2 : 120 deg (up-left)
    k = 3 : 180 deg (left)
    k = 4 : 240 deg (down-left)
    k = 5 : 300 deg (down-right)

Fixing this order matters: it is how we tell the two edge types apart
(see `triangle_types` below).

Boundaries are periodic, but the simulation stops before the flake gets near
the border, so the wrap-around never affects the results.
"""
import numpy as np

SQRT3 = np.sqrt(3.0)

# (di, dj) for each direction k, for even rows and for odd rows.
_EVEN = [(1, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]
_ODD = [(1, 0), (1, 1), (0, 1), (-1, 0), (0, -1), (1, -1)]


class TriangularLattice:
    def __init__(self, L=100, margin=3):
        if L % 2:
            raise ValueError("L must be even for periodic offset rows")
        self.L = L
        self.N = L * L

        # nbr[s] = tuple of the 6 neighbour indices of site s, in order k=0..5
        nbr = []
        for j in range(L):
            offs = _ODD if (j & 1) else _EVEN
            for i in range(L):
                nbr.append(tuple(((j + dj) % L) * L + (i + di) % L for di, dj in offs))
        self.nbr = nbr

        # Cartesian positions (lattice spacing = 1) for plotting and geometry.
        jj, ii = np.divmod(np.arange(self.N), L)
        self.x = ii + 0.5 * (jj & 1)
        self.y = jj * SQRT3 / 2.0

        # Sites close to the border: if the flake reaches one, we stop the run.
        self.border = (ii < margin) | (ii >= L - margin) | (jj < margin) | (jj >= L - margin)

    def index(self, i, j):
        return (j % self.L) * self.L + (i % self.L)

    def center(self):
        return self.index(self.L // 2, self.L // 2)

    def hex_cluster(self, site, radius):
        """All sites within `radius` hops of `site` (a small hexagon). radius=0 -> one site."""
        cluster = {site}
        frontier = {site}
        for _ in range(radius):
            new = set()
            for s in frontier:
                new.update(self.nbr[s])
            new -= cluster
            cluster |= new
            frontier = new
        return sorted(cluster)

    def site_at_offset(self, site, k, steps):
        """Walk `steps` hops from `site` along direction k."""
        for _ in range(steps):
            site = self.nbr[site][k]
        return site


def triangle_types(conv, nb):
    """
    Count how many complete unit triangles an oxide site would close with the flake.

    The 6 triangles around a site are formed by consecutive neighbour pairs
    (k, k+1). They alternate between two orientations:

        k even -> "up" triangle   (Delta)
        k odd  -> "down" triangle (nabla)

    In 1H-MoS2 the sulfur columns sit over only ONE of these two triangle
    sets (we pick the Delta set). This is the root of the two edge types:

      * Joining an S-terminated zigzag edge closes a Delta triangle, so
        the bridging S atom is already in place.
      * Joining a Mo-terminated zigzag edge closes a nabla triangle, so
        there is no S there and the new Mo has to be sulfided from scratch.

    Edges facing opposite directions are always of opposite type, so as you go
    around a hexagon the edges alternate S, Mo, S, Mo, S, Mo. Nothing here
    mentions triangles as a shape; the shape comes out on its own.

    Returns (n_delta, n_nabla).
    """
    n_d = n_n = 0
    for k in range(6):
        if conv[nb[k]] and conv[nb[(k + 1) % 6]]:
            if k & 1:
                n_n += 1
            else:
                n_d += 1
    return n_d, n_n
