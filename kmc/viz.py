"""Drawing flakes and making growth movies."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

OXIDE = "#d9d4c7"


def _hex_size(ax, span):
    """Marker area (pt^2) so neighbouring hexagons just touch, for `span` lattice units across."""
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    side_in = min(bbox.width, bbox.height)
    pts_per_unit = side_in * 72.0 / span
    return (pts_per_unit * 1.12) ** 2


def growth_order(conv_time):
    """For each converted site: the fraction of the final flake that existed when it converted (0..1)."""
    order = np.full(conv_time.shape, np.nan)
    done = np.isfinite(conv_time)
    idx = np.flatnonzero(done)
    ranks = np.argsort(np.argsort(conv_time[idx], kind="stable"), kind="stable")
    order[idx] = (ranks + 1) / len(idx)
    return order


BANDS = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 10)))


def draw_flake(ax, lat, conv_time, t=None, crop=None, show_oxide=True):
    """
    Draw MoS2 sites coloured in 10 bands by the ORDER in which they converted
    (band 1 = first 10% of the final flake, ...). The band borders are the flake's
    outline at earlier moments, so you can see the shape at every stage.
    Oxide sites are beige.
    """
    t = np.inf if t is None else t
    done = conv_time <= t
    order = growth_order(conv_time)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if crop is not None:
        (x0, x1), (y0, y1) = crop
    else:
        x0, x1, y0, y1 = lat.x.min() - 1, lat.x.max() + 1, lat.y.min() - 1, lat.y.max() + 1
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    s = _hex_size(ax, max(x1 - x0, y1 - y0))
    if show_oxide:
        ax.scatter(lat.x[~done], lat.y[~done], s=s, marker="h", c=OXIDE, linewidths=0)
    sc = ax.scatter(lat.x[done], lat.y[done], s=s, marker="h", c=order[done], cmap=BANDS,
                    vmin=0, vmax=1, linewidths=0)
    return sc


def crop_box(lat, conv_time, pad=4):
    done = np.isfinite(conv_time)
    xs, ys = lat.x[done], lat.y[done]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def growth_movie(lat, conv_time, path, n_frames=45, title="", fps=10, crop=None):
    """
    Animated GIF. Frames are spaced evenly in flake SIZE rather than time, so the
    slow start of a low-pressure run doesn't fill the whole movie. The clock is shown
    in the title.
    """
    finite = np.sort(conv_time[np.isfinite(conv_time)])
    idx = np.unique(np.linspace(0, len(finite) - 1, n_frames).astype(int))
    times = list(finite[idx]) + [finite[-1]] * 6   # hold the last frame

    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    fig.subplots_adjust(0, 0, 1, 0.92)

    def frame(k):
        ax.clear()
        draw_flake(ax, lat, conv_time, t=times[k], crop=crop)
        n = int((conv_time <= times[k]).sum())
        ax.set_title(f"{title}\nt = {times[k]:.2f} s   size = {n} Mo sites", fontsize=10)

    anim = animation.FuncAnimation(fig, frame, frames=len(times))
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
