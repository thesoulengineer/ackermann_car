"""
sim/track.py

Closed-loop race-track geometry built from a set of 2D control waypoints.

This is the foundation module of the environment (see docs/INTERFACE.md).
It fits a PERIODIC cubic spline through the waypoints so that position,
tangent, and curvature are all continuous across the start/finish seam,
then resamples the centerline at UNIFORM ARC-LENGTH spacing.

Heading and curvature are computed ANALYTICALLY from the spline derivatives
(not by finite differences). Left-pointing unit normals and the left/right
track boundaries follow from the heading.

All public arrays are numpy arrays of equal length P (one row per centerline
sample). Angles are in radians, using the atan2 convention (measured from the
+x axis, CCW positive).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splprep, splev


class Track:
    """A closed-loop track fitted with a periodic cubic spline.

    Public attributes (numpy arrays of length P unless noted):
        s           (P,)    arc length of each sample, metres, starts at 0
        cx, cy      (P,)    centerline coordinates, metres
        theta       (P,)    heading = atan2(y', x'), radians, wrapped to (-pi, pi]
        kappa       (P,)    signed curvature, 1/m (left/CCW turn positive)
        normals     (P, 2)  left-pointing unit normals (-sin theta, cos theta)
        left_bound  (P, 2)  centerline + half_width * normal
        right_bound (P, 2)  centerline - half_width * normal
        width       float   full track width, metres
        half_width  float   width / 2, metres
        length      float   total centerline length, metres
        ds          float   arc-length sample spacing, metres
        waypoints   (K, 2)  control points the spline was fit through
        tck                 scipy spline representation (periodic)
    """

    def __init__(self, waypoints, width=10.0, ds=0.1, n_dense=20000):
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2:
            raise ValueError("waypoints must have shape (K, 2)")
        # A periodic spline closes the loop itself; drop a duplicated seam point.
        if np.allclose(wp[0], wp[-1]):
            wp = wp[:-1]
        if len(wp) < 4:
            raise ValueError("need at least 4 distinct waypoints for a cubic spline")

        self.waypoints = wp
        self.width = float(width)
        self.half_width = self.width / 2.0
        self.ds = float(ds)

        # Periodic cubic spline through the waypoints (C2-continuous at the seam).
        tck, _ = splprep([wp[:, 0], wp[:, 1]], s=0.0, per=True, k=3)
        self.tck = tck

        # Dense sample to build an arc-length <-> parameter (u) lookup table.
        # Uniform spacing in u is NOT uniform in arc length, so we map between them.
        u_dense = np.linspace(0.0, 1.0, n_dense)
        xd, yd = splev(u_dense, tck)
        seg = np.hypot(np.diff(xd), np.diff(yd))
        s_dense = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(s_dense[-1])

        # Resample at uniform arc length. arange excludes the endpoint, so the
        # seam point is not duplicated and spacing stays ds even across the seam.
        s_samples = np.arange(0.0, self.length, self.ds)
        u_samples = np.interp(s_samples, s_dense, u_dense)

        x, y, dx, dy, ddx, ddy = self._eval(u_samples)
        self.s = s_samples
        self.cx = x
        self.cy = y
        self.theta = np.arctan2(dy, dx)
        self.kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)

        nx = -np.sin(self.theta)
        ny = np.cos(self.theta)
        self.normals = np.column_stack([nx, ny])
        center = np.column_stack([self.cx, self.cy])
        self.left_bound = center + self.half_width * self.normals
        self.right_bound = center - self.half_width * self.normals

    def _eval(self, u):
        """Position and first/second derivatives wrt the spline parameter u.

        Heading and the signed-curvature formula are invariant to the spline
        parameterisation, so evaluating at the arc-length sample u-values is
        exact even though u itself is not arc length.
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        x, y = splev(u, self.tck, der=0)
        dx, dy = splev(u, self.tck, der=1)
        ddx, ddy = splev(u, self.tck, der=2)
        return x, y, dx, dy, ddx, ddy

    def seam_continuity(self):
        """Geometry mismatch across the start/finish seam (spline at u=0 vs u=1).

        For a clean periodic loop these are all ~0; a non-periodic fit would
        typically show a jump in curvature here.
        """
        x, y, dx, dy, ddx, ddy = self._eval([0.0, 1.0])
        theta = np.arctan2(dy, dx)
        kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
        dpos = float(np.hypot(x[1] - x[0], y[1] - y[0]))
        dtheta = float((theta[1] - theta[0] + np.pi) % (2 * np.pi) - np.pi)
        dkappa = float(kappa[1] - kappa[0])
        return {"position": dpos, "heading": dtheta, "curvature": dkappa}

    # ----- built-in presets ---------------------------------------------
    @classmethod
    def oval(cls, a=50.0, b=30.0, n=16, width=10.0, ds=0.1):
        """A simple oval (ellipse), traversed counter-clockwise."""
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        wp = np.column_stack([a * np.cos(t), b * np.sin(t)])
        return cls(wp, width=width, ds=ds)

    @classmethod
    def circuit(cls, width=8.0, ds=0.1):
        """A closed circuit with varying curvature (star-shaped, CCW)."""
        wp = np.array([
            (70.0, 0.0),
            (55.0, 35.0),
            (15.0, 45.0),
            (-20.0, 35.0),
            (-25.0, 10.0),
            (-55.0, 5.0),
            (-70.0, -25.0),
            (-35.0, -45.0),
            (10.0, -40.0),
            (45.0, -30.0),
        ])
        return cls(wp, width=width, ds=ds)

    @classmethod
    def figure_eight(cls, a=40.0, n=24, width=6.0, ds=0.1):
        """A figure-eight (lemniscate of Gerono), one full closed loop.

        x = a*cos t,  y = a*sin t*cos t. Self-intersects once at the origin.
        """
        t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        wp = np.column_stack([a * np.cos(t), a * np.sin(t) * np.cos(t)])
        return cls(wp, width=width, ds=ds)


# ----- visual + numeric sanity check ------------------------------------
def _plot_layout(ax, track, title):
    import numpy as np

    ax.plot(track.cx, track.cy, "-", color="tab:blue", lw=1.6, label="centerline")
    ax.plot(track.left_bound[:, 0], track.left_bound[:, 1], "-",
            color="tab:green", lw=1.0, label="left bound")
    ax.plot(track.right_bound[:, 0], track.right_bound[:, 1], "-",
            color="tab:red", lw=1.0, label="right bound")
    ax.scatter(track.waypoints[:, 0], track.waypoints[:, 1],
               color="k", s=30, zorder=5, label="waypoints")

    # Start/finish point (index 0) + heading arrow.
    extent = max(np.ptp(track.cx), np.ptp(track.cy))
    hlen = max(3.0 * track.half_width, 0.07 * extent)
    ax.scatter([track.cx[0]], [track.cy[0]], color="magenta", s=90,
               zorder=6, marker="o", label="start/finish")
    ax.quiver(track.cx[0], track.cy[0],
              np.cos(track.theta[0]) * hlen, np.sin(track.theta[0]) * hlen,
              angles="xy", scale_units="xy", scale=1, color="magenta",
              width=0.006, zorder=7)

    # A handful of left normals (length = half_width, so they reach the boundary).
    idx = np.linspace(0, len(track.cx), 12, endpoint=False).astype(int)
    ax.quiver(track.cx[idx], track.cy[idx],
              track.normals[idx, 0] * track.half_width,
              track.normals[idx, 1] * track.half_width,
              angles="xy", scale_units="xy", scale=1, color="darkorange",
              width=0.004, zorder=4, label="left normals")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)


def _print_stats(name, track):
    sc = track.seam_continuity()
    print(f"=== {name} ===")
    print(f"  points          : {len(track.cx)}")
    print(f"  total length    : {track.length:8.2f} m")
    print(f"  curvature min   : {track.kappa.min():+8.4f} 1/m")
    print(f"  curvature max   : {track.kappa.max():+8.4f} 1/m")
    print(f"  seam d(position): {sc['position']:.3e} m")
    print(f"  seam d(heading) : {sc['heading']:.3e} rad")
    print(f"  seam d(curv)    : {sc['curvature']:.3e} 1/m")
    print()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    presets = [
        ("Oval", Track.oval()),
        ("Circuit", Track.circuit()),
        ("Figure-eight", Track.figure_eight()),
    ]

    for name, trk in presets:
        _print_stats(name, trk)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, trk) in zip(axes, presets):
        _plot_layout(ax, trk, f"{name} track layout")
    fig.tight_layout()
    plt.show()
