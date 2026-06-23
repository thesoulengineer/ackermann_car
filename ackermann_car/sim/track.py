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
from scipy.interpolate import splev, splprep


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

    def __init__(self, waypoints, width=10.0, ds=0.1, n_dense=20000, v_max=3.0, a_lat_max=2.0):
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

        # Speed-profile parameters are environment-owned (the controller owns
        # only N and dt). v_ref is computed lazily and cached (see `v_ref`).
        self._speed_kwargs = dict(v_max=v_max, a_lat_max=a_lat_max)
        self._v_ref = None

        # Runtime-query state: the last matched index lets nearest_index search
        # a forward window instead of the whole track; the last dt lets
        # get_boundary_data reproduce the exact horizon get_reference stepped.
        self._last_index = None
        self._last_dt = None

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

    # ----- runtime interface to the controller (see docs/INTERFACE.md) ---
    @property
    def v_ref(self):
        """Per-sample reference speed (m/s), computed once and cached."""
        if self._v_ref is None:
            try:
                from .speed_profile import speed_profile
            except ImportError:  # running as a script with sim/ on the path
                from speed_profile import speed_profile
            self._v_ref = speed_profile(self, **self._speed_kwargs)
        return self._v_ref

    @property
    def last_index(self):
        """Index matched by the most recent get_reference call (None if never)."""
        return self._last_index

    def nearest_index(self, state, last_index=None, window_m=5.0, back_m=0.0):
        """Index of the centerline point closest to the car position state[:2].

        With ``last_index`` given, only a forward arc-length window of
        ``window_m`` (plus an optional ``back_m`` slack) around it is searched,
        wrapping across the start/finish seam. This stops the match from
        sliding backwards or jumping to the far branch on self-overlapping
        sections such as the figure-eight crossing. With ``last_index=None``
        (first call) the whole track is searched to acquire the position.
        """
        px, py = float(state[0]), float(state[1])
        if last_index is None:
            d2 = (self.cx - px) ** 2 + (self.cy - py) ** 2
            return int(np.argmin(d2))

        P = len(self.cx)
        fwd = max(1, int(round(window_m / self.ds)))
        back = max(0, int(round(back_m / self.ds)))
        cand = (last_index + np.arange(-back, fwd + 1)) % P
        d2 = (self.cx[cand] - px) ** 2 + (self.cy[cand] - py) ** 2
        return int(cand[np.argmin(d2)])

    def _horizon_indices(self, i0, N, dt):
        """Indices of the N+1 horizon points, stepping ~v_ref*dt in arc length.

        Deterministic in (i0, N, dt), so get_reference and get_boundary_data
        produce the same horizon. Steps advance at least one sample so the
        horizon always moves forward, and wrap across the seam.
        """
        v = self.v_ref
        P = len(self.cx)
        idxs = np.empty(N + 1, dtype=int)
        cur = int(i0) % P
        idxs[0] = cur
        for n in range(1, N + 1):
            step = max(1, int(round(v[cur] * dt / self.ds)))
            cur = (cur + step) % P
            idxs[n] = cur
        return idxs

    def get_reference(self, car_state, N, dt, last_index=None):
        """Reference array (N+1, 4) for the controller: [x, y, v_ref, theta].

        Row 0 is the centerline point nearest the car; each subsequent row
        steps ~v_ref*dt forward in arc length, so lookahead is a consistent
        physical distance. The theta column is unwrapped LOCALLY against the
        car's current heading (car_state[3]): np.unwrap removes ±pi jumps
        across the horizon, then the whole column is shifted by a multiple of
        2*pi so row 0 sits within pi of the car heading. Nothing global is
        stored, so heading never drifts by 2*pi as laps accumulate.

        The matched start index and dt are remembered so get_boundary_data can
        reproduce the same horizon.
        """
        if last_index is None:
            last_index = self._last_index
        i0 = self.nearest_index(car_state, last_index)
        self._last_index = i0
        self._last_dt = dt

        idxs = self._horizon_indices(i0, N, dt)
        theta = np.unwrap(self.theta[idxs])
        if len(car_state) >= 4:
            car_theta = float(car_state[3])
            theta = theta - round((theta[0] - car_theta) / (2 * np.pi)) * 2 * np.pi

        return np.column_stack([self.cx[idxs], self.cy[idxs], self.v_ref[idxs], theta])

    def get_boundary_data(self, index, N, dt=None):
        """Left-pointing unit normals (N+1, 2) over the horizon, and half_width.

        Uses the same stepping as get_reference, so the normals align row-for-row
        with the reference points for the controller's wall constraints. ``dt``
        defaults to the dt of the most recent get_reference call; pass it
        explicitly if calling standalone.
        """
        if dt is None:
            dt = self._last_dt
        if dt is None:
            raise ValueError("no dt available: call get_reference first or pass dt explicitly")
        idxs = self._horizon_indices(index, N, dt)
        return self.normals[idxs], self.half_width

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
        wp = np.array(
            [
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
            ]
        )
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
    ax.plot(
        track.left_bound[:, 0],
        track.left_bound[:, 1],
        "-",
        color="tab:green",
        lw=1.0,
        label="left bound",
    )
    ax.plot(
        track.right_bound[:, 0],
        track.right_bound[:, 1],
        "-",
        color="tab:red",
        lw=1.0,
        label="right bound",
    )
    ax.scatter(
        track.waypoints[:, 0], track.waypoints[:, 1], color="k", s=30, zorder=5, label="waypoints"
    )

    # Start/finish point (index 0) + heading arrow.
    extent = max(np.ptp(track.cx), np.ptp(track.cy))
    hlen = max(3.0 * track.half_width, 0.07 * extent)
    ax.scatter(
        [track.cx[0]],
        [track.cy[0]],
        color="magenta",
        s=90,
        zorder=6,
        marker="o",
        label="start/finish",
    )
    ax.quiver(
        track.cx[0],
        track.cy[0],
        np.cos(track.theta[0]) * hlen,
        np.sin(track.theta[0]) * hlen,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="magenta",
        width=0.006,
        zorder=7,
    )

    # A handful of left normals (length = half_width, so they reach the boundary).
    idx = np.linspace(0, len(track.cx), 12, endpoint=False).astype(int)
    ax.quiver(
        track.cx[idx],
        track.cy[idx],
        track.normals[idx, 0] * track.half_width,
        track.normals[idx, 1] * track.half_width,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="darkorange",
        width=0.004,
        zorder=4,
        label="left normals",
    )

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

    # ----- runtime reference test: fake car on the oval, near the heading
    # wrap point (where atan2 flips +/-pi), so the heading column would jump
    # without local unwrapping.
    oval = Track.oval()
    jw = int(np.argmax(np.abs(oval.theta)))  # sample where theta ~ +/-pi
    j = (jw - 12) % len(oval.cx)  # start a little before it
    car_theta = float(oval.theta[j])
    car_state = np.array(
        [
            oval.cx[j] + 0.3 * oval.normals[j, 0],  # nudge off the centerline
            oval.cy[j] + 0.3 * oval.normals[j, 1],
            2.0,  # speed (unused by lookup)
            car_theta,
        ]
    )

    N, dt = 25, 0.2
    ref = oval.get_reference(car_state, N, dt)
    normals, half_width = oval.get_boundary_data(oval.last_index, N)
    wrapped = (ref[:, 3] + np.pi) % (2 * np.pi) - np.pi  # what raw atan2 would give

    print("=== get_reference test (oval, crossing the heading wrap) ===")
    print(f"  car heading        : {car_theta:+.3f} rad")
    print(f"  reference shape     : {ref.shape}   (expected ({N + 1}, 4))")
    print(f"  boundary normals    : {normals.shape}, half_width = {half_width:.2f} m")
    print("  heading column (rad):")
    print("   ", np.array2string(ref[:, 3], precision=3, separator=", "))
    print(f"  max |d.heading| raw wrapped (no unwrap): {np.abs(np.diff(wrapped)).max():.3f} rad")
    print(f"  max |d.heading| ours (local unwrap)    : {np.abs(np.diff(ref[:, 3])).max():.3f} rad")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, trk) in zip(axes, presets):
        _plot_layout(ax, trk, f"{name} track layout")
    fig.tight_layout()
    plt.show()
