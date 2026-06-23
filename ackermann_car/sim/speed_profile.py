"""
sim/speed_profile.py

Curvature-based reference speed profile for a Track.

The reference speed at each centerline sample is the lateral-acceleration
("friction-circle") cornering limit,

    v_ref[k] = min(v_max, sqrt(a_lat_max / |kappa[k]|)),

clamped to v_max on straights where curvature is ~0. The raw profile is then
lightly smoothed so the controller sees a continuous reference rather than the
slope kinks that appear where a straight meets a corner. Smoothing is periodic
because the track is a closed loop (see the note in `speed_profile`).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def speed_profile(track, v_max=3.0, a_lat_max=2.0, smooth_m=0.5, kappa_eps=1e-6):
    """Reference speed at every centerline sample, derived from curvature.

    Parameters
    ----------
    track : Track
        Provides ``kappa`` (signed curvature, 1/m) and ``ds`` (sample spacing, m).
    v_max : float
        Speed cap on straights / gentle curves, m/s.
    a_lat_max : float
        Maximum lateral (cornering) acceleration, m/s^2.
    smooth_m : float
        Smoothing length in metres (Gaussian sigma). 0 disables smoothing.
    kappa_eps : float
        Floor on |kappa| to avoid division by zero on straights.

    Returns
    -------
    np.ndarray
        ``v_ref``, same length as the centerline and indexed identically.

    Smoothing
    ---------
    Curvature here is analytic (from spline derivatives), so it is already
    continuous; but fitting through sparse waypoints can leave small curvature
    ripple, and the ``min(v_max, .)`` clamp introduces a slope kink at every
    straight/corner transition. Both make ``v_ref`` jumpy, which the controller
    dislikes. A light 1-D Gaussian filter removes them. It is applied with
    ``mode="wrap"`` so smoothing is periodic and the profile stays continuous
    across the start/finish seam. The window is given in metres and converted
    to samples via ``track.ds``, so it is independent of sample resolution.
    Tradeoff: symmetric smoothing slightly relaxes the limit right at an apex,
    so keep ``smooth_m`` small (set it to 0 to get the exact clamp).
    """
    kappa = np.abs(np.asarray(track.kappa, dtype=float))
    # max(|kappa|, eps) makes straights (kappa ~ 0) yield a huge speed that the
    # min() below clamps to v_max, with no division-by-zero.
    v_curve = np.sqrt(a_lat_max / np.maximum(kappa, kappa_eps))
    v_ref = np.minimum(v_max, v_curve)

    if smooth_m and smooth_m > 0.0:
        sigma = smooth_m / track.ds
        v_ref = gaussian_filter1d(v_ref, sigma=sigma, mode="wrap")
        # A weighted mean of values <= v_max stays <= v_max; clamp as a safety net.
        v_ref = np.minimum(v_ref, v_max)
    return v_ref


if __name__ == "__main__":
    try:
        from sim.track import Track
    except ImportError:  # run directly as `python sim/speed_profile.py`
        from track import Track

    presets = [
        ("Oval", Track.oval()),
        ("Circuit", Track.circuit()),
        ("Figure-eight", Track.figure_eight()),
    ]

    print("default params (v_max=3.0, a_lat_max=2.0):")
    for name, trk in presets:
        v = speed_profile(trk)
        print(f"  {name:13s} v_ref min/max = {v.min():.2f} / {v.max():.2f} m/s")
    print("  -> every preset's corners are gentler than the 3 m/s lateral limit,")
    print("     so the profile clamps flat at v_max. Raise v_max to see it bite.\n")

    print("demo params (v_max=8.0, a_lat_max=2.0):")
    for name, trk in presets:
        v = speed_profile(trk, v_max=8.0)
        k = np.abs(trk.kappa)
        i = int(v.argmin())
        print(f"  {name:13s} v_ref min/max = {v.min():.2f} / {v.max():.2f} m/s"
              f"  (slowest at s={trk.s[i]:6.1f} m, |kappa|={k[i]:.3f} 1/m)")
