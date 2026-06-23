"""
sim/lap_manager.py

Lap detection and timing for the closed-loop track (environment module).

The car's progress is tracked by its arc-length position s along the
centerline (which the Track provides via nearest-index lookup). A lap is
completed when s wraps forward across the start/finish seam, i.e. it drops by
more than half the lap length between consecutive updates. A minimum-lap-time
debounce prevents a double count from small backward jitter near the seam.
"""

from __future__ import annotations


class LapManager:
    """Counts laps and tracks lap times from the car's arc-length position.

    Attributes:
        lap              int    current lap number (1-based)
        laps_completed   int    number of completed laps
        current_lap_time float  time elapsed on the current lap, seconds
        last_lap_time    float  duration of the most recent completed lap (or None)
        best_lap_time    float  fastest completed lap (or None)
        lap_times        list   all completed lap times, in order
    """

    def __init__(self, track, min_lap_time=1.0):
        self.length = float(track.length)
        self.min_lap_time = float(min_lap_time)

        self._s_prev = None
        self.lap = 1
        self.laps_completed = 0
        self.lap_start_t = None
        self.current_lap_time = 0.0
        self.last_lap_time = None
        self.best_lap_time = None
        self.lap_times = []

    def update(self, s, t):
        """Advance the lap state with the car's arc-length position s at time t."""
        if self._s_prev is None:  # first sample: start lap 1's clock
            self._s_prev = s
            self.lap_start_t = t
            self.current_lap_time = 0.0
            return

        # A forward wrap across the seam shows up as s dropping by ~one lap.
        wrapped = (self._s_prev - s) > 0.5 * self.length
        if wrapped and (t - self.lap_start_t) >= self.min_lap_time:
            lap_time = t - self.lap_start_t
            self.lap_times.append(lap_time)
            self.last_lap_time = lap_time
            self.best_lap_time = (
                lap_time if self.best_lap_time is None else min(self.best_lap_time, lap_time)
            )
            self.laps_completed += 1
            self.lap += 1
            self.lap_start_t = t

        self._s_prev = s
        self.current_lap_time = t - self.lap_start_t

    def hud_info(self):
        """Dict consumed by the visualizer HUD."""
        return {"lap": self.lap, "lap_time": self.current_lap_time}


if __name__ == "__main__":
    try:
        from .track import Track
    except ImportError:  # run directly as `python sim/lap_manager.py`
        from track import Track

    # Self-test: drive a synthetic car at constant speed for ~2.5 laps and
    # confirm laps are detected with the expected time (length / v).
    track = Track.oval()
    lm = LapManager(track)
    v, dt = 3.0, 0.1
    s, t = 0.0, 0.0
    while lm.laps_completed < 2:
        lm.update(s % track.length, t)
        s += v * dt
        t += dt

    print(f"track length     : {track.length:.2f} m")
    print(f"expected lap time: {track.length / v:.2f} s  (length / {v} m/s)")
    print(f"detected laps    : {lm.laps_completed}")
    print(f"lap times        : {[f'{x:.2f}' for x in lm.lap_times]} s")
    print(f"best lap         : {lm.best_lap_time:.2f} s")
