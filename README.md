# ackermann_car

MPC path-tracking controller for an Ackermann-steering car on a racetrack,
simulated in MuJoCo. University robotics project — three-way team split.

## Repository layout

```
ackermann_car/
├── assets/         # MuJoCo model files: scene XML, meshes, track geometry
├── controllers/    # MPC solver (CVXPY-based) — Jamiu
├── docs/           # Diagrams, writeup, figures
└── sim/            # Simulated world — Katlego, David
    ├── track.py        # Centerline, boundaries, headings, curvature, arc length
    ├── speed_profile.py # Reference speed derived from curvature
    ├── lap_manager.py  # Lap detection and timing
    └── visualizer.py   # Matplotlib visualisation helpers
```

## Responsibility split

| Area | Owner | Modules |
|------|-------|---------|
| Environment | Katlego | `sim/track.py`, `sim/speed_profile.py`, `sim/lap_manager.py`, `sim/visualizer.py` |
| Vehicle dynamics / integrator | David | `sim/car.py` (TBD) |
| MPC solver | Jamiu | `controllers/` |

## Setup

```bash
pip install -r requirements.txt
```
