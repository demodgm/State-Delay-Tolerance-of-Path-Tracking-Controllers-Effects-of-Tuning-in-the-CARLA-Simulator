"""Generate a synthetic reference route so the pipeline can be exercised
without CARLA. The real study uses a route recorded by driving Town03
manually; this file exists only so `--selftest` has something to track."""
import numpy as np, os

def route():
    segs, xy, yaw, p = [], [], 0.0, np.array([0.0, 0.0])
    plan = [("s", 60), ("l", 40), ("s", 40), ("r", 40), ("s", 50), ("l", 30), ("s", 40)]
    for kind, n in plan:
        for _ in range(n):
            if kind == "l": yaw += np.deg2rad(2.0)
            elif kind == "r": yaw -= np.deg2rad(2.0)
            p = p + np.array([np.cos(yaw), np.sin(yaw)])
            xy.append(p.copy()); segs.append(kind)
    xy = np.array(xy)
    v = np.where(np.array(segs) == "s", 11.0, 7.0)
    k = np.ones(9) / 9
    v = np.convolve(np.pad(v, 4, mode="edge"), k, mode="valid")
    return xy, v

if __name__ == "__main__":
    xy, v = route()
    os.makedirs("data", exist_ok=True)
    out = "data/selftest_reference.txt"
    np.savetxt(out, np.column_stack([xy, v]), delimiter=",", fmt="%.6f")
    print(f"wrote {out}: {len(xy)} points, {np.sum(np.hypot(*np.diff(xy,axis=0).T)):.0f} m")
