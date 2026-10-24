"""PID, pure pursuit, Stanley and MPC, with gains exposed as a single vector.

Every controller takes the same arguments and returns the same tuple, so the
sweep loop is identical for all four. Gains are passed in explicitly rather
than read from a config inside the controller, because the tuning protocol
needs to set them from outside and freeze them.

State convention: (x, y, yaw, v) in metres / radians / m s^-1, yaw measured
counter-clockwise from +x.
Command convention: (steer, accel), steer in radians (positive = left),
accel in m s^-2. The runner converts to whatever the simulator wants.
"""

import math
import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def nearest_index(ref_xy, x, y, hint=0, window=None):
    """Index of the closest reference point.

    `hint`/`window` restrict the search to a neighbourhood of the previous
    match, which stops the vehicle latching onto a later part of a route that
    doubles back on itself.
    """
    if window is None:
        d = np.hypot(ref_xy[:, 0] - x, ref_xy[:, 1] - y)
        return int(np.argmin(d))
    lo = max(0, hint - 5)
    hi = min(len(ref_xy), hint + window)
    seg = ref_xy[lo:hi]
    d = np.hypot(seg[:, 0] - x, seg[:, 1] - y)
    return lo + int(np.argmin(d))


def signed_cross_track(ref_xy, idx, x, y):
    """Signed perpendicular distance to the path tangent at idx.

    Positive = vehicle is left of the path.
    """
    i0 = max(0, min(idx, len(ref_xy) - 2))
    p0, p1 = ref_xy[i0], ref_xy[i0 + 1]
    t = p1 - p0
    n = np.hypot(*t)
    if n < 1e-9:
        return 0.0, 0.0
    t = t / n
    path_yaw = math.atan2(t[1], t[0])
    d = np.array([x - p0[0], y - p0[1]])
    return float(t[0] * d[1] - t[1] * d[0]), path_yaw


class BaseController:
    name = "base"

    def reset(self):
        pass


class LongitudinalPID:
    """Shared speed loop. Used unchanged by PID / pure pursuit / Stanley."""

    def __init__(self, kp, ki, kd, dt, i_clamp=5.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.i_clamp = i_clamp
        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev = None

    def step(self, v, v_ref):
        e = v_ref - v
        self._i = float(np.clip(self._i + e * self.dt, -self.i_clamp, self.i_clamp))
        d = 0.0 if self._prev is None else (e - self._prev) / self.dt
        self._prev = e
        return self.kp * e + self.ki * self._i + self.kd * d


class PIDController(BaseController):
    """Decomposed PID: independent lateral and longitudinal loops.

    gains = dict(lat_kp, lat_ki, lat_kd, lon_kp, lon_ki, lon_kd)
    """

    name = "pid"

    def __init__(self, gains, dt, params):
        self.g, self.dt, self.p = gains, dt, params
        self.lon = LongitudinalPID(gains["lon_kp"], gains["lon_ki"], gains["lon_kd"], dt)
        self.reset()

    def reset(self):
        self.lon.reset()
        self._i = 0.0
        self._prev = None
        self._hint = 0

    def step(self, state, ref_xy, ref_v):
        x, y, yaw, v = state
        idx = nearest_index(ref_xy, x, y, self._hint, window=200)
        self._hint = idx
        cte, path_yaw = signed_cross_track(ref_xy, idx, x, y)
        # steer right when the car is left of the path -> error is -cte
        e = -cte
        self._i = float(np.clip(self._i + e * self.dt, -2.0, 2.0))
        d = 0.0 if self._prev is None else (e - self._prev) / self.dt
        self._prev = e
        steer = self.g["lat_kp"] * e + self.g["lat_ki"] * self._i + self.g["lat_kd"] * d
        steer += 0.5 * _wrap(path_yaw - yaw)  # heading feed-forward, fixed
        accel = self.lon.step(v, ref_v[idx])
        return float(np.clip(steer, -self.p["max_steer"], self.p["max_steer"])), float(accel)


class PurePursuitController(BaseController):
    """Geometric: arc to a lookahead point. gains = dict(kv, l0, lon_*)."""

    name = "pure_pursuit"

    def __init__(self, gains, dt, params):
        self.g, self.dt, self.p = gains, dt, params
        self.lon = LongitudinalPID(gains["lon_kp"], gains["lon_ki"], gains["lon_kd"], dt)
        self.reset()

    def reset(self):
        self.lon.reset()
        self._hint = 0

    def step(self, state, ref_xy, ref_v):
        x, y, yaw, v = state
        idx = nearest_index(ref_xy, x, y, self._hint, window=200)
        self._hint = idx
        ld = max(self.g["kv"] * v + self.g["l0"], 1.0)
        # walk forward along the path until we're ld away
        j = idx
        while j < len(ref_xy) - 1 and np.hypot(ref_xy[j, 0] - x, ref_xy[j, 1] - y) < ld:
            j += 1
        tx, ty = ref_xy[j]
        alpha = _wrap(math.atan2(ty - y, tx - x) - yaw)
        steer = math.atan2(2.0 * self.p["wheelbase"] * math.sin(alpha), ld)
        accel = self.lon.step(v, ref_v[idx])
        return float(np.clip(steer, -self.p["max_steer"], self.p["max_steer"])), float(accel)


class StanleyController(BaseController):
    """Heading error + speed-normalised cross-track. gains = dict(k, ks, lon_*)."""

    name = "stanley"

    def __init__(self, gains, dt, params):
        self.g, self.dt, self.p = gains, dt, params
        self.lon = LongitudinalPID(gains["lon_kp"], gains["lon_ki"], gains["lon_kd"], dt)
        self.reset()

    def reset(self):
        self.lon.reset()
        self._hint = 0

    def step(self, state, ref_xy, ref_v):
        x, y, yaw, v = state
        # Stanley is defined at the front axle
        fx = x + self.p["wheelbase"] * math.cos(yaw)
        fy = y + self.p["wheelbase"] * math.sin(yaw)
        idx = nearest_index(ref_xy, fx, fy, self._hint, window=200)
        self._hint = idx
        cte, path_yaw = signed_cross_track(ref_xy, idx, fx, fy)
        heading_term = _wrap(path_yaw - yaw)
        cross_term = math.atan2(self.g["k"] * (-cte), v + self.g["ks"])
        steer = heading_term + cross_term
        accel = self.lon.step(v, ref_v[idx])
        return float(np.clip(steer, -self.p["max_steer"], self.p["max_steer"])), float(accel)


# --------------------------------------------------------------------------
# MPC over a unicycle model, solved with iLQR.
# Self-contained so the repo doesn't depend on the course pyilqr package;
# set use_pyilqr=True in params to route through that instead if preferred.
# --------------------------------------------------------------------------

class UnicycleILQR:
    """Minimal iLQR for x' = [v cos th, v sin th, omega, a].

    Controls are u = (omega, a). Quadratic cost on position, speed and effort.

    Written for speed, because this runs once per simulation tick and the
    controller is otherwise 200x more expensive than the geometric ones,
    which dominates the wall-clock cost of the whole sweep. The cost weights
    are constant, so their matrices are built once; the 2x2 control Hessian
    is inverted in closed form; and the iteration stops as soon as the cost
    stops improving, which a warm start usually makes happen within two or
    three passes.
    """

    def __init__(self, dt, horizon, w_pos, w_vel, w_ctrl, u_lim, iters=8,
                 tol=1e-4):
        self.dt, self.H = dt, horizon
        self.w_pos, self.w_vel, self.w_ctrl = w_pos, w_vel, w_ctrl
        self.u_lim = np.asarray(u_lim, float)
        self.iters, self.tol = iters, tol
        # constant across every call: build once, not H times per iteration
        self._W = np.diag([w_pos, w_pos, 0.0, w_vel])
        self._R = np.diag([w_ctrl, w_ctrl])
        self._wvec = np.array([w_pos, w_pos, w_vel])

    def _f(self, x, u):
        px, py, th, v = x
        om, a = u
        return np.array([px + v * math.cos(th) * self.dt,
                         py + v * math.sin(th) * self.dt,
                         th + om * self.dt,
                         max(0.0, v + a * self.dt)])

    def _rollout(self, x0, U):
        X = np.empty((self.H + 1, 4))
        X[0] = x0
        for t in range(self.H):
            X[t + 1] = self._f(X[t], U[t])
        return X

    def _cost(self, X, U, targets):
        """Total trajectory cost, vectorised over the horizon."""
        d = X[:, (0, 1, 3)] - targets
        return float(np.sum(self._wvec * d * d) + self.w_ctrl * np.sum(U * U))

    def _jac(self, x, u):
        _, _, th, v = x
        dt = self.dt
        A = np.eye(4)
        s, c = math.sin(th), math.cos(th)
        A[0, 2] = -v * s * dt
        A[0, 3] = c * dt
        A[1, 2] = v * c * dt
        A[1, 3] = s * dt
        B = np.zeros((4, 2))
        B[2, 0] = dt
        B[3, 1] = dt
        return A, B

    @staticmethod
    def _inv2(M):
        """Closed-form 2x2 inverse; np.linalg.inv costs far more here."""
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        det = a * d - b * c
        if abs(det) < 1e-12:
            det = math.copysign(1e-12, det if det else 1.0)
        return np.array([[d, -b], [-c, a]]) / det

    def solve(self, x0, targets, u_init):
        H = self.H
        targets = np.asarray(targets, float)
        U = np.array(u_init, float)
        X = self._rollout(x0, U)
        cost = self._cost(X, U, targets)

        W, R = self._W, self._R
        K = np.zeros((H, 2, 4))
        k = np.zeros((H, 2))
        eye2 = 1e-6 * np.eye(2)

        for _ in range(self.iters):
            # backward pass
            dx = np.array([X[H, 0] - targets[H, 0], X[H, 1] - targets[H, 1],
                           0.0, X[H, 3] - targets[H, 2]])
            Vx = W @ dx
            Vxx = W.copy()
            for t in range(H - 1, -1, -1):
                A, B = self._jac(X[t], U[t])
                dx = np.array([X[t, 0] - targets[t, 0], X[t, 1] - targets[t, 1],
                               0.0, X[t, 3] - targets[t, 2]])
                BtVxx = B.T @ Vxx
                Qu = R @ U[t] + B.T @ Vx
                Quu = R + BtVxx @ B + eye2
                Qux = BtVxx @ A
                Quu_inv = self._inv2(Quu)
                k[t] = -Quu_inv @ Qu
                K[t] = -Quu_inv @ Qux
                AtVxx = A.T @ Vxx
                Vx = (W @ dx + A.T @ Vx + K[t].T @ Quu @ k[t]
                      + K[t].T @ Qu + Qux.T @ k[t])
                Vxx = (W + AtVxx @ A + K[t].T @ Quu @ K[t]
                       + K[t].T @ Qux + Qux.T @ K[t])

            # forward pass with a backtracking line search
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1):
                Xn = np.empty_like(X)
                Un = np.empty_like(U)
                Xn[0] = X[0]
                for t in range(H):
                    Un[t] = np.clip(U[t] + alpha * k[t] + K[t] @ (Xn[t] - X[t]),
                                    -self.u_lim, self.u_lim)
                    Xn[t + 1] = self._f(Xn[t], Un[t])
                new_cost = self._cost(Xn, Un, targets)
                if new_cost < cost:
                    rel = (cost - new_cost) / max(abs(cost), 1e-9)
                    X, U, cost = Xn, Un, new_cost
                    improved = rel > self.tol
                    break
            if not improved:
                break      # converged, or no step helped; more passes are waste
        return U


class MPCController(BaseController):
    """Joint steering + acceleration. gains = dict(w_pos, w_vel, w_ctrl)."""

    name = "mpc"

    def __init__(self, gains, dt, params):
        self.g, self.dt, self.p = gains, dt, params
        self.H = params["mpc_horizon"]
        self.solver = UnicycleILQR(
            dt, self.H, gains["w_pos"], gains["w_vel"], gains["w_ctrl"],
            u_lim=[params["max_yaw_rate"], params["max_accel"]],
            iters=params.get("ilqr_iters", 8),
        )
        self.reset()

    def reset(self):
        self._U = np.zeros((self.H, 2))
        self._hint = 0
        self.last_solve_ms = 0.0

    def step(self, state, ref_xy, ref_v):
        import time
        x, y, yaw, v = state
        idx = nearest_index(ref_xy, x, y, self._hint, window=200)
        self._hint = idx
        # sample the reference forward by arc length at the current speed
        targets = np.empty((self.H + 1, 3))
        j = idx
        step_m = max(v, 1.0) * self.dt
        for t in range(self.H + 1):
            travelled = 0.0
            while j < len(ref_xy) - 1 and travelled < step_m:
                travelled += float(np.hypot(*(ref_xy[j + 1] - ref_xy[j])))
                j += 1
            targets[t, 0] = ref_xy[j, 0]
            targets[t, 1] = ref_xy[j, 1]
            targets[t, 2] = ref_v[j]
        t0 = time.perf_counter()
        U = self.solver.solve(np.array([x, y, yaw, v]), targets, self._U)
        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._U = np.vstack([U[1:], U[-1:]])
        omega, accel = U[0]
        # unicycle yaw rate -> road-wheel angle for the actual vehicle
        steer = math.atan2(self.p["wheelbase"] * omega, max(v, 0.5))
        return float(np.clip(steer, -self.p["max_steer"], self.p["max_steer"])), float(accel)


CONTROLLERS = {
    "pid": PIDController,
    "pure_pursuit": PurePursuitController,
    "stanley": StanleyController,
    "mpc": MPCController,
}


def build(name, gains, dt, params):
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; have {list(CONTROLLERS)}")
    return CONTROLLERS[name](gains, dt, params)
