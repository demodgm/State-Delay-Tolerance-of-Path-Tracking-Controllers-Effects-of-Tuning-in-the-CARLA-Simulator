"""Sensing-to-actuation delay injection.

The entire experimental manipulation lives in this file. Every state the
simulator produces is pushed into a FIFO; the controller reads the entry that
is `n` steps old. Nothing in any controller is modified, so all four
controllers see identically-constructed inputs and any behavioural difference
is attributable to the control law rather than to the harness.
"""

from collections import deque


class DelayBuffer:
    """FIFO holding the last n+1 states; .stale() returns the n-step-old one.

    Before the buffer has filled (the first n steps of a run) it returns the
    oldest state available, so the vehicle starts from rest with a consistent
    if stale view of itself rather than with no view at all.
    """

    def __init__(self, n_steps: int):
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        self.n = int(n_steps)
        self._buf = deque(maxlen=self.n + 1)

    @classmethod
    def from_delay(cls, delay_s: float, dt: float):
        """Build from a delay in seconds. Delay is quantised to whole steps.

        Raises if the requested delay isn't close to a multiple of dt, since
        silent rounding would put a systematic error into the x-axis of every
        figure in the paper.
        """
        exact = delay_s / dt
        n = round(exact)
        if abs(exact - n) > 1e-6:
            raise ValueError(
                f"delay {delay_s * 1000:.3f} ms is not a whole multiple of "
                f"dt={dt * 1000:.3f} ms (would need {exact:.4f} steps). "
                "Choose sweep values that are exact multiples of dt."
            )
        return cls(n)

    def push(self, state):
        self._buf.append(state)

    def stale(self):
        if not self._buf:
            raise RuntimeError("stale() called before any push()")
        return self._buf[0]

    @property
    def is_warm(self) -> bool:
        """True once the buffer holds a genuinely n-step-old state."""
        return len(self._buf) == self.n + 1

    def reset(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)


class JitterBuffer(DelayBuffer):
    """Variable delay: n is resampled every step.

    Not used by the primary sweep. Present because future work item 1 is to
    replace constant delay with a distribution, and the interface shouldn't
    have to change when that happens.
    """

    def __init__(self, mean_steps: int, std_steps: float, rng, max_steps=None):
        cap = max_steps if max_steps is not None else int(mean_steps + 6 * std_steps + 2)
        super().__init__(cap)
        self.mean_steps = mean_steps
        self.std_steps = std_steps
        self.rng = rng
        self._current_n = mean_steps

    def push(self, state):
        self._buf.append(state)
        n = int(round(self.rng.gauss(self.mean_steps, self.std_steps)))
        self._current_n = max(0, min(n, len(self._buf) - 1))

    def stale(self):
        if not self._buf:
            raise RuntimeError("stale() called before any push()")
        # deque[-1] is newest; index from the end by the sampled age
        return self._buf[len(self._buf) - 1 - self._current_n]
