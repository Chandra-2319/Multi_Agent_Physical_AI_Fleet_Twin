# SPDX-License-Identifier: Apache-2.0
"""numpy / torch / warp compatibility for the Isaac Sim 6.x walking policies.

Why this exists
---------------
In Isaac Sim 6.0.x the policy classes in `isaacsim.robot.policy.examples.robots` (go2.py, h1.py, spot.py) build
their observation as a *torch* tensor on `robot._device` and assign the velocity command into it:

    obs = torch.zeros(48, device=device)
    obs[9:12] = command            # command must be a torch tensor/list, NOT a numpy array

The official examples pass `torch.tensor([...], device="cuda")`. A numpy array raises
    TypeError: can't assign a numpy.ndarray to a torch.FloatTensor
and, because that happens before `set_dof_position_targets`, no joint ever gets a target.

`PolicyRunner` builds the command as a torch tensor on the policy's own device (falling back to numpy for older
Isaac Sim versions), remembers which type was accepted, and turns a persistent failure into a clean error state
instead of a traceback on every physics step. `to_numpy` converts torch (CPU/CUDA) / warp / numpy to numpy.
"""
from __future__ import annotations

import traceback

import numpy as np

STYLES = ("torch", "numpy")
MAX_CONSECUTIVE_FAILURES = 25  # ~0.1 s at 200 Hz: long enough to ride out a not-yet-ready physics view


def is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def is_warp(x) -> bool:
    return type(x).__module__.split(".")[0] == "warp"


def to_numpy(x, dtype=np.float64) -> np.ndarray:
    """torch tensor (CPU or CUDA) / warp array / numpy / sequence -> numpy array."""
    if is_torch(x):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, (np.ndarray, list, tuple, int, float)) and callable(getattr(x, "numpy", None)):
        x = x.numpy()  # warp arrays (host copy of a CUDA array)
    return np.asarray(x, dtype=dtype)


def to_torch(x, device=None):
    """Anything array-like -> float32 torch tensor on `device` (a torch device string such as 'cuda:0')."""
    import torch

    arr = to_numpy(x, np.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=device) if device else torch.as_tensor(arr, dtype=torch.float32)


def convert(value, style: str, device=None):
    """Convert arrays / sequences / tensors to the requested style; leave scalars, strings, None alone."""
    if isinstance(value, (np.ndarray, list, tuple)) or is_torch(value) or is_warp(value):
        return to_torch(value, device) if style == "torch" else to_numpy(value)
    return value


def make_command(command, style: str, device=None):
    return convert(np.asarray([float(v) for v in command], dtype=np.float64), style, device)


def zeros_like_any(x):
    """zeros_like that keeps torch tensors as torch tensors (same dtype/device) and numpy as numpy."""
    if is_torch(x):
        import torch

        return torch.zeros_like(x)
    return np.zeros_like(np.asarray(x))


def policy_device(policy) -> str | None:
    """The device the policy's tensors live on (`policy.robot._device`), e.g. 'cuda:0' or 'cpu'."""
    dev = getattr(getattr(policy, "robot", None), "_device", None)
    return str(dev) if dev is not None else None


def call_variants(obj, name: str, *args, **kwargs) -> tuple:
    """Call obj.name(...) with numpy arguments, then with torch arguments if that raises.

    Returns (ok, error). A method that does not exist is not an error: (False, "").
    """
    fn = getattr(obj, name, None)
    if not callable(fn):
        return False, ""
    last = ""
    for style in ("numpy", "torch"):
        try:
            fn(*[convert(a, style) for a in args], **{k: convert(v, style) for k, v in kwargs.items()})
            return True, ""
        except Exception as exc:  # noqa: BLE001 - wrong array type, API difference, view not ready ...
            last = f"{type(exc).__name__}: {exc}"[:120]
    return False, last


class PolicyRunner:
    """Calls policy.forward(dt, command) with whichever command type this Isaac Sim version accepts."""

    def __init__(self) -> None:
        self.style: str | None = None  # learned on the first successful call
        self.failures = 0
        self.last_error = ""
        self.failed = False  # latched after MAX_CONSECUTIVE_FAILURES: stop calling, report, wait for re-init
        self._reported = False

    def forward(self, policy, dt: float, command) -> bool:
        """True when the policy accepted the command. Never raises."""
        if self.failed:
            return False
        device = policy_device(policy)
        errors = []
        for style in ([self.style] if self.style else list(STYLES)):
            try:
                policy.forward(dt, make_command(command, style, device))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{style}: {type(exc).__name__}: {exc}")
                continue
            if self.style is None:
                print(f"[Physical_AI_Fleet] walking policy accepts '{style}' commands (device={device})")
            self.style, self.failures = style, 0
            return True
        self.failures += 1
        self.last_error = " | ".join(errors)[:200]
        if not self._reported:  # one report in the log, not one per physics step
            self._reported = True
            print(f"[Physical_AI_Fleet] policy.forward failed: {self.last_error}\n{traceback.format_exc()}")
        if self.failures >= MAX_CONSECUTIVE_FAILURES:
            self.failed = True
        return False

    def reset(self) -> None:
        """After a (re)initialisation or an operator reset, so a latched failure can be retried."""
        self.failures, self.failed, self.last_error, self._reported = 0, False, "", False
