"""The backend interface every robot (simulated, real, mock) implements, plus the mock
used by the tests.

Positions are metres in the robot frame (x forward, y left, z up). The interface is
deliberately coarse: pick / place / park are backend primitives because the two
backends sense success differently (contacts in MuJoCo, the gripper angle on the
real arm). The manipulation module sequences them and owns retries and verification.
"""
from __future__ import annotations

import threading


class EStop(RuntimeError):
    """Raised inside any motion primitive as soon as the emergency stop is set."""


class RobotError(RuntimeError):
    pass


class StopFlag:
    """Shared emergency-stop flag: set from any thread, checked every control tick."""

    def __init__(self):
        self._ev = threading.Event()
        self.reason = ""

    def set(self, reason="e-stop"):
        self.reason = reason
        self._ev.set()

    def clear(self):
        self._ev.clear()
        self.reason = ""

    def is_set(self):
        return self._ev.is_set()

    def check(self):
        if self._ev.is_set():
            raise EStop(self.reason)


class Robot:
    """What the game needs from a robot. All motion calls block, poll the stop flag and
    raise EStop when it is set."""
    stop: StopFlag

    # -- perception / io ---------------------------------------------------- #
    def look(self):                      # -> RGB image from the head camera (numpy uint8 HxWx3)
        raise NotImplementedError

    def say(self, text):                 # speak (or print) a short sentence; non-blocking is fine
        print(f"[say] {text}", flush=True)

    def led(self, rgb, period_ms=0):     # status colour
        pass

    def chime(self, notes):              # [(hz, seconds), ...]
        pass

    def wait_signal(self, timeout):      # -> True if the hands-free 'turn complete' signal fired
        return False

    # -- arm ------------------------------------------------------------------ #
    def ready(self):                     # arm to the working pose above the board edge
        raise NotImplementedError

    def park(self):                      # arm home, out of the camera's view; gripper closed
        raise NotImplementedError

    def pick(self, xy, kind) -> bool:    # approach from above, grasp, lift; False = nothing in hand
        raise NotImplementedError

    def place(self, xy, surface_z) -> bool:   # set the held piece down; False = it was not delivered
        raise NotImplementedError

    def in_hand(self) -> bool:           # is something (still) gripped
        raise NotImplementedError

    def abort_hold(self):                # open the gripper wherever the arm is and back off upward
        raise NotImplementedError

    def gesture(self, name):             # optional expressive motion ('think', 'wave', 'celebrate')
        pass

    def estop(self):                     # cut torque / freeze; called once when the stop flag is set
        pass

    def close(self):
        pass


class MockRobot(Robot):
    """Records every call; a scripted list of pick outcomes lets the tests exercise retries.
    Images come from a callable so the tests can feed synthetic renders."""

    def __init__(self, image_fn=None, pick_results=None):
        self.stop = StopFlag()
        self.log = []
        self.image_fn = image_fn
        self.pick_results = list(pick_results or [])
        self.holding = None
        self.pieces = {}          # xy (rounded) -> kind, a tiny 'physical' world for the tests
        self.on_pick = None       # optional callback(xy, kind) -> bool

    def _key(self, xy):
        return (round(float(xy[0]), 3), round(float(xy[1]), 3))

    def look(self):
        self.log.append(("look",))
        self.stop.check()
        return self.image_fn() if self.image_fn else None

    def say(self, text):
        self.log.append(("say", text))

    def led(self, rgb, period_ms=0):
        self.log.append(("led", rgb, period_ms))

    def ready(self):
        self.stop.check()
        self.log.append(("ready",))

    def park(self):
        self.stop.check()
        self.log.append(("park",))

    def pick(self, xy, kind):
        self.stop.check()
        self.log.append(("pick", self._key(xy), kind))
        ok = self.pick_results.pop(0) if self.pick_results else True
        if self.on_pick is not None:
            ok = self.on_pick(xy, kind)
        if ok:
            self.holding = self.pieces.pop(self._key(xy), kind)
        return ok

    def place(self, xy, surface_z):
        self.stop.check()
        self.log.append(("place", self._key(xy), round(surface_z, 3)))
        if self.holding is None:
            return False
        self.pieces[self._key(xy)] = self.holding
        self.holding = None
        return True

    def in_hand(self):
        return self.holding is not None

    def abort_hold(self):
        self.log.append(("abort_hold",))
        self.holding = None

    def gesture(self, name):
        self.log.append(("gesture", name))

    def estop(self):
        self.log.append(("estop",))
