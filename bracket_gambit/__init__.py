"""Bracket Gambit: BracketBot plays chess against a human with Stockfish.

Modules
  config        board geometry / vision / motion parameters (JSON calibration file)
  vision        camera image -> corner markers -> rectified board -> square occupancy
  engine        Stockfish (UCI) with a labelled built-in fallback, plus grounded explanations
  game          the turn state machine (start / turn complete / pause / reset / stop / e-stop)
  manipulation  chess move -> pick-and-place steps with grasp verification and retries
  robot         the backend interface + a mock for tests
  sim_robot     MuJoCo backend (chopped_urdf_v2/sim)
  bbos_robot    real robot backend (bbos daemons)  -- hardware-unverified
  ui            keyboard/HTTP control, status display, speech
"""
__version__ = "0.1.0"
