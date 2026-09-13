# Stockfish binaries

Bracket Gambit looks here automatically (`bracket_gambit/engine.py: find_stockfish`), after
`--stockfish PATH` and `$STOCKFISH` and before `stockfish` on PATH. Drop the binary for your
platform in this folder; any file named `stockfish*` counts.

| platform | download (Stockfish 19) | file to put here |
|---|---|---|
| Windows x86-64 (laptop, simulation) | https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip | `stockfish.exe` (already here) |
| Linux arm64 (the robot, if `uname -m` = aarch64) | https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-arm64-universal.tar.gz | `stockfish-linux-arm64-universal` (`chmod +x`) |
| Linux x86-64 | https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz | `stockfish-linux-x86-64-universal` |
| macOS | https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz | `stockfish-macos-universal` |

On the robot `sudo apt install stockfish` also works (it lands on PATH). Stockfish is GPL-3;
the binary is unmodified and this folder is the only place it lives in the project.
