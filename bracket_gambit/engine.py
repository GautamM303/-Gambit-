"""Move selection: Stockfish over UCI (python-chess), with a clearly labelled built-in
fallback so the demo still runs if the binary is missing, and short explanations that
are grounded in the position / the engine's evaluation rather than invented reasoning.
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field

import chess
import chess.engine

PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


@dataclass
class Choice:
    move: chess.Move
    score_cp: int | None            # from the mover's point of view, centipawns (None if mate)
    mate_in: int | None
    pv: list = field(default_factory=list)
    depth: int | None = None
    source: str = "stockfish"


class Engine:
    name = "engine"

    def choose(self, board: chess.Board) -> Choice: ...
    def close(self): ...


class StockfishEngine(Engine):
    """Stockfish through python-chess. skill 0..20 (Skill Level); think_time seconds per move."""

    def __init__(self, path, skill=5, think_time=1.0, depth=None, threads=1):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.name = self.engine.id.get("name", "Stockfish")
        self.engine.configure({"Skill Level": int(skill), "Threads": int(threads)})
        self.limit = chess.engine.Limit(time=think_time, depth=depth)
        self.skill = skill

    def choose(self, board):
        r = self.engine.play(board, self.limit, info=chess.engine.INFO_ALL)
        score = r.info.get("score")
        pov = score.pov(board.turn) if score is not None else None
        return Choice(r.move, None if pov is None or pov.is_mate() else pov.score(),
                      pov.mate() if pov is not None and pov.is_mate() else None,
                      list(r.info.get("pv", [])), r.info.get("depth"), self.name)

    def analyse(self, board, depth=12):
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        return info["score"].pov(board.turn)

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


class FallbackEngine(Engine):
    """Small alpha-beta searcher used ONLY when Stockfish is unavailable. Every choice is
    labelled source='fallback' so the status display and logs never pass it off as Stockfish."""
    name = "built-in fallback (NOT Stockfish)"

    def __init__(self, depth=3, seed=0):
        self.depth = depth
        self.rng = random.Random(seed)

    def evaluate(self, board):
        if board.is_checkmate():
            return -100000
        score = 0
        for sq, p in board.piece_map().items():
            f, r = chess.square_file(sq), chess.square_rank(sq)
            v = VALUES[p.piece_type] + 4 * (3 - abs(f - 3.5) - abs(r - 3.5))
            score += v if p.color == board.turn else -v
        return int(score)

    def search(self, board, depth, alpha=-10 ** 9, beta=10 ** 9):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board), None
        best_mv, best = None, -10 ** 9
        moves = sorted(board.legal_moves, key=lambda m: (not board.is_capture(m), self.rng.random()))
        for mv in moves:
            board.push(mv)
            score = -self.search(board, depth - 1, -beta, -alpha)[0]
            board.pop()
            if score > best:
                best, best_mv = score, mv
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best, best_mv

    def choose(self, board):
        score, mv = self.search(board, self.depth)
        return Choice(mv, score, None, [mv], self.depth, "fallback")

    def analyse(self, board, depth=2):
        return chess.engine.Cp(self.search(board, depth)[0])


def find_stockfish(path=None):
    """Explicit path, $STOCKFISH, the project's tools/stockfish/ folder, then 'stockfish' on PATH."""
    import glob
    import pathlib
    tools = pathlib.Path(__file__).resolve().parents[1] / "tools" / "stockfish"
    local = sorted(glob.glob(str(tools / "stockfish*")))
    local = [p for p in local if os.path.isfile(p) and not p.endswith((".zip", ".tar.gz", ".txt", ".md"))]
    for cand in [path, os.environ.get("STOCKFISH")] + local + [shutil.which("stockfish"), shutil.which("stockfish.exe")]:
        if cand and os.path.exists(cand):
            return cand
    return None


def open_engine(path=None, skill=5, think_time=1.0, allow_fallback=True, log=print):
    exe = find_stockfish(path)
    if exe is not None:
        e = StockfishEngine(exe, skill=skill, think_time=think_time)
        log(f"engine: {e.name} ({exe}), skill {skill}, {think_time:.1f}s/move")
        return e
    if not allow_fallback:
        raise FileNotFoundError("Stockfish not found: pass --stockfish PATH or set $STOCKFISH")
    log("engine: STOCKFISH NOT FOUND - using the built-in fallback searcher (weak; labelled as such)")
    return FallbackEngine()


# --------------------------------------------------------------------------- #
# explanations
# --------------------------------------------------------------------------- #
def describe_move(board: chess.Board, move: chess.Move) -> str:
    """'knight from g1 to f3', 'queen takes the pawn on g5', 'castles kingside'."""
    if board.is_castling(move):
        return "castles " + ("kingside" if chess.square_file(move.to_square) == 6 else "queenside")
    piece = board.piece_at(move.from_square)
    name = PIECE_NAMES[piece.piece_type] if piece else "piece"
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        vname = PIECE_NAMES[victim.piece_type] if victim else "pawn"
        return f"{name} from {frm} takes the {vname} on {to}"
    return f"{name} from {frm} to {to}"


def explain(board: chess.Board, choice: Choice) -> str:
    """One or two sentences grounded in concrete features of the move and the engine's
    evaluation. Never claims human-style intent."""
    mv = choice.move
    facts = []
    after = board.copy(stack=False)
    after.push(mv)
    if after.is_checkmate():
        facts.append("it is checkmate")
    elif after.is_check():
        facts.append("it gives check")
    if board.is_capture(mv):
        victim = board.piece_at(mv.to_square)
        if victim is not None:
            facts.append(f"it wins a {PIECE_NAMES[victim.piece_type]}" if VALUES[victim.piece_type] >= 300
                         else "it takes a pawn")
    if board.is_castling(mv):
        facts.append("it brings the king to safety and connects the rooks")
    piece = board.piece_at(mv.from_square)
    if piece is not None and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and \
            chess.square_rank(mv.from_square) in (0, 7):
        facts.append("it develops a piece")
    if mv.to_square in (chess.D4, chess.E4, chess.D5, chess.E5):
        facts.append("it occupies the centre")
    # concrete threats: does the moved piece now attack something more valuable than itself?
    if piece is not None:
        targets = [after.piece_at(s) for s in after.attacks(mv.to_square)]
        big = [t for t in targets if t is not None and t.color != piece.color and VALUES[t.piece_type] > VALUES[piece.piece_type]]
        if big and not facts:
            facts.append(f"it attacks the {PIECE_NAMES[max(big, key=lambda t: VALUES[t.piece_type]).piece_type]}")
    src = "Stockfish" if choice.source.lower().startswith("stockfish") else "the fallback engine"
    if choice.mate_in is not None:
        ev = f"{src} sees mate in {abs(choice.mate_in)}"
    elif choice.score_cp is not None:
        pawns = choice.score_cp / 100
        ev = f"{src} rates the position {pawns:+.1f} pawns for me" if abs(pawns) >= 0.3 else f"{src} calls the position level"
    else:
        ev = f"{src} chose it"
    head = "I played this because " + " and ".join(facts[:2]) + "." if facts else "It was the best move in my search."
    tail = ev + (f", expecting {board.variation_san(choice.pv[:3])}." if len(choice.pv) > 1 else ".")
    return f"{head} {tail}"
