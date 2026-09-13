"""Operator / spectator interfaces.

  Console   keyboard commands on stdin (works in any terminal, no dependencies):
              Enter or d = my turn is complete      n = start a new game
              p = pause   r = resume   x = reset    s = stop
              e (or space) = EMERGENCY STOP         h = I finished the step by hand
              w = what move did you see?            y = why did you play that?
  HttpUI    the same commands as buttons plus the live status frame, served on
            http://<robot>:8010/ so a phone or laptop can drive the demo from across
            the table (and spectators can watch the board map / camera / engine line).
  Display   composes the status frame: 3-D scene or camera, board map with the last
            move and any mismatches, rectified board, engine + explanation, event log.
  speech    say() helpers: espeak-ng -> PCM for the robot's speaker; SAPI / pyttsx3 on
            a laptop; text fallback everywhere.
"""
from __future__ import annotations

import http.server
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import wave
import io

import chess
import cv2
import numpy as np

from .vision import draw_overlay

KEYMAP = {"": "done", "d": "done", "done": "done", "n": "start", "start": "start", "p": "pause", "pause": "pause",
          "r": "resume", "resume": "resume", "x": "reset", "reset": "reset", "s": "stop", "stop": "stop",
          "e": "estop", " ": "estop", "estop": "estop", "h": "help_done", "w": "what", "y": "why", "q": "stop"}


class Console(threading.Thread):
    """Reads stdin lines and submits commands. 'e' sets the stop flag directly (before the
    game loop gets to it) so the arm stops within one control tick."""

    def __init__(self, game):
        super().__init__(daemon=True)
        self.game = game

    def run(self):
        print("keys: Enter=turn complete  n=start  p=pause  r=resume  x=reset  s=stop  e=E-STOP  h=help done  w=what  y=why",
              flush=True)
        while self.game.running:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            key = line.rstrip("\r\n").strip().lower() if line.strip() else ""
            cmd = KEYMAP.get(key)
            if cmd is None:
                print(f"? {key!r}", flush=True)
                continue
            if cmd == "estop":
                self.game.robot.stop.set("keyboard e-stop")
            self.game.submit(cmd)


# --------------------------------------------------------------------------- #
GLYPHS = {"K": "\u265A", "Q": "\u265B", "R": "\u265C", "B": "\u265D", "N": "\u265E", "P": "\u265F"}
FONT_CANDIDATES = [r"C:\Windows\Fonts\seguisym.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/System/Library/Fonts/Apple Symbols.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]


def piece_font(size):
    """A TrueType font with the chess glyphs, or None (letters are drawn instead)."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


class Display:
    """The status frame, laid out for playing on it: a big board on the right (100 px
    squares), the 3-D scene, the cameras and the status text on the left."""
    W, H = 1920, 1080
    # board panel geometry (also used by the window's mouse handler)
    SQ, OX, OY = 100, 60, 40
    PANEL_W, PANEL_H = 920, 1080
    BOARD_ORIGIN = (1000, 0)        # where the board panel sits on the canvas
    LEFT_W = 1000

    def __init__(self, title="Bracket Gambit"):
        self.title = title
        self.lock = threading.Lock()
        self.game = None
        self.board_panel = np.full((self.PANEL_H, self.PANEL_W, 3), 30, np.uint8)
        self.cam_panel = np.full((480, 640, 3), 24, np.uint8)
        self.rect_panel = None
        self.scene_panel = None          # BGR from the simulator (or None)
        self.hand_panel = None           # BGR hand camera (or None)
        self.selected = None             # square picked by the user with the mouse
        self._frame, self._frame_t, self._dirty = None, 0.0, True
        self.font = piece_font(int(self.SQ * 0.78))

    def update(self, game):
        with self.lock:
            self.game = game
            st = game.status
            occ = st.observation.occupancy if st.observation is not None else None
            self.board_panel = self.draw_board(game.board, occ, st.mismatches, st.moves, self.selected, self.font)
            if st.observation is not None:
                self.cam_panel = cv2.resize(draw_overlay(st.observation, game.cfg.board.marker_offset), (640, 480))
                self.rect_panel = cv2.cvtColor(st.observation.rectified, cv2.COLOR_RGB2BGR)
            self._dirty = True

    def set_views(self, scene, hand):
        with self.lock:
            self.scene_panel, self.hand_panel = scene, hand
            self._dirty = True

    def select(self, sq):
        with self.lock:
            self.selected = sq
            if self.game is not None:
                st = self.game.status
                occ = st.observation.occupancy if st.observation is not None else None
                self.board_panel = self.draw_board(self.game.board, occ, st.mismatches, st.moves, sq, self.font)
            self._dirty = True

    def compose(self, scene=None):
        """Full status frame (BGR). Left column (1000 px): 3-D scene 1000x562, then head
        camera / rectified board / hand camera, then the status text. Right: the board."""
        with self.lock:
            scene = scene if scene is not None else self.scene_panel
            canvas = np.full((self.H, self.W, 3), 24, np.uint8)
            L = self.LEFT_W
            canvas[:562, :L] = cv2.resize(scene if scene is not None else self.cam_panel, (L, 562))
            canvas[562:862, :400] = cv2.resize(self.cam_panel, (400, 300))
            cv2.putText(canvas, "head camera", (8, 582), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self.rect_panel is not None:
                canvas[562:862, 400:700] = cv2.resize(self.rect_panel, (300, 300))
                cv2.putText(canvas, "rectified board", (408, 582), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if self.hand_panel is not None:
                canvas[562:787, 700:L] = cv2.resize(self.hand_panel, (300, 225))
                cv2.putText(canvas, "hand camera", (708, 582), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for x in (400, 700):
                cv2.line(canvas, (x, 562), (x, 862), (60, 60, 60), 1)
            bx, by = self.BOARD_ORIGIN
            canvas[by:by + self.PANEL_H, bx:bx + self.PANEL_W] = self.board_panel
            g = self.game
            y = 895
            cv2.putText(canvas, self.title, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if g is not None:
                st = g.status
                phase = st.phase.value.upper()
                colour = {"needs help": (0, 0, 255), "stopped": (0, 0, 255), "recovering": (0, 120, 255),
                          "waiting for the human": (255, 200, 80), "moving": (255, 80, 255)}.get(st.phase.value, (120, 220, 255))
                cv2.putText(canvas, f"{phase}  {st.detail}", (16, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
                lines = [f"engine: {st.engine}   eval: {st.eval}     human: {st.last_human or '-'}   robot: {st.last_robot or '-'}"]
                lines += wrap(st.explanation, 100)[:2]
                recent = []
                for e in st.events[-3:]:
                    recent += wrap(e[9:] if len(e) > 9 else e, 100)[:1]
                lines += recent[-3:]
                for i, line in enumerate(lines):
                    cv2.putText(canvas, line, (16, y + 66 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (230, 230, 230) if i == 0 else (190, 190, 190), 1, cv2.LINE_AA)
            return canvas

    def frame(self):
        """The composed frame, rebuilt only when something changed (and at most ~10 Hz)."""
        with self.lock:
            f = self._frame
            stale = f is None or ((time.time() - self._frame_t) > 0.1 and self._dirty)
        if stale:
            f = self.compose()
            with self.lock:
                self._frame, self._frame_t, self._dirty = f, time.time(), False
        return f

    @classmethod
    def square_at(cls, x, y):
        """Board panel pixel -> square index, or None."""
        f, r = (x - cls.OX) // cls.SQ, 7 - (y - cls.OY) // cls.SQ
        return chess.square(int(f), int(r)) if 0 <= f < 8 and 0 <= r < 8 else None

    @classmethod
    def draw_board(cls, board, occ, mismatches, moves, selected=None, font=None):
        panel = np.full((cls.PANEL_H, cls.PANEL_W, 3), 30, np.uint8)
        s, ox, oy = cls.SQ, cls.OX, cls.OY
        bad = {m[0] for m in (mismatches or [])}
        last = board.peek() if board.move_stack else None
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x0, y0 = ox + f * s, oy + (7 - r) * s
            light = (f + r) % 2 == 1
            col = (140, 190, 220) if light else (70, 110, 150)
            if last is not None and sq in (last.from_square, last.to_square):
                col = (120, 210, 200) if light else (80, 160, 140)          # last move: greenish tint
            cv2.rectangle(panel, (x0, y0), (x0 + s, y0 + s), col, -1)
            if sq == selected:
                cv2.rectangle(panel, (x0 + 3, y0 + 3), (x0 + s - 3, y0 + s - 3), (0, 220, 255), 5)
            if chess.square_name(sq) in bad:
                cv2.rectangle(panel, (x0 + 3, y0 + 3), (x0 + s - 3, y0 + s - 3), (0, 0, 255), 5)
            seen = occ.get(sq) if occ else None
            if seen is not None:      # small dot: what the camera saw
                cv2.circle(panel, (x0 + s - 12, y0 + 12), 6, (255, 255, 255) if seen == "w" else (0, 0, 0), -1)
        if selected is not None and board.piece_at(selected) is not None:
            for mv in board.legal_moves:
                if mv.from_square == selected:
                    tf, tr = chess.square_file(mv.to_square), chess.square_rank(mv.to_square)
                    cv2.circle(panel, (ox + tf * s + s // 2, oy + (7 - tr) * s + s // 2), 13, (0, 220, 255), -1)
        cls._draw_pieces(panel, board, font)
        for i in range(8):
            cv2.putText(panel, "abcdefgh"[i], (ox + i * s + s // 2 - 9, oy + 8 * s + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(panel, str(8 - i), (ox - 36, oy + i * s + s // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        if last is not None:
            colour = (60, 200, 60) if not board.turn else (60, 140, 255)
            pa = (ox + chess.square_file(last.from_square) * s + s // 2, oy + (7 - chess.square_rank(last.from_square)) * s + s // 2)
            pb = (ox + chess.square_file(last.to_square) * s + s // 2, oy + (7 - chess.square_rank(last.to_square)) * s + s // 2)
            cv2.arrowedLine(panel, pa, pb, colour, 4, cv2.LINE_AA, tipLength=0.2)
        y = oy + 8 * s + 70
        cv2.putText(panel, "You play black: click a piece, then its square.   f = full screen   e = E-STOP", (ox, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        text = " ".join(f"{i // 2 + 1}. {m}" if i % 2 == 0 else m for i, m in enumerate(moves))
        for i, line in enumerate(wrap(text, 70)[-4:]):
            cv2.putText(panel, line, (ox, y + 34 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 1, cv2.LINE_AA)
        return panel

    @classmethod
    def _draw_pieces(cls, panel, board, font):
        s, ox, oy = cls.SQ, cls.OX, cls.OY
        if font is None:
            for sq, p in board.piece_map().items():
                x0, y0 = ox + chess.square_file(sq) * s, oy + (7 - chess.square_rank(sq)) * s
                cv2.circle(panel, (x0 + s // 2, y0 + s // 2), int(s * 0.36), (245, 240, 235) if p.color else (25, 25, 25), -1)
                cv2.putText(panel, p.symbol().upper(), (x0 + s // 2 - 16, y0 + s // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (30, 30, 30) if p.color else (230, 230, 230), 3, cv2.LINE_AA)
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for sq, p in board.piece_map().items():
            x0, y0 = ox + chess.square_file(sq) * s, oy + (7 - chess.square_rank(sq)) * s
            glyph = GLYPHS[p.symbol().upper()]
            fill, outline = ((248, 246, 240), (40, 40, 40)) if p.color else ((28, 28, 30), (200, 200, 200))
            draw.text((x0 + s / 2, y0 + s / 2 + 3), glyph, font=font, fill=fill, anchor="mm",
                      stroke_width=3, stroke_fill=outline)
        panel[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def wrap(text, width):
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


class Window(threading.Thread):
    """The dashboard in a resizable OpenCV window: drag it to any size, press `f` for full
    screen, click the board to play. (A resizable window reports mouse positions in image
    pixels, so the board geometry needs no scaling.)"""

    def __init__(self, display, game, width=1600, height=900):
        super().__init__(daemon=True)
        self.display, self.game = display, game
        self.width, self.height = width, height
        self.fullscreen = False

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx, by = self.display.BOARD_ORIGIN
        sq = self.display.square_at(x - bx, y - by)
        if sq is None:
            self.display.select(None)
            return
        sel = self.display.selected
        if sel is None or sel == sq:
            self.display.select(None if sel == sq else sq)
        else:
            self.display.select(None)
            self.game.submit(f"move:{chess.square_name(sel)}{chess.square_name(sq)}")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.display.title, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        if not self.fullscreen:
            cv2.resizeWindow(self.display.title, self.width, self.height)

    def run(self):
        cv2.namedWindow(self.display.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.display.title, self.width, self.height)
        cv2.setMouseCallback(self.display.title, self.on_mouse)
        while self.game.running:
            cv2.imshow(self.display.title, self.display.frame())
            k = cv2.waitKey(100) & 0xFF
            if k == 255:
                continue
            if k in (ord("f"), ord("F")):
                self.toggle_fullscreen()
                continue
            if k == 27 and self.fullscreen:          # Esc leaves full screen
                self.toggle_fullscreen()
                continue
            cmd = KEYMAP.get(chr(k).lower() if k not in (10, 13) else "")
            if cmd:
                if cmd == "estop":
                    self.game.robot.stop.set("window e-stop")
                self.game.submit(cmd)
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Bracket Gambit</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:12px}
button{font-size:20px;padding:12px 18px;margin:5px;border-radius:8px;border:0;background:#2d6cdf;color:#fff}
button.red{background:#d9302c}button.grey{background:#555}#st{font-size:20px;margin:8px 0}
img{max-width:100%}
#wrap{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
#board{display:grid;grid-template-columns:repeat(8,1fr);width:min(92vw,480px);aspect-ratio:1;border:3px solid #333;user-select:none}
.sq{display:flex;align-items:center;justify-content:center;font-size:min(9vw,46px);cursor:pointer;line-height:1}
.l{background:#dcc39a}.d{background:#8b5a2b}.sel{outline:4px solid #ffb000;outline-offset:-4px}
.last{box-shadow:inset 0 0 0 60px rgba(80,220,120,.35)}.bad{box-shadow:inset 0 0 0 4px #f22}
.dot::after{content:"";width:22%;height:22%;border-radius:50%;background:rgba(255,176,0,.9);position:absolute}
.sq{position:relative}.w{color:#fff;text-shadow:0 0 3px #000,0 0 3px #000,0 0 3px #000}.b{color:#111;text-shadow:0 0 3px #fff,0 0 3px #fff}
#hint{color:#aaa;font-size:15px;margin:6px 0}#why{color:#8cf;font-size:16px;max-width:480px}
</style></head><body>
<div><button onclick="c('done')">My turn is complete</button><button onclick="c('start')">Start game</button>
<button class="grey" onclick="c('pause')">Pause</button><button class="grey" onclick="c('resume')">Resume</button>
<button class="grey" onclick="c('reset')">Reset</button><button class="grey" onclick="c('what')">What did you see?</button>
<button class="grey" onclick="c('why')">Why?</button><button class="grey" onclick="c('help_done')">Fixed by hand</button>
<button class="red" onclick="c('estop')">EMERGENCY STOP</button><button class="red" onclick="c('stop')">Stop</button></div>
<div id="st">...</div>
<div id="wrap"><div><div id="board"></div>
<div id="hint">You play black: tap a piece, then its destination. (Simulation: this moves the piece on the table. Robot: move it on the real board too - the camera decides.)</div>
<div id="why"></div></div>
<div style="flex:1;min-width:320px"><img id="f" src="/frame.jpg"></div></div>
<script>
function c(x){fetch('/cmd?c='+x)}
const G={K:'\\u265A',Q:'\\u265B',R:'\\u265C',B:'\\u265D',N:'\\u265E',P:'\\u265F'};
let sel=null, fen='', legal=[], last='', bad=[];
function applyLocal(u){ // show the move at once; the server's position replaces it on the next poll
 const pm=parseFen(fen);const p=pm[u.slice(0,2)];if(!p)return;delete pm[u.slice(0,2)];pm[u.slice(2,4)]=p;
 let rows=[];for(let r=8;r>=1;r--){let s='',e=0;for(const f of 'abcdefgh'){const q=pm[f+r];if(q){if(e){s+=e;e=0}s+=q}else e++}if(e)s+=e;rows.push(s)}
 fen=rows.join('/')+' '+fen.split(' ').slice(1).join(' ');last=u;legal=[];render()}
function parseFen(f){const rows=f.split(' ')[0].split('/');const m={};rows.forEach((row,ri)=>{let file=0;for(const ch of row){if(/\\d/.test(ch)){file+=+ch}else{m['abcdefgh'[file]+(8-ri)]=ch;file++}}});return m}
function render(){const b=document.getElementById('board');b.innerHTML='';const pm=parseFen(fen);
 for(let r=8;r>=1;r--)for(let f=0;f<8;f++){const name='abcdefgh'[f]+r;const d=document.createElement('div');
  d.className='sq '+(((f+r)%2)?'l':'d');const p=pm[name];
  if(p){d.textContent=G[p.toUpperCase()];d.classList.add(p===p.toUpperCase()?'w':'b')}
  if(sel===name)d.classList.add('sel');
  if(last&&(last.slice(0,2)===name||last.slice(2,4)===name))d.classList.add('last');
  if(bad.includes(name))d.classList.add('bad');
  if(sel&&legal.some(u=>u.slice(0,2)===sel&&u.slice(2,4)===name))d.classList.add('dot');
  d.onclick=()=>{if(sel===null){if(p){sel=name;render()}}else if(sel===name){sel=null;render()}else{const u=sel+name;sel=null;fetch('/cmd?c=move:'+u);applyLocal(u)}};
  b.appendChild(d)}}
let seen='';
setInterval(()=>{fetch('/status.json').then(r=>r.json()).then(s=>{document.getElementById('st').textContent=s.phase.toUpperCase()+' - '+s.detail+'   | '+s.moves.join(' ');
 document.getElementById('why').textContent=s.explanation||'';
 const key=s.fen+'|'+JSON.stringify(s.bad);
 if(key!==seen){seen=key;fen=s.fen;legal=s.legal||[];last=s.last_uci||'';bad=s.bad||[];render()}})},300);
setInterval(()=>{document.getElementById('f').src='/frame.jpg?'+Date.now()},500)
</script></body></html>"""


class HttpUI:
    def __init__(self, game, display, port=8010):
        self.game, self.display, self.port = game, display, port
        ui = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                if url.path == "/":
                    self._send(200, "text/html", PAGE.encode())
                elif url.path == "/frame.jpg":
                    ok, buf = cv2.imencode(".jpg", cv2.resize(ui.display.frame(), (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 75])
                    self._send(200, "image/jpeg", buf.tobytes())
                elif url.path == "/status.json":
                    st = ui.game.status
                    board = ui.game.board
                    self._send(200, "application/json", json.dumps({
                        "phase": st.phase.value, "detail": st.detail, "moves": st.moves, "last_human": st.last_human,
                        "last_robot": st.last_robot, "eval": st.eval, "engine": st.engine, "explanation": st.explanation,
                        "fen": board.fen(), "legal": [m.uci() for m in board.legal_moves],
                        "last_uci": board.peek().uci() if board.move_stack else "",
                        "bad": [m[0] for m in st.mismatches], "events": st.events[-10:], "result": st.result}).encode())
                elif url.path == "/cmd":
                    cmd = urllib.parse.parse_qs(url.query).get("c", [""])[0]
                    if cmd == "estop":
                        ui.game.robot.stop.set("web e-stop")
                    ui.game.submit(cmd)
                    self._send(200, "text/plain", b"ok")
                else:
                    self._send(404, "text/plain", b"not found")

        self.server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        print(f"web UI + spectator display: http://localhost:{self.port}/  (also reachable on the LAN)", flush=True)

    def stop(self):
        self.server.shutdown()


# --------------------------------------------------------------------------- #
# speech
# --------------------------------------------------------------------------- #
def tts_pcm(text, sample_rate):
    """PCM int16 mono at sample_rate from espeak-ng / espeak (Linux, the robot), or None."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-s", "150", "-v", "en", "--stdout", text], capture_output=True, timeout=10).stdout
        with wave.open(io.BytesIO(out)) as w:
            sr, n, data = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
        pcm = np.frombuffer(data, np.int16).reshape(-1, n)[:, 0].astype(np.float32)
        if sr != sample_rate:
            t = np.arange(0, len(pcm) / sr, 1 / sample_rate)
            pcm = np.interp(t, np.arange(len(pcm)) / sr, pcm)
        return pcm.astype(np.int16)
    except Exception:
        return None


def speak_local(text):
    """Best-effort speech on a laptop (background thread): pyttsx3, then Windows SAPI."""
    def run():
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.say(text)
            e.runAndWait()
            return
        except Exception:
            pass
        if platform.system() == "Windows":
            safe = text.replace("'", "")
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"],
                           capture_output=True, timeout=30)
    threading.Thread(target=run, daemon=True).start()

