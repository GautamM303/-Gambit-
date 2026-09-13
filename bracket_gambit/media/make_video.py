"""Cut the judges' demo video (~3 min, narrated) from a recorded simulation.

    uv run python bracket_gambit/media/make_video.py --video chopped_urdf_v2/sim/gambit.mp4 \
        --timeline chopped_urdf_v2/sim/gambit.timeline.json --out bracket_gambit/media/Bracket_Gambit_demo.mp4

Uses the phase timeline written by `gambit.py play --sim --record` to find the moments
(setup read, first move, human move inference, a capture, ...), speeds each segment so the
whole story fits, overlays captions and a SIMULATION tag, and adds narration rendered with
the Windows speech synthesiser (skipped, with a note, on other platforms).
"""
import argparse
import json
import pathlib
import platform
import subprocess
import wave

import cv2
import imageio
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 30
SR = 22050
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# narration
# --------------------------------------------------------------------------- #
def tts_wav(text, path, rate=1):
    if platform.system() != "Windows":
        return None
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.Rate = {rate}; $s.SetOutputToWaveFile('{path}'); $s.Speak(@'\n{text}\n'@); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=120)
    return path if pathlib.Path(path).exists() else None


def load_wav(path):
    with wave.open(str(path)) as w:
        sr, n, data = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    pcm = np.frombuffer(data, np.int16).reshape(-1, n)[:, 0].astype(np.float32)
    if sr != SR:
        t = np.arange(0, len(pcm) / sr, 1 / SR)
        pcm = np.interp(t, np.arange(len(pcm)) / sr, pcm)
    return pcm


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def card(lines, accent_first=True, sub=None):
    img = np.full((H, W, 3), (22, 18, 18), np.uint8)
    y = H // 2 - 30 * len(lines)
    for i, line in enumerate(lines):
        size, thick = (2.0, 4) if (i == 0 and accent_first) else (1.0, 2)
        colour = (0, 176, 255) if (i == 0 and accent_first) else (235, 235, 235)
        (tw, th), _ = cv2.getTextSize(line, FONT, size, thick)
        cv2.putText(img, line, ((W - tw) // 2, y + th), FONT, size, colour, thick, cv2.LINE_AA)
        y += th + (40 if i == 0 else 26)
    if sub:
        cv2.putText(img, sub, (40, H - 40), FONT, 0.6, (150, 150, 160), 1, cv2.LINE_AA)
    return img


def checklist_card(title, items):
    img = np.full((H, W, 3), (22, 18, 18), np.uint8)
    cv2.putText(img, title, (60, 90), FONT, 1.5, (0, 176, 255), 3, cv2.LINE_AA)
    for i, (mark, text) in enumerate(items):
        colour = {"+": (120, 230, 120), "~": (0, 176, 255), "-": (150, 150, 160)}[mark]
        glyph = {"+": "[x]", "~": "[~]", "-": "[ ]"}[mark]
        cv2.putText(img, f"{glyph} {text}", (70, 160 + 44 * i), FONT, 0.85, colour, 2, cv2.LINE_AA)
    return img


def caption(img, text, tag="SIMULATION  (MuJoCo, robot CAD model)"):
    out = img.copy()
    if tag:
        cv2.rectangle(out, (0, 0), (430, 34), (20, 20, 20), -1)
        cv2.putText(out, tag, (10, 24), FONT, 0.55, (0, 176, 255), 1, cv2.LINE_AA)
    if text:
        lines = wrap(text, 70)
        h = 22 + 34 * len(lines)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, H - h), (W, H), (15, 15, 15), -1)
        out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (30, H - h + 32 + 34 * i), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
# story
# --------------------------------------------------------------------------- #
def find_segments(timeline, total_frames):
    """(start, end, phase, detail) spans between consecutive phase changes."""
    spans = []
    for i, e in enumerate(timeline):
        end = timeline[i + 1]["frame"] if i + 1 < len(timeline) else total_frames
        spans.append((e["frame"], end, e["phase"], e.get("detail", "")))
    return spans


def span_from_to(spans, start_pred, end_pred, after=0):
    """Frames from the first span matching start_pred (index >= after) to the first later span matching end_pred."""
    for i in range(after, len(spans)):
        if start_pred(spans[i]):
            for j in range(i + 1, len(spans)):
                if end_pred(spans[j]):
                    return spans[i][0], spans[j][0], j
            return spans[i][0], spans[-1][1], len(spans) - 1
    return None


def plan_short(spans, total):
    """~70 s cut: title, roll-up, board read, first move, the human's move, a capture, close."""
    P = lambda name: (lambda s: s[2] == name)
    story = [dict(still=card(["Gambit", "BracketBot plays chess against you", "camera  ·  Stockfish  ·  its own arm"],
                             sub="MuJoCo simulation of the robot's CAD model"), target=5, caption=None,
                  narration="Gambit. Bracket Bot plays chess against you, with its camera, Stockfish, and its own arm.")]
    first_obs = next(i for i, s in enumerate(spans) if s[2] == "OBSERVING")
    story.append(dict(src=(0, spans[first_obs][0]), target=4, caption="Rolling up to the table",
                      narration="It rolls up to the table."))
    a, b, k = span_from_to(spans, P("OBSERVING"), lambda s: s[2] in ("THINKING", "WAITING_FOR_HUMAN"))
    story.append(dict(src=(a, b), target=7, caption="Reads the board: 4 corner markers → homography → 64 squares. 16 white, 16 black.",
                      narration="It reads the board with its head camera: corner markers, a homography, every square classified. Sixteen white, sixteen black."))
    a, b, k = span_from_to(spans, P("THINKING"), P("VERIFYING"))
    story.append(dict(src=(a, b), target=16, caption="Stockfish picks the move; the robot announces it and picks the piece by its shaft — watch the hand camera",
                      narration="Stockfish chooses. The robot says its move, turns to face the square, picks the piece by its shaft, and sets it down until it feels the board. Then it looks again before it trusts the position."))
    a, b, k2 = span_from_to(spans, P("VERIFYING"), P("WAITING_FOR_HUMAN"), after=k)
    a, b, k3 = span_from_to(spans, P("OBSERVING"), P("THINKING"), after=k2)
    story.append(dict(src=(a, b), target=7, caption="Your move: inferred from the change on the board, matched against every legal move",
                      narration="Your turn. It infers your move from what changed on the board, matched against every legal move. No piece recognition, no typing."))
    cap = next((i for i, s in enumerate(spans) if s[2] == "MOVING" and "captured" in s[3]), None)
    if cap is not None:
        a = spans[cap][0]
        b = next((s[0] for s in spans[cap + 1:] if s[2] == "VERIFYING"), spans[-1][1])
        story.append(dict(src=(a, b), target=16, caption="A capture: the taken piece goes to the tray first, then the attacker moves",
                          narration="A capture: the taken piece goes to the tray first, then the attacking piece moves. Every move is verified by the camera; a wrong board is rejected and explained."))
    story.append(dict(still=card(["Gambit", "Play a game against the robot.", "github.com/GautamM303/Gambit"],
                                 sub="Bracket Bot hackathon"), target=5, caption=None,
                      narration="Gambit. Come and play a game."))
    return story


def plan(spans, total):
    """The cut: list of dicts {src:(a,b) | still:img, target:s, caption, narration}."""
    P = lambda name: (lambda s: s[2] == name)
    story = []
    story.append(dict(still=card(["Bracket Gambit", "BracketBot plays chess against you",
                                  "camera  ·  Stockfish  ·  its own arm"], sub="Bracket Bot hackathon"),
                      target=7, caption=None,
                      narration="Bracket Gambit. Bracket Bot plays chess against a human, using its camera, Stockfish, and its own arm."))
    first_obs = next(i for i, s in enumerate(spans) if s[2] == "OBSERVING")
    story.append(dict(src=(0, spans[first_obs][0]), target=8, caption="The robot rolls up to the table (simulation of the real robot's CAD model)",
                      narration="Everything you see is the MuJoCo simulation of the robot's own CAD model. First, it rolls up to the table."))
    a, b, k = span_from_to(spans, P("OBSERVING"), lambda s: s[2] in ("THINKING", "WAITING_FOR_HUMAN"))
    story.append(dict(src=(a, b), target=9, caption="Setup check: head camera → 4 corner markers → homography → 64 squares. 16 white, 16 black.",
                      narration="It reads the board with its head camera: four corner markers give a homography, every square is classified from an empty-board reference, and the starting position is verified. Sixteen white, sixteen black."))
    a, b, k = span_from_to(spans, P("THINKING"), P("VERIFYING"))
    story.append(dict(src=(a, b), target=26, caption="Stockfish picks the move; the robot announces it, then picks the piece by its shaft and places it",
                      narration="Stockfish chooses the opening move. The robot announces it out loud, then picks the piece by its shaft, carries it, and sets it down within a millimetre."))
    a, b, k2 = span_from_to(spans, P("VERIFYING"), P("WAITING_FOR_HUMAN"), after=k)
    story.append(dict(src=(a, b), target=7, caption="Verification: the arm parks, the camera re-reads the board; the internal position advances only on a match",
                      narration="Then it parks the arm and looks again. The chess position is only updated once the camera confirms the board."))
    a, b, k3 = span_from_to(spans, P("OBSERVING"), P("THINKING"), after=k2)
    story.append(dict(src=(a, b), target=9, caption="The human moves and signals 'turn complete'. The move is inferred by matching the occupancy change against every legal move.",
                      narration="Now the human moves and says: my turn is complete. The robot infers the move by matching the change it sees against every legal move. No piece recognition needed."))
    cap = next((i for i, s in enumerate(spans) if s[2] == "MOVING" and "captured" in s[3]), None)
    if cap is not None:
        a = spans[cap][0]
        b = next((s[0] for s in spans[cap + 1:] if s[2] == "VERIFYING"), spans[-1][1])
        story.append(dict(src=(a, b), target=30, caption="A capture: the taken piece goes to the tray first, then the attacker moves",
                          narration="A capture. The taken piece is carried to the tray first; only then does the attacking piece move."))
        montage_start = b
    else:
        montage_start = spans[k3][1]
    a = montage_start
    b = next((s[0] for s in spans if s[2] == "GAME_OVER"), spans[-1][1])
    if b > a + FPS * 5:
        story.append(dict(src=(a, b), target=28, caption="Turn after turn: observe → infer → think → announce → move → verify",
                          narration="And so it goes, turn after turn: observe, infer, think, announce, move, verify. Every move is spoken, and asking why gets a short explanation grounded in Stockfish's evaluation."))
    story.append(dict(still=cv2.resize(cv2.imread(str(HERE / "rejected_move.png")), (W, H)), target=11,
                      caption="Fails safely: an illegal or ambiguous board is rejected with the offending squares named; failed grasps retry, then ask for help; e-stop cuts torque within one control tick",
                      narration="It fails safely. If the board does not show a legal move, the robot names the squares that are wrong and waits. A failed grasp is retried, then it asks for a hand. And an emergency stop, from the keyboard, the web page, or control C, cuts torque within one control tick."))
    story.append(dict(still=checklist_card("Bracket Bot capabilities used", [
        ("+", "Head camera: board reading, move inference, verification"),
        ("+", "7-DOF arm + gripper: pick and place, captures to a tray"),
        ("+", "Speaker: announces moves, check, help requests"),
        ("+", "Wake word: 'my turn is complete' hands-free"),
        ("+", "LED: waiting / looking / thinking / moving / needs help"),
        ("+", "Stockfish 19 over UCI, adjustable strength, grounded 'why'"),
        ("~", "Navigation: simulation only (no base-drive channel in the SDK)"),
        ("~", "Real-arm backend written on the daemon API, hardware-unverified"),
        ("-", "Promotion, gestures with the arms, left arm: omitted on purpose")]), target=13, caption=None,
        narration="What we built on: the head camera, one seven-degree-of-freedom arm and gripper, the speaker, the wake word, and the LED. Navigation is simulation only, and the real-arm backend is written on the daemon API but not yet run on hardware. We say so, rather than pretend."))
    a = next((s[0] for s in spans if s[2] == "GAME_OVER"), None)
    if a is not None and total - a > FPS:
        story.append(dict(src=(a, total), target=5, caption="End of the demo", narration="Thank you for the game."))
    story.append(dict(still=card(["Bracket Gambit", "Play a game against the robot.",
                                  "gambit.py  ·  bracket_gambit/  ·  30 tests  ·  MuJoCo + Stockfish 19"], sub="github: gambit.py + bracket_gambit/"),
                      target=7, caption=None, narration="Bracket Gambit. Come and play a game."))
    return story


# --------------------------------------------------------------------------- #
def render(video, timeline, out, narrate=True, short=False):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    spans = find_segments(json.load(open(timeline)), total)
    story = (plan_short if short else plan)(spans, total)
    tmp = HERE / "_narration"
    tmp.mkdir(exist_ok=True)
    # narration first, so each segment is at least as long as its sentence
    for i, seg in enumerate(story):
        seg["audio"] = None
        if narrate and seg.get("narration"):
            p = tts_wav(seg["narration"], tmp / f"n{i:02d}.wav")
            if p is not None:
                seg["audio"] = load_wav(p)
                seg["target"] = max(seg["target"], len(seg["audio"]) / SR + 0.8)
    writer = imageio.get_writer(str(out.with_suffix(".silent.mp4")), fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    track = []
    t_total = 0.0
    # The source segments are in chronological order, so the video is decoded in a single
    # forward pass (seeking an H.264 file per frame is prohibitively slow).
    cur = -1
    last = None

    def frame_at(want):
        nonlocal cur, last
        while cur < want:
            ok, fr = cap.read()
            if not ok:
                break
            cur += 1
            last = fr
        return last

    for seg in story:
        n_out = int(round(seg["target"] * FPS))
        if "still" in seg:
            frame = caption(seg["still"], seg.get("caption"), tag=None)
            for _ in range(n_out):
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            a, b = seg["src"]
            n_src = max(1, b - a)
            speed = n_src / n_out
            text = seg.get("caption", "") + (f"   [{speed:.1f}x]" if speed > 1.3 else "")
            prev_want, frame = None, None
            for j in range(n_out):
                want = max(a, min(b - 1, int(a + j * speed)))
                if want != prev_want:
                    src = frame_at(want)
                    frame = caption(cv2.resize(src, (W, H)), text)
                    prev_want = want
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(f"  segment {seg.get('caption', '')[:50]!r}: {n_src / src_fps:.0f}s source -> {n_out / FPS:.0f}s ({speed:.1f}x)", flush=True)
        silence = np.zeros(int(round(seg["target"] * SR)), np.float32)
        if seg["audio"] is not None:
            silence[:min(len(seg["audio"]), len(silence))] = seg["audio"][:len(silence)]
        track.append(silence)
        t_total += seg["target"]
    writer.close()
    cap.release()
    print(f"video {t_total:.0f} s")
    audio = np.concatenate(track)
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(audio, -32768, 32767).astype(np.int16).tobytes())
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out.with_suffix(".silent.mp4")), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], check=True)
    out.with_suffix(".silent.mp4").unlink()
    wav.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=pathlib.Path, required=True)
    ap.add_argument("--timeline", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "Bracket_Gambit_demo.mp4")
    ap.add_argument("--no-narration", action="store_true")
    ap.add_argument("--short", action="store_true", help="~70 s cut instead of the 3-minute one")
    args = ap.parse_args()
    render(args.video, args.timeline, args.out, narrate=not args.no_narration, short=args.short)
