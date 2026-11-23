"""
Transcribe proof-of-concept reels so the daily reel task can copy the hook word for word
and follow the script beat by beat.

Runs after build.py in the daily GitHub Action. For each outlier reel (good enough to be a
proof of concept) that has no transcript yet, it downloads the video from the temporary
link Apify returned, transcribes the audio with faster-whisper (free, runs on the runner),
and reads the on-screen text from the first seconds with Tesseract OCR.

Output: data/transcripts.json  {shortCode: {...}}
"""
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = ["build.py"]
import build  # noqa: E402  (reuses the dashboard's own outlier score)

OUT = ROOT / "data" / "transcripts.json"
MIN_SCORE = float(os.environ.get("TX_MIN_SCORE", "2.0"))
MIN_VIEWS = int(os.environ.get("TX_MIN_VIEWS", "250000"))
MAX_PER_RUN = int(os.environ.get("TX_MAX_PER_RUN", "25"))
BUDGET_S = int(os.environ.get("TX_BUDGET_SECONDS", "900"))
MODEL = os.environ.get("TX_MODEL", "small.en")
OCR_TIMES = [0.3, 1.0, 2.0, 3.5]


def log(msg):
    print(f"[transcribe] {msg}", flush=True)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def download(url, target):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
        f.write(r.read())
    return os.path.getsize(target) > 10_000


def ocr_frames(video, workdir):
    texts = []
    for t in OCR_TIMES:
        img = os.path.join(workdir, f"f{t}.png")
        if run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1",
                "-vf", "scale=1080:-2,format=gray", img]).returncode != 0 or not os.path.exists(img):
            continue
        r = run(["tesseract", img, "stdout", "--psm", "11"])
        lines = [ln.strip() for ln in r.stdout.splitlines() if len(ln.strip()) >= 3]
        texts.append({"t": t, "text": " / ".join(lines)[:400]})
    return texts


def main():
    history = json.loads(build.HISTORY_PATH.read_text())
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    scored = build.score_posts(history)
    cutoff = build.NOW - dt.timedelta(days=21)
    todo = []
    for p in scored:
        if p.get("kind") != "reel" or p["code"] in done or not p.get("video"):
            continue
        if build.parse_ts(p["posted"]) < cutoff:
            continue
        good = (p.get("score") or 0) >= MIN_SCORE or (p.get("views") or 0) >= MIN_VIEWS
        if good:
            todo.append(p)
    todo.sort(key=lambda p: (p.get("score") or 0), reverse=True)
    todo = todo[:MAX_PER_RUN]
    log(f"{len(todo)} new proof-of-concept reels to transcribe ({len(done)} already done)")
    if not todo:
        return

    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    start = time.time()
    for p in todo:
        if time.time() - start > BUDGET_S:
            log("time budget used up; the rest will be done next run")
            break
        with tempfile.TemporaryDirectory() as wd:
            vid = os.path.join(wd, "v.mp4")
            try:
                if not download(p["video"], vid):
                    raise RuntimeError("empty download")
            except Exception as e:  # link expired or blocked
                log(f"skip {p['code']} (@{p['owner']}): download failed: {e}")
                continue
            wav = os.path.join(wd, "a.wav")
            has_audio = run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-vn", "-ac", "1", "-ar", "16000", wav]).returncode == 0
            segs = []
            if has_audio and os.path.exists(wav):
                pieces, _ = model.transcribe(wav, vad_filter=True, beam_size=1)
                segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in pieces]
            onscreen = ocr_frames(vid, wd)
            hook_spoken = " ".join(s["text"] for s in segs if s["start"] < 3.0).strip()
            done[p["code"]] = {
                "url": p["url"], "owner": p["owner"], "group": p.get("group"),
                "views": p.get("views"), "score": p.get("score"), "duration": p.get("duration"),
                "hook_spoken": hook_spoken,
                "hook_onscreen": onscreen[0]["text"] if onscreen else "",
                "onscreen": onscreen,
                "transcript": " ".join(s["text"] for s in segs).strip(),
                "segments": segs,
                "transcribed_at": build.NOW.isoformat(),
            }
            log(f"ok {p['code']} @{p['owner']} score {p.get('score')} — {len(segs)} segments")
            OUT.write_text(json.dumps(done, indent=1))  # save as we go

    # keep the file small: drop transcripts of posts that have left the history window
    keep = set(history["posts"])
    done = {c: v for c, v in done.items() if c in keep}
    OUT.write_text(json.dumps(done, indent=1))
    log(f"transcripts.json now holds {len(done)} reels")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never break the dashboard refresh
        log(f"FAILED: {e}")
    sys.stdout.flush()
    os._exit(0)
