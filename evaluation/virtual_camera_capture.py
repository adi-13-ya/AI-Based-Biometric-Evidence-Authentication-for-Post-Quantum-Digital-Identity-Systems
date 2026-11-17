"""Real virtual-camera injection test (OBS Virtual Camera on macOS).

Genuine clips come from a real camera with a person in front of it; attack
clips come from OBS Virtual Camera playing a video file. Both are captured
the same way (OpenCV, AVFoundation, 640x480 at 30 fps, 4 s per clip, the
same 8 spread + 8 consecutive frames the live client sends), so a
difference between them is due to the injection, not the capture path.
Step-by-step guide: docs/VIRTUAL_CAMERA_TEST.md.

Commands
  make-videos   build the attack videos that OBS plays (one per condition)
  cameras       list camera indexes and save a preview image of each
  record        record clips from one camera (genuine: person in front of the
                camera, guided prompts; attack: OBS playing a video, automatic)
  check         count the recorded clips per condition and person
  evaluate      score every clip through the final server (all detectors, and
                the head-motion challenge) and write the results

Clips are saved under datasets/vcam/ (gitignored: faces are never committed).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report" / "scripts"))

VCAM = REPO / "datasets" / "vcam"
VIDEOS = VCAM / "attack_videos"
CLIPS = VCAM / "clips"
RESULTS = REPO / "evaluation" / "results"
WIDTH, HEIGHT, FPS = 640, 480, 30
SECONDS = 4.0
LOOK_STRAIGHT = 1.2
PROMPTS = ["Turn your head slightly left", "Turn your head slightly right", "Nod your head once"]

# Attack conditions: the attacker's video file, played by OBS into a 30 fps
# virtual camera. Same file settings as the emulated benchmark (Section 4.3).
CONDITIONS = {
    "obs_25fps_crf23": (25, 23),
    "obs_24fps_crf28": (24, 28),
    "obs_30fps_crf23": (30, 23),
    "obs_30fps_crf18": (30, 18),
    "obs_30fps_lossless": (30, 0),
}
SEGMENT_SECONDS = 6


# --------------------------------------------------------------------------- make-videos
def make_videos(args) -> None:
    """One looping MP4 per condition: the first 6 s of each of the 10 bona
    fide webcam videos of the public webcam sample (people who are not the
    test subjects, i.e. a victim's recordings), re-encoded at the
    condition's frame rate and quality."""
    import av

    from evaluation.train_injection import decode

    VIDEOS.mkdir(parents=True, exist_ok=True)
    sources = sorted((REPO / "datasets" / "webcam_liveness" / "files" / "real").glob("*.mp4"))
    if not sources:
        sys.exit("datasets/webcam_liveness/files/real/*.mp4 not found (run evaluation/fetch_data.py first)")
    for name, (fps, crf) in CONDITIONS.items():
        out_path = VIDEOS / f"{name}.mp4"
        with av.open(str(out_path), mode="w") as out:
            stream = out.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
            stream.options = {"qp": "0", "preset": "veryfast"} if crf == 0 else {"crf": str(crf), "preset": "veryfast"}
            for src in sources:
                frames = decode(src, max_frames=SEGMENT_SECONDS * fps)
                for f in frames:
                    f = np.ascontiguousarray(_fit(f))
                    for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                        out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        print(f"{out_path.relative_to(REPO)}  ({fps} fps, {'lossless' if crf == 0 else f'CRF {crf}'}, "
              f"{out_path.stat().st_size / 2**20:.1f} MB, {len(sources)} people x {SEGMENT_SECONDS} s)")


def _fit(frame: np.ndarray) -> np.ndarray:
    """Centre-crop and resize to WIDTHxHEIGHT."""
    from PIL import Image

    h, w = frame.shape[:2]
    target = WIDTH / HEIGHT
    if w / h > target:
        nw = int(h * target)
        frame = frame[:, (w - nw) // 2: (w - nw) // 2 + nw]
    else:
        nh = int(w / target)
        frame = frame[(h - nh) // 2: (h - nh) // 2 + nh]
    return np.asarray(Image.fromarray(frame).resize((WIDTH, HEIGHT), Image.BILINEAR))


# --------------------------------------------------------------------------- capture
class Camera:
    """A real or virtual camera (int index), or a video file for dry runs."""

    def __init__(self, source: str):
        import cv2

        self.cv2 = cv2
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            self.live = True
        else:
            self.cap = cv2.VideoCapture(source)
            self.live = False
        if not self.cap.isOpened():
            sys.exit(f"Could not open camera {source!r}. On macOS: System Settings > Privacy & Security > Camera, "
                     "allow your terminal app (Terminal or Visual Studio Code), then restart it.")
        self.source = source

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if not self.live:  # loop the dry-run file
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
        return np.ascontiguousarray(frame[..., ::-1])  # BGR -> RGB

    def warm_up(self, seconds: float = 1.0) -> None:
        t = time.time()
        while time.time() - t < seconds:
            self.read()

    def clip(self, seconds: float = SECONDS) -> tuple[list[np.ndarray], list[float]]:
        frames, stamps = [], []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            f = self.read()
            if f is None:
                break
            frames.append(f)
            stamps.append(time.perf_counter() - t0)
            if not self.live:
                time.sleep(1 / FPS)
        return frames, stamps

    def close(self) -> None:
        self.cap.release()


def cameras(args) -> None:
    out = VCAM / "camera_previews"
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    print("Probing camera indexes 0-5 (macOS may ask for camera permission the first time)...")
    found = 0
    for i in range(6):
        import cv2

        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        frame = None
        for _ in range(30):
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            print(f"  camera {i}: opens but gives no frames (permission missing, or OBS Virtual Camera not started)")
            continue
        path = out / f"camera_{i}.png"
        Image.fromarray(frame[..., ::-1]).save(path)
        found += 1
        print(f"  camera {i}: {frame.shape[1]}x{frame.shape[0]}  preview saved to {path.relative_to(REPO)}")
    if not found:
        print("No camera gave frames. Allow camera access for your terminal app and try again.")
    else:
        print("Open the preview images to see which index is the FaceTime camera and which is OBS Virtual Camera.")


def save_clip(folder: Path, frames: list[np.ndarray], stamps: list[float], meta: dict) -> dict:
    from PIL import Image

    from client_demo.recorder import select_frames

    sent, burst = select_frames(frames)
    folder.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(sent):
        Image.fromarray(f).save(folder / f"frame_{k:02d}.png")
    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    identical = sum(1 for a, b in zip(frames, frames[1:]) if np.array_equal(a, b))
    meta = {**meta, "frames_captured": len(frames), "measured_fps": round(fps, 2), "identical_consecutive": identical,
            "frames_sent": len(sent), "burst_frames": burst, "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
            "timestamps": stamps}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def record(args) -> None:
    kind = args.kind
    if kind == "genuine" and not args.person:
        sys.exit("--person is required for genuine clips (a short label such as p1, never a real name)")
    cond = args.condition or ("camera" if kind == "genuine" else None)
    if kind == "attack" and cond not in CONDITIONS and not cond.startswith("custom_"):
        sys.exit(f"--condition must be one of {list(CONDITIONS)} or start with custom_ (for example custom_phone_video)")
    base = CLIPS / kind / cond / (args.person if kind == "genuine" else "obs")
    existing = len(list(base.glob("clip_*"))) if base.exists() else 0
    cam = Camera(args.camera)
    print(f"Camera {args.camera} open. Warming up...")
    cam.warm_up(1.5)
    try:
        for k in range(existing, existing + args.clips):
            meta = {"kind": kind, "condition": cond, "person": args.person, "camera": args.camera, "lighting": args.lighting,
                    "note": args.note, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            if kind == "genuine":
                prompt = random.choice(PROMPTS)
                meta["prompt"] = prompt
                input(f"\nClip {k + 1 - existing}/{args.clips}. Sit about 50 cm from the camera, face it, then press Enter...")
                for c in (3, 2, 1):
                    print(f"  {c}...", flush=True)
                    time.sleep(0.8)
                print(f"  RECORDING: look straight at the camera, then when you see the prompt: {prompt.upper()}", flush=True)
                t0 = time.time()
                frames_a, stamps_a = cam.clip(LOOK_STRAIGHT)
                print(f"  >>> {prompt.upper()} <<<", flush=True)
                frames_b, stamps_b = cam.clip(SECONDS - (time.time() - t0))
                frames, stamps = frames_a + frames_b, stamps_a + [s + stamps_a[-1] + 1 / FPS for s in stamps_b] if stamps_a else stamps_b
            else:
                meta["prompt"] = None
                print(f"Attack clip {k + 1 - existing}/{args.clips} ({cond})...", flush=True)
                frames, stamps = cam.clip(SECONDS)
                time.sleep(args.gap)
            if len(frames) < 20:
                print(f"  only {len(frames)} frames captured - is the camera running? Clip skipped.")
                continue
            m = save_clip(base / f"clip_{k:03d}", frames, stamps, meta)
            warn = "  WARNING: frames look frozen (OBS paused or nothing playing?)" if m["identical_consecutive"] > len(frames) * 0.8 else ""
            print(f"  saved {m['frames_captured']} frames at {m['measured_fps']} fps -> {base.relative_to(REPO)}/clip_{k:03d}{warn}")
    finally:
        cam.close()
    print("\nDone. Check progress with: python evaluation/virtual_camera_capture.py check")


def check(args) -> None:
    if not CLIPS.exists():
        print("No clips yet.")
        return
    for kind in ("genuine", "attack"):
        for cond in sorted(p.name for p in (CLIPS / kind).glob("*") if p.is_dir()) if (CLIPS / kind).exists() else []:
            for who in sorted(p.name for p in (CLIPS / kind / cond).glob("*") if p.is_dir()):
                clips = sorted((CLIPS / kind / cond / who).glob("clip_*/meta.json"))
                fps = [json.loads(c.read_text())["measured_fps"] for c in clips]
                print(f"{kind:8s} {cond:22s} {who:6s} {len(clips):3d} clips, median {statistics.median(fps) if fps else 0:.1f} fps")


# --------------------------------------------------------------------------- evaluate
def load_clips():
    from PIL import Image

    out = []
    for meta_path in sorted(CLIPS.glob("*/*/*/clip_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(meta_path.parent.glob("frame_*.png"))]
        out.append({**meta, "path": str(meta_path.parent.relative_to(REPO)), "frames": frames})
    return out


def evaluate(args) -> None:
    import malicious_client as mc

    from detection.injection_detector import InjectionDetector

    clips = load_clips()
    if not clips:
        sys.exit("No clips under datasets/vcam/clips - record some first.")
    counts = defaultdict(int)
    for c in clips:
        counts[f"{c['kind']}/{c['condition']}"] += 1
    print(f"{len(clips)} clips: {dict(counts)}")
    inj = InjectionDetector()
    from server.config import Settings

    thr = Settings.model_fields["threshold_injection"].default
    rows = []
    for c in clips:
        burst = c["burst_frames"]
        score = inj.predict_sequence(c["frames"][-burst:] if burst else c["frames"]).score
        rows.append({"path": c["path"], "kind": c["kind"], "condition": c["condition"], "person": c.get("person"),
                     "lighting": c.get("lighting"), "prompt": c.get("prompt"), "measured_fps": c["measured_fps"],
                     "injection_score": score, "injection_veto": score < thr})

    # End to end through the final server: (1) typed-digit challenge, so that
    # all five detectors decide; (2) head-motion challenge, as deployed.
    for label, extra in [("digits", None), ("motion", {"CHALLENGE_KINDS": "motion"})]:
        server = mc.Server("integrated", extra_env={**mc.ABLATIONS["final"], **(extra or {})})
        try:
            client = mc.new_client()
            for c, r in zip(clips, rows):
                client.reset_replay_store()
                if label == "motion" and c.get("prompt"):
                    s = None
                    for _ in range(40):  # the session whose prompt the person performed
                        s = client.session_start()
                        if s["challenge"] == c["prompt"]:
                            break
                        client.end_session(s)
                else:
                    s = mc.session_with(client, "digits" if label == "digits" else "motion")
                fb = [mc.png(f) for f in c["frames"]]
                status, body, _ = mc.post(s["session_id"], mc.envelope(client, s, fb, c["burst_frames"]))
                o = mc.outcome(status, body)
                r[f"{label}_accepted"] = o["accepted"]
                r[f"{label}_failed"] = o["binding_failed"] or o["vetoes"]
                if label == "motion":
                    r["liveness_passed"] = o["liveness_passed"]
        finally:
            server.stop()

    RESULTS.mkdir(parents=True, exist_ok=True)
    import csv

    with open(RESULTS / "vcam_clips.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from evaluation.metrics import wilson_interval

    summary = {}
    for (kind, cond), rs in sorted(_group(rows, ("kind", "condition")).items()):
        n = len(rs)
        veto = sum(r["injection_veto"] for r in rs)
        e = {"kind": kind, "n": n, "people": len({r["person"] for r in rs if r["person"]}),
             "injection_vetoed": veto, "digits_accepted": sum(r["digits_accepted"] for r in rs),
             "motion_liveness_passed": sum(bool(r["liveness_passed"]) for r in rs), "motion_accepted": sum(r["motion_accepted"] for r in rs),
             "median_fps": statistics.median(r["measured_fps"] for r in rs)}
        rate = (n - veto) / n if kind == "attack" else veto / n
        lo, hi = wilson_interval(n - veto if kind == "attack" else veto, n)
        e["injection_apcer" if kind == "attack" else "injection_bpcer"] = {"value": rate, "ci95": [lo, hi]}
        summary[cond] = e
    emu = RESULTS / "injection_eval.json"
    if emu.exists():
        em = json.loads(emu.read_text())["classifier"]
        summary["_emulation_for_comparison"] = {"apcer_per_condition": em["apcer_per_condition"], "bpcer": em["bpcer"]}
    (RESULTS / "vcam_summary.json").write_text(json.dumps(summary, indent=1))
    for cond, e in summary.items():
        if not cond.startswith("_"):
            k = "injection_apcer" if e["kind"] == "attack" else "injection_bpcer"
            print(f"{e['kind']:8s} {cond:22s} n={e['n']:3d}  {k}={e[k]['value']:.1%} [{e[k]['ci95'][0]:.1%}, {e[k]['ci95'][1]:.1%}]  "
                  f"accepted (digits) {e['digits_accepted']}/{e['n']}  motion: liveness {e['motion_liveness_passed']}/{e['n']}, accepted {e['motion_accepted']}/{e['n']}")
    print("Wrote evaluation/results/vcam_clips.csv and vcam_summary.json")


def _group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-videos").set_defaults(fn=make_videos)
    sub.add_parser("cameras").set_defaults(fn=cameras)
    r = sub.add_parser("record")
    r.add_argument("--camera", required=True, help="camera index from 'cameras' (or a video file path for a dry run)")
    r.add_argument("--kind", required=True, choices=["genuine", "attack"])
    r.add_argument("--condition", help="attack condition (see make-videos); genuine clips use 'camera'")
    r.add_argument("--person", help="genuine only: anonymous label such as p1")
    r.add_argument("--clips", type=int, default=10)
    r.add_argument("--lighting", default="normal", help="free text, e.g. normal, dim, window")
    r.add_argument("--note", default="")
    r.add_argument("--gap", type=float, default=1.5, help="attack only: seconds between clips")
    r.set_defaults(fn=record)
    sub.add_parser("check").set_defaults(fn=check)
    sub.add_parser("evaluate").set_defaults(fn=evaluate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
