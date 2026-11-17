# Real virtual-camera test: step-by-step guide

This test answers one question: **does the injection detector catch a video
that is fed through a real virtual-camera driver (OBS Virtual Camera), and
does it leave real cameras alone?** Until now the injection results were an
emulation; this replaces it with a real driver.

You record two kinds of clips on this Mac, with one script:

| | What is in front of the camera | Who does it | Clips |
|---|---|---|---|
| **Genuine** | A real person, live, sitting in front of the built-in FaceTime camera | You and 2-3 teammates | 10 per person |
| **Attack** | Nobody. OBS plays a video file into its virtual camera | Automatic, while you watch | 30 per condition, 5 conditions |

Both are captured the same way (640x480, 30 fps, 4 s per clip, the same 16
frames the real client sends), so any difference is due to the injection,
not to how the clips were taken.

**You do not send me anything.** The clips are saved in this project folder
(`datasets/vcam/clips/`), which I can read directly. That folder is
excluded from git, so faces are never uploaded to GitHub.

**Time:** about 30-60 min of setup, 15 min per person for the genuine clips,
about 20 min for the attack clips (it runs by itself).

---

## Before you start

- Ask each teammate who will be recorded whether they agree. Their clips
  stay on this Mac, are used only for this evaluation, are never
  published, and can be deleted afterwards. Use labels `p1`, `p2`, ...,
  never real names.
- Plug the Mac into power. Every recording command below starts with
  `caffeinate -i`, which stops the Mac from sleeping during a recording
  (a sleeping Mac broke one of our earlier runs).
- Open a terminal **in this project folder** and activate the environment:

  ```bash
  cd ~/Downloads/projects/AI-Based-Biometric-Evidence-Authentication-for-Post-Quantum-Digital-Identity-Systems/AI-Based-Biometric-Evidence-Authentication-for-Post-Quantum-Digital-Identity-Systems
  source .venv/bin/activate
  ```

  Use the same terminal app (Terminal or the VS Code terminal) for every
  step: macOS grants camera access per app.

The attack videos that OBS will play are already built in
`datasets/vcam/attack_videos/` (the 10 bona fide people of the public
webcam sample, standing in for a victim's recordings):

| File | Simulates an attacker's video file that is |
|---|---|
| `obs_25fps_crf23.mp4` | 25 fps, normal quality (PAL-style recording) |
| `obs_24fps_crf28.mp4` | 24 fps, low quality (film rate, heavily compressed) |
| `obs_30fps_crf23.mp4` | 30 fps, normal quality (same rate as the camera) |
| `obs_30fps_crf18.mp4` | 30 fps, high quality |
| `obs_30fps_lossless.mp4` | 30 fps, lossless (the hardest case) |

(If you ever need to rebuild them: `python evaluation/virtual_camera_capture.py make-videos`.)

---

## Step 1. Install OBS Studio

Either with Homebrew:

```bash
brew install --cask obs
```

or download the macOS (Apple Silicon) installer from
<https://obsproject.com/download>, open the `.dmg` and drag OBS to
Applications.

Open OBS from Applications. If macOS says it is from the internet, click
**Open**. If the **Auto-Configuration Wizard** appears, choose
**"Optimize just for the virtual camera"**, click Next, and accept the
defaults.

## Step 2. Set OBS to the camera's format

1. **OBS > Settings** (or **Settings** at the bottom right) > **Video**.
2. **Base (Canvas) Resolution:** type `640x480`.
3. **Output (Scaled) Resolution:** type `640x480`.
4. **Common FPS Values:** `30`.
5. Click **Apply**, then **OK**.

## Step 3. Add the attack video as a source

1. In the **Sources** box (bottom left), click **+** > **Media Source** >
   name it `attack` > **OK**.
2. Click **Browse**, open the project folder, go to
   `datasets/vcam/attack_videos/` and choose **`obs_25fps_crf23.mp4`**.
3. Tick **Loop**. Leave the other boxes as they are. Click **OK**.
4. Right-click the source in the preview > **Transform** > **Fit to
   screen**. The preview should show a person's face filling the frame.

## Step 4. Start the virtual camera (one-time permission)

1. Click **Start Virtual Camera** (Controls box, bottom right).
2. The first time, macOS blocks the camera extension. Open **System
   Settings > Privacy & Security**, scroll down to the message about
   **OBS** (or "system software ... blocked"), click **Allow**, enter your
   password.
3. If OBS asks, quit and reopen it, then click **Start Virtual Camera**
   again. When it works, the button reads **Stop Virtual Camera**.

## Step 5. Find the camera numbers

With the virtual camera **started**, run:

```bash
python evaluation/virtual_camera_capture.py cameras
```

- The first time, macOS asks whether your terminal app may use the camera:
  click **OK / Allow**. If you clicked Don't Allow before: **System
  Settings > Privacy & Security > Camera**, switch on Terminal (or Visual
  Studio Code), quit the app and reopen it, and run the command again.
- It prints lines such as `camera 0: 640x480 preview saved to
  datasets/vcam/camera_previews/camera_0.png`. Open those images:
  - the one showing **you** is the **FaceTime camera** (write down its number);
  - the one showing **the video's face** is **OBS Virtual Camera** (write down its number).
- Your iPhone may also appear (Continuity Camera). Ignore it, or turn it
  off on the iPhone: Settings > General > AirPlay & Continuity > Continuity
  Camera.

Below, replace `F` with the FaceTime number and `V` with the OBS number.

## Step 6. Record the genuine clips (live, in front of the camera)

Yes, this part is **live**: each person sits in front of the Mac. Quit
OBS first (it is not needed and it uses CPU).

For each person (`p1`, `p2`, ...), run two commands, 5 clips each:

```bash
caffeinate -i python evaluation/virtual_camera_capture.py record --camera F --kind genuine --person p1 --lighting normal --clips 5
caffeinate -i python evaluation/virtual_camera_capture.py record --camera F --kind genuine --person p1 --lighting dim --clips 5
```

For every clip the script does this:

1. It says `Clip 1/5 ... press Enter`. The person sits about 50 cm away,
   faces the camera, and someone presses **Enter**.
2. Countdown `3... 2... 1...`, then **RECORDING**.
3. **Look straight at the camera** for about one second.
4. A prompt appears in capitals: **TURN YOUR HEAD SLIGHTLY LEFT**, **...RIGHT**
   or **NOD YOUR HEAD ONCE**. Do it clearly, then look back. The clip ends
   about 3 s later and is saved.

Tips: the whole face visible, one face only, no hand in front of the face.
Glasses are fine (mix with and without if you can). For `dim`, switch off
the main light or sit with a window to the side; it does not need to be
dark. A first clip that went wrong is not a problem: just keep going, or
record extra clips (the script adds to what is there).

Target: **3-4 people x 10 clips = 30-40 genuine clips.**

## Step 7. Record the attack clips (automatic, nobody in front of the camera)

Open OBS again and start the virtual camera. For **each** of the five
conditions:

1. In OBS, double-click the `attack` source > **Browse** > choose the
   file for that condition > **OK** (Loop stays ticked). Check that the
   preview is playing.
2. Make sure **Stop Virtual Camera** is showing (that is, it is running).
3. Run the matching command and leave the Mac alone for about 3 minutes:

```bash
caffeinate -i python evaluation/virtual_camera_capture.py record --camera V --kind attack --condition obs_25fps_crf23 --clips 30
```

| Load this file in OBS | Then run with `--condition` |
|---|---|
| `obs_25fps_crf23.mp4` | `obs_25fps_crf23` |
| `obs_24fps_crf28.mp4` | `obs_24fps_crf28` |
| `obs_30fps_crf23.mp4` | `obs_30fps_crf23` |
| `obs_30fps_crf18.mp4` | `obs_30fps_crf18` |
| `obs_30fps_lossless.mp4` | `obs_30fps_lossless` |

**The file loaded in OBS must match `--condition`.** If you notice a
mismatch, delete that folder (`datasets/vcam/clips/attack/<condition>`)
and record it again.

**Optional, but valuable: a real attacker's video.** Film a consenting
teammate on a phone for about 30 s (looking at the phone, turning the head
now and then), AirDrop it to the Mac, load it in OBS the same way, and
run:

```bash
caffeinate -i python evaluation/virtual_camera_capture.py record --camera V --kind attack --condition custom_phone_video --clips 20
```

## Step 8. Check what you recorded

```bash
python evaluation/virtual_camera_capture.py check
```

You should see something like:

```
genuine  camera                 p1      10 clips, median 30.0 fps
genuine  camera                 p2      10 clips, median 29.9 fps
attack   obs_25fps_crf23        obs     30 clips, median 30.0 fps
...
```

If the genuine fps is much lower than 30 (for example 15) in dim light,
that is normal for the FaceTime camera; note it and carry on.

## Step 9. Tell me "clips recorded"

That is all I need. I will then:

1. run `python evaluation/virtual_camera_capture.py evaluate`. It scores
   every clip with the injection detector, then sends every clip through
   the final server twice: once with the typed-digit challenge (so all five
   detectors decide) and once with the head-motion challenge as deployed;
2. compare the real-OBS results with the emulation in Section 4.3;
3. update the paper (Section 4.3, Tables 8 and 10, limitations) and the
   README, and push the script and the summary numbers (not the clips).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` or `opens but gives no frames` | Camera permission for your terminal app: System Settings > Privacy & Security > Camera, switch it on, quit and reopen the terminal app |
| OBS camera index missing in `cameras` | Start the virtual camera in OBS first, then run `cameras` again |
| OBS preview black | Source not fitted: right-click > Transform > Fit to screen; or the file path is wrong (double-click the source and browse again) |
| `WARNING: frames look frozen` | OBS is paused or the video ended: tick Loop, click the play button on the source |
| "System extension blocked" | System Settings > Privacy & Security > Allow for OBS, then restart OBS (sometimes the Mac) |
| The Mac went to sleep during a recording | Always start record commands with `caffeinate -i`; delete the incomplete clip folder and record it again |
| Wrong person label or a bad clip | Delete that clip folder in `datasets/vcam/clips/...` and record more |

**Deleting everything afterwards:** `rm -rf datasets/vcam/clips` removes all
recorded faces. The results in `evaluation/results/vcam_*` contain only
scores and counts, no images.
