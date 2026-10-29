# GNM for TouchDesigner

[![Download .tox](https://img.shields.io/badge/%E2%AC%87%EF%B8%8F_DOWNLOAD-google__gnm__model.tox-2ea44f?style=for-the-badge)](https://github.com/IsaacSante/gnm-touchdesigner/releases/latest/download/google_gnm_model.tox)

[Google GNM Head](https://github.com/google/GNM) — a parametric 3D human head
model learned from real 3D scans — as a single, self-contained TouchDesigner
component. Drag one `.tox` into your project and you get a fully controllable
head: **253 identity + 383 expression parameters and a 4-joint pose rig
(neck, head, eyes), all evaluated on the GPU in real time.**

New in v1.1: **face tracking** from your webcam (via Torin Blankensmith's
MediaPipe plugin), **emotion detection** with an optional emotion recolor,
**anatomical colors** (skin, lips, eyes, teeth, tongue), and an **Explore**
page with random faces and an auto-explore timer.

## Download

Grab **`google_gnm_model.tox`** from the
[latest release](../../releases/latest) — that one file is everything
(mesh, morph data, and rig are embedded inside it).

## Requirements

- TouchDesigner **2025 official build or newer** (uses POPs). Tested on
  2025.33070, macOS, including the Non-Commercial license.
- For face tracking only: Torin Blankensmith's
  [MediaPipe TouchDesigner plugin](https://github.com/torinmb/mediapipe-touchdesigner)
  (tested with v0.6.0). The head works without it.

## Use

1. Drag `google_gnm_model.tox` into your network.
2. Wait a couple of seconds on first load (the morph data decompresses once).
3. Select the component and press **P**:
   - **Explore** — Random Face, Random Expression, Reset Face, Variety (how
     wild the random faces get), Seed (the same seed always gives the same
     face), Face / Expression Strength (0 = average, 2 = caricature), and
     Auto Explore (a new random face and/or expression every N seconds)
   - **Face Tracking** — drive the head from your face (see below)
   - **Skin Colors** — colors for skin, lips, iris, pupil, sclera, teeth, gums
     and tongue (on by default)
   - **Emotion Colors** — Emotion Recolor (off by default) tints the skin with
     the detected emotion's color; one color per emotion
   - **Default** — a starter set of the highest-impact sliders
   - **Identity** — who the person is (head shape, teeth, eyeballs), grouped
     by region
   - **Expression** — what the face is doing (eyes, lower face, tongue, iris)
   - **Pose** — neck/head rotation and per-eye gaze
   - **Translation** — head position
4. All sliders run −3 … +3 (0 = neutral/average). The first sliders in each
   group make the biggest changes; later ones are progressively subtler.

Inside the component, the full 636-channel control stream is exposed in CHOP
form (`override` / `to_samples`) with canonical channel names
(`i_head_000` … `e_tongue_031`) — drive any subset by name for face tracking,
audio reactivity, or animation.

### Outputs

1. **Geometry** (POP) — the head, with a `Color` point attribute. Use a MAT
   that shows point colors, or ignore it.
2. **`emotion_out`** (CHOP) — `neutral happy surprise angry sad fear` scores
   (0..1) and `dominant_id` (0 = neutral … 5 = fear). Channel names never
   change.
3. **`emotion_label_out`** (DAT) — the dominant emotion's name as text.

### Face tracking

1. Install Torin Blankensmith's MediaPipe plugin and wire `MediaPipe` into its
   `face_tracking` component, as in his examples. Set its number of faces to 1.
2. On the head's **Face Tracking** page, set **MediaPipe Face Tracking** to
   that `face_tracking` component.
3. Look at the camera with a relaxed face and press **Calibrate Neutral Face**.
   This also fits the head's identity to your face (turn off **Apply
   Identity** to keep the slider face) and makes emotion detection work for
   your resting expression.

Expression Gain, Smoothing, Gaze Gain and Max Coefficient tune the tracking;
the tracked face adds on top of the sliders. Emotions come from MediaPipe's
face blendshapes, plus head tilt for fear; raise **Emotion Gain** if they're
hard to trigger.

### Driving parameters from CHOPs

Reference a CHOP channel in a slider's expression, or use CHOP exports. If you
delete a controlling CHOP, reset the parameter afterwards (right-click →
Reset) — a leftover reference to a deleted operator will error.

## Notes & limitations

- GNM's pose-corrective shapes are not applied, so extreme neck/eye rotations
  deviate slightly from Google's reference implementation.
- The model was trained on binary gender categories and four broad demographic
  groups; see Google's own
  [fairness notes](https://github.com/google/GNM/blob/main/gnm/shape/README.md)
  before using it in applications where representation matters.
- Textures/materials are not included — the component outputs geometry (with
  UVs and point colors) for your own render chain.
- Emotion detection is a simple heuristic on facial movement, tuned on few
  faces. It reads expressions, not feelings — don't use it to judge people.

## Credits & license

- Model data: [Google GNM](https://github.com/google/GNM), Apache License 2.0.
  The `.tox` embeds a converted copy of the GNM Head v3.0 model data,
  redistributed under that license — see `LICENSE` and `NOTICE.md`.
- TouchDesigner integration: Isaac (this repository), Apache License 2.0.
- Face tracking input: Torin Blankensmith's
  [MediaPipe TouchDesigner plugin](https://github.com/torinmb/mediapipe-touchdesigner)
  (MIT), which you install separately — it is not included in the `.tox`.
- Not affiliated with Google or Derivative.
