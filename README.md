# Pokémon Card Grader

Films a short **tilt video** of a card and outputs a PSA-scale grade. One command,
no API keys, runs offline on CPU.

Why video and not a photo: card defects only separate from glare and foil shimmer
under motion. Tilting the card sweeps the glare across every spot (so each point is
glare-free in *some* frame), foil shimmer reveals itself as foil, and real defects
(scratch, crease, stain, whitening) hold still across frames. The grader takes the
per-pixel median across frames (a clean, glare-free composite) and looks for what
moves vs. what stays.

## Install

```bash
pip install -r requirements.txt
```

That's it — numpy, opencv, scikit-image. No GPU, no API key.

## 1. Record the input

One **~8-second tilt video per face**. Short version:

- Rest the card flat on something **rigid you can tilt** (a hardback book, a
  clipboard) on a **matte, dark** background. Rest it — don't grip it — so all four
  corners stay visible.
- **One** light source, off to one side (you *want* a glare spot to sweep).
- Phone **steady** on the **main 1× lens** (not ultra-wide), card centered at
  ~60–70% of frame, all four corners in view with a margin.
- **Lock focus & exposure** (iPhone: long-press until "AE/AF LOCK").
- **Tilt the card, not the phone**, slowly in a "plus": far edge down/up, then
  left edge up / right edge up. ~20–30°, go slow enough that no frame is blurry.
- 1080p, 30fps. No flash, no sleeve/toploader.

Full protocol with the reasoning: **[CAPTURE.md](CAPTURE.md)**.

## 2. Input format

Put the videos in a folder per card, named by face:

```
mycard/
  front.mp4        # or front.MOV
  back.mp4         # optional; or back.MOV
```

`.mp4` and `.MOV` both work. The folder name is yours; `front`/`back` filenames matter.

## 3. Run

```bash
python grade.py mycard --face front
python grade.py mycard --face back
```

Output (leads with the PSA grade):

```
  PSA grade:    1.5   (limiting: surface)
  Score /1000:  261.1  (-> grade 2.5)
  centering : 10
  corners   : 2
  edges     : 2
  surface   : 1

Debug: mycard/debug/front_temporal
```

- **PSA grade** = the limiting (worst) pillar, PSA-style.
- **Score /1000** = a finer weighted blend across the four pillars.
- The `debug/` folder gets the glare-free median frame, the foil map, and a
  defect-saliency overlay so you can see what it flagged.

Backs automatically use the bundled canonical card-back reference
(`references/pokemon_back.jpg`) — nothing to set up.

## Honest limitations

- **Flat, matte marks/stains can be missed.** The model sees defects that change
  with tilt (creases, scratches that glint); a perfectly flat ink mark looks the
  same at every angle and can blend into the median. Creases, scratches, edge
  whitening, and corner wear are the strong suit.
- **Validated on one card so far** (a reverse-holo Fearow, front + back) against
  hand-labeled ground truth: front catches all annotated defects, back catches all
  but the faint flat marks above. Generalization across other cards is untested —
  this is an early prototype.
