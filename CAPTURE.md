# Card capture protocol (video)

A grade is only as good as the capture. We film a short **tilt video** — not a
single photo — because the physics of card defects only separate under **motion**:

- **Glare moves** as the card tilts, so every spot is glare-free in *some* frame →
  we see past highlights (e.g. a crease hidden under glare in a still).
- **Holo / foil shimmer changes with angle** → flagged as "foil behaving like
  foil," not damage.
- **Real defects (scratch, crease, stain, whitening) hold still** → they persist
  across frames and stand out.
- **Creases pop under grazing light** → a tilt sweep naturally hits that angle.

Keep it simple. ~8 seconds per face. **Do this for both faces of every card** —
it's one rule with no judgement calls, and the grader automatically does less work
on a plain matte face (it just uses the steady median frame).

## Setup
1. **Rest the card on a flat, RIGID surface you can tilt** — a hardback book, a box
   lid, a clipboard, or a glass — on a **matte, dark, non-reflective** background.
   *Rest* it, don't grip it: keeps the card flat and your fingers off the corners
   (the detector needs all four corners visible).
   - **Rigid matters:** a soft surface (your hand/palm) bows the card, which smears
     the composite's edges. Use something stiff.
   - **If you stabilize on glass or a glossy table, lay a matte dark sheet** (dark
     paper, cloth, a mousepad) on it for the card to sit on — bare glass is
     reflective and see-through, which adds reflections and a confusing background,
     and the card can slide on it.
2. **Light:** **one** light source only (a desk lamp or a window), **off to one
   side**, not directly overhead. You *want* a visible glare spot — the tilt sweeps
   it across the card. Don't move the light during capture.
3. **Framing:** all four corners visible with a small margin of background around
   them.

## Camera settings
4. **Hold or prop the phone STEADY**, on the **main (1×) lens — NOT the ultra-wide
   (0.5×)**, which bends the card's edges. Back off enough that the card sits
   **centered with a margin all around (~60–70% of the frame)**: a lens is sharpest
   and least distorted at its center, which keeps the card's *edges* crisp. Don't go
   so far the card gets tiny — you still need pixels on fine defects. The phone does
   not move during capture; you tilt the card.
5. **Focus + exposure LOCK.** Tap the card to focus, then lock AF/AE (iPhone:
   long-press until "AE/AF LOCK"). This stops exposure pumping frame to frame,
   which is essential for comparing frames.
6. **1080p, 30 fps** is plenty (4K is fine but larger). No flash.

## The motion (tilt the CARD, not the phone)
7. Keep the phone still. **Tilt the card** (rock the book/surface it rests on) in a
   slow "plus":
   - tilt the **far edge down, then back up** (forward–back pass), then
   - tilt the **left edge up, then the right edge up** (left–right pass).
8. Moderate tilt (**~20–30°**), ~2 seconds per pass. The goal in one sentence:
   **make the bright shine slide across every part of the card at least once.**
9. **Go slow** — slow enough that no frame is motion-blurred. Blur breaks frame
   alignment.

## Per face
10. One video for the **front**, one for the **back**, same protocol. Save as
    `samples/<sample>/front.mp4` and `back.mp4`.
11. For reverse-holo / holo / EX cards, make sure you **see the foil flare** during
    the tilt — that shimmer is the signal the grader uses to tell foil from damage.

## Don'ts
- Don't tilt or move the phone — only the card moves.
- No sleeve, toploader, or slab (extra reflections).
- Don't let your hand or the phone's shadow cross the card.
- Don't move the light or change zoom mid-capture.
- Don't rush — a slow 8-second clip beats a fast 2-second one.
