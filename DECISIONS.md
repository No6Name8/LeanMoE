# Decisions Log

Format: **Decision** — what we chose. **Why** — reasoning. **Rejected** — what we didn't pick and why.

---

## Project scope
**Decision:** Single-machine (8GB laptop) MoE expert prefetching, not the full multi-device ExpertRelay vision.
**Why:** Only one 8GB machine available for the hackathon window; multi-device sync adds a networking failure mode we can't afford to debug in 48 hours.
**Rejected:** Multi-device Soft Bus version — too much infrastructure risk for a 2-day deadline.

## Challenge picked
**Decision:** Challenge 2 — Deliver AI with Less Compute.
**Why:** No external dataset is provided for any challenge, so all three are scored on self-defined rigor. Challenge 2 gives a built-in, objective metric (time/RAM/cost) that's easy to measure and hard to fake, which maps directly onto "Task Performance" and "Evaluation & Iteration."

---

(Add entries below as real build decisions get made — model choice, predictor design, cache policy, etc.)
