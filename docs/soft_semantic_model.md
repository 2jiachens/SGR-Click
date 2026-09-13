# SGR-Click Soft Semantic Model

This document describes the maintained default SGR-Click implementation.

## Difference from the existing pipeline

The experiment exposes no separate candidate-threshold parameter.  To reproduce
the external launcher's intermediate masks exactly, its compatibility scaffold
uses the same fixed 0.5 convention as the GUI prediction threshold.  The final
quarter then hands off to a continuous DINOv3 soft target.

Following the official INSID3 implementation, position-stable feature
directions are estimated from a normalized zero image, centered, decomposed by
SVD, and removed with an orthogonal projection.

```text
0 <= g <= 0.75: external-model level masks
0 <= g <= 0.50: external click-root component levels
0.50 < g < 1: external DINO semantic-component jumps
g = 1: union with the validated soft endpoint
```

The last-quarter masks are nested by construction: moving the slider right may
add pixels but cannot remove pixels already present.  This prevents a secondary
object from appearing, disappearing, and then returning to the clicked object.
SimpleClick boundary projection is fixed on from 0.5 upward.  Projection is
recomputed from the current semantic level and is not OR-cached across slider
positions, so a refined boundary can remove block-shaped pixels left by the
previous level.  Low-evidence secondary components are handled by the semantic
tail audit rather than by a separate projection gate.

The zero and intermediate masks start from the same level-mask code used by
`start_granularity_recovery_no_training_gui.bat`.  Root levels are retained
exactly.  In the semantic tail, every newly added connected component is
audited with its aggregate soft SimpleClick evidence; components below the
existing `expansion-evidence-floor` are removed from that and later levels.
This prevents a low-evidence background component from becoming permanently
embedded before the actual object is reached.  A separate multi-threshold
DINOv3 hierarchy finds the validated final component.  The same evidence audit
also rejects dog-to-person spill in the final handoff.  The GUI "Predictions
threshold" remains the final probability-to-mask threshold.

## Run

Double-click the default project launcher:

```text
start_granularity_recovery_no_training_gui.bat
```

or run:

```powershell
python -m granularity_recovery.demo_granularity_recovery `
  --checkpoint weights/simpleclick_models/cocolvis_vit_base.pth
```

The remaining experimental parameters are `--prototype-radius`,
`--similarity-temperature`, and `--expansion-evidence-floor`.  The validated
soft-logit gain and boundary projection behavior are fixed internally.  There
is no `--granularity-candidate-threshold` argument in this experiment.
