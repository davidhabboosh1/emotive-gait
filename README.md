# emotive-gait

Emotion-modulated humanoid gait in MuJoCo. Six continuous emotion intensities (anger, disgust,
fear, happiness, sadness, surprise) are mapped through three affective axes onto the parameters
of a walking gait for the Unitree G1. Every claim the model relies on is either cited to a
source or explicitly flagged as unsourced.

| Sadness = 1.0 | Anger = 1.0 |
| --- | --- |
| ![sad gait](docs/media/sad.gif) | ![angry gait](docs/media/angry.gif) |

## Quickstart

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m mujoco_menagerie prefetch unitree_g1
emotive-gait demo --emotion sadness=0.8     # writes demo.mp4 (needs ffmpeg on PATH)
emotive-gait evidence                       # every claim with its grounding and citation
pytest
```

`demo` also takes `--physics`, `--root {free,z-rail,weld}`, `--hold-speed`, `--gif` and
`--dry-run`; see `emotive-gait demo --help`.

## Requirements

Python 3.11 or newer, 3.12 recommended. `mujoco-menagerie` downloads the robot model on first
use and extracts it with `tarfile.extractall(filter=...)` (PEP 706), which needs Python 3.11.4
or newer. If that step fails the loader raises an error naming the prefetch command above and
the sparse-checkout fallback. MuJoCo 3.2.0 or newer.

## Architecture

**`robot/`** maps semantic joint roles (hip pitch, elbow, neck pitch, ...) to actuator names, so
nothing above it hard-codes a model. The G1 has no neck, yet head inclination is the strongest
posture cue in the literature, so the loader injects a pitch neck joint and splits the head's
mass out of the torso.

**`gait/`** defines `GaitParameters`, which are the same quantities the literature reports
(cadence, step length, trunk and neck pitch, arm swing, ...). Joint trajectories are periodic
C2 curves, and the in-place generator turns a parameter set into per-role angles over a gait
cycle.

**`emotion/`** runs evidence, then axes, then the model. The evidence module records signed
directions (more, same, less) with a citation and a grounding level for each claim. Emotion
intensities become activation, valence and tension, and those shift the parameters. The model
encodes directions, not magnitudes; the gains are tuned, not cited.

**`sim/`** has a kinematic tier that poses the robot with no physics (the source of stimuli
for video) and a physical tier that steps position servos open loop with the pelvis on a
vertical rail. ZMP and center-of-pressure are measured for instrumentation and never used for
control.

## Limitations

- The physics tier is in-place stepping on a vertical rail. Open-loop stepping on a free
  floating base falls within 0.74-2.25 s at every expressive amplitude we measured, and there
  is no balance controller.
- Forward stride exists only in the kinematic tier. Physical forward walking needs task-space
  IK, which is a later phase.
- Disgust and surprise have no gait kinematics in any source we found. They ship as placements
  on the arousal/valence plane, flagged `UNSOURCED`, and the CLI warns when you use them.
- No recognition rate has been measured: nobody has yet checked that people read these gaits as
  the intended emotions.

## Grounded versus unsourced

| Emotion | Grounding | Primary source |
| --- | --- | --- |
| anger | measured speed, directional posture | Roether et al. 2011 |
| fear | measured speed, directional posture | Roether et al. 2011 |
| happiness | measured speed, directional posture | Roether et al. 2011 |
| sadness | measured speed, directional posture | Roether et al. 2011 |
| disgust | unsourced | none |
| surprise | unsourced | none |

Other cited sources include Montepare 1987, Michalak 2009, Gross 2012, Barliya 2013,
Halovic and Kroos 2018, and Randhavane 2019. `emotive-gait evidence` prints the full table.

## Related work

- **EMoG** ([arXiv 2609.14432](https://arxiv.org/abs/2609.14432)): emotion-modulated gait
  generation from a continuous style code through an MLP and an RL tracking policy, trained on
  a proprietary performer dataset. Which robot it used is unverified.
- **PAMoR** ([arXiv 2608.28213](https://arxiv.org/pdf/2608.28213)): parameterized affective
  motion generation in real time; the closest in approach.
- **EmoPose** ([arXiv 2609.23414](https://arxiv.org/pdf/2609.23414)): emotion-aware gesture,
  not gait.
- **Walk Like This**: stylized locomotion via example-driven diffusion.

This project differs in being analytic, interpretable, cited and open, where EMoG is learned
from a proprietary dataset. That is a difference, not an advantage.

## License

MIT. The Unitree G1 model is BSD-3-Clause, from MuJoCo Menagerie.
