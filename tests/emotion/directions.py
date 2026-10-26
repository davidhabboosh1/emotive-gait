"""``check_directions``: every grounded claim against an emotion model (spec 4.5).

A helper module, not a test module: imported by ``test_model.py`` and
``test_artefact_committed.py`` (tests/emotion is on ``sys.path`` via ``tests/conftest.py``).
"""

from emotive_gait.emotion.evidence import GROUNDED_EMOTIONS, NO_CHANGE_CLAIMS, claims_for
from emotive_gait.emotion.learned import LearnedEmotionModel
from emotive_gait.emotion.model import EmotionModel, modulate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK

# The two kinds of direction-0 claim are explained beside NO_CHANGE_CLAIMS in evidence.py.
NO_CHANGE = NO_CHANGE_CLAIMS


def _reported(model: LearnedEmotionModel, emotion: str, parameter: str) -> bool:
    return any(
        c.get("emotion") == emotion
        and c.get("parameter") == parameter
        and c.get("scaling", model.scaling) == model.scaling
        for c in model.artefact.contradictions
    )


def _check_learned_cell(model: LearnedEmotionModel, claim) -> None:
    """Spec 4.5: a fitted or thin cell agrees in sign, or is a reported contradiction.

    Recomputed from the estimate and interval, never from the artefact's own
    ``direction_check``, so a writer bug cannot pass its own gate.
    """
    emotion, parameter = claim.emotion.value, claim.parameter
    if _reported(model, emotion, parameter):
        return
    cell = model.cell(emotion, parameter)
    name = f"{emotion}.{parameter}"
    if claim.direction == 0:
        if (claim.emotion, parameter) not in NO_CHANGE_CLAIMS or cell.grounding == "thin":
            return  # "not reported vs neutral" is exempt; a thin cell has no interval
        assert cell.ci_low is not None and cell.ci_high is not None, f"{name}: no interval"
        assert cell.ci_low <= 0.0 <= cell.ci_high, (
            f"{name} interval [{cell.ci_low}, {cell.ci_high}] excludes zero and is not a "
            f"reported contradiction ({claim.citation})"
        )
        return
    assert cell.estimate * claim.direction > 0, (
        f"{name} estimate {cell.estimate} contradicts {claim.citation} and is not reported"
    )


def check_directions(model: EmotionModel) -> None:
    """Every grounded claim against ``model``; fails only on an unreported contradiction."""
    learned = isinstance(model, LearnedEmotionModel)
    p = GaitParameters()
    for emotion in GROUNDED_EMOTIONS:
        q = modulate(p, {emotion: 1.0}, G1_PROFILE_WITH_NECK, model=model)
        for c in claims_for(emotion):
            if not hasattr(p, c.parameter):
                continue
            if learned:
                if c.parameter == "expansiveness":
                    continue  # no estimate: fixed, carried by arm swing, abduction, step height
                if model.cell(emotion, c.parameter) is not None:
                    _check_learned_cell(model, c)
                    continue
                if _reported(model, emotion.value, c.parameter):
                    continue  # e.g. speed, derived from two cells: checked as a delta below
            delta = getattr(q, c.parameter) - getattr(p, c.parameter)
            if (emotion, c.parameter) in NO_CHANGE:
                assert c.direction == 0, f"{emotion}.{c.parameter} is no longer a no-change claim"
                assert abs(delta) < 1e-9, f"{emotion}.{c.parameter} moved by {delta} ({c.citation})"
                continue
            if c.direction == 0:
                continue
            assert delta * c.direction >= 0, f"{emotion}.{c.parameter} contradicts {c.citation}"
