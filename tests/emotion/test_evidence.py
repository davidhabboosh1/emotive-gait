from emotive_gait.emotion.evidence import (
    EVIDENCE,
    GROUNDED_EMOTIONS,
    MEASURED_SPEEDS,
    UNSOURCED_EMOTIONS,
    Claim,
    Emotion,
    Grounding,
    claims_for,
)


def test_disgust_and_surprise_are_marked_unsourced():
    assert UNSOURCED_EMOTIONS == {Emotion.DISGUST, Emotion.SURPRISE}
    for e in UNSOURCED_EMOTIONS:
        assert all(c.grounding is Grounding.UNSOURCED for c in claims_for(e))
        assert all(
            c.citation == "none - placed by arousal/valence (design 10.2)" for c in claims_for(e)
        )


def test_the_four_grounded_emotions_have_citations():
    assert GROUNDED_EMOTIONS == {Emotion.ANGER, Emotion.FEAR, Emotion.HAPPINESS, Emotion.SADNESS}
    for e in GROUNDED_EMOTIONS:
        cs = claims_for(e)
        assert cs
        assert all(c.citation for c in cs)
        assert all(c.grounding is not Grounding.UNSOURCED for c in cs)


def test_no_claim_asserts_a_magnitude_in_degrees():
    """Appendix A.2: only m/s speeds and swing-phase percentages have published values."""
    for c in EVIDENCE:
        assert "degree" not in c.note.lower() and "deg" not in c.note.lower().split()


def test_measured_speeds_order_matches_the_literature():
    s = MEASURED_SPEEDS
    assert s[Emotion.ANGER] > s[Emotion.HAPPINESS] > s[Emotion.FEAR] > s[Emotion.SADNESS]


def test_headline_cue_directions():
    def d(e, p):
        return next(c.direction for c in claims_for(e) if c.parameter == p)

    assert d(Emotion.SADNESS, "neck_pitch") > 0  # head down
    assert d(Emotion.SADNESS, "arm_swing_amp") < 0
    assert d(Emotion.ANGER, "elbow_flexion") > 0
    assert d(Emotion.FEAR, "elbow_flexion") > 0
    assert d(Emotion.HAPPINESS, "elbow_flexion") < 0
    assert d(Emotion.FEAR, "knee_flexion_bias") > 0


def test_claims_are_frozen_and_directions_are_signs():
    assert all(c.direction in (-1, 0, 1) for c in EVIDENCE)
    try:
        EVIDENCE[0].direction = 0  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("Claim must be frozen")
    assert isinstance(EVIDENCE[0], Claim)


def test_claims_per_emotion_and_table_order():
    counts = {e: len(claims_for(e)) for e in Emotion}
    assert counts == {
        Emotion.ANGER: 12,
        Emotion.FEAR: 11,
        Emotion.HAPPINESS: 10,
        Emotion.SADNESS: 13,
        Emotion.DISGUST: 4,
        Emotion.SURPRISE: 4,
    }
    assert len(EVIDENCE) == 54
    assert [c.emotion for c in EVIDENCE[::1]] == sorted(
        (c.emotion for c in EVIDENCE),
        key=[
            Emotion.ANGER,
            Emotion.FEAR,
            Emotion.HAPPINESS,
            Emotion.SADNESS,
            Emotion.DISGUST,
            Emotion.SURPRISE,
        ].index,
    )


def test_inert_parameters_are_present():
    params = {c.parameter for c in EVIDENCE}
    assert {"jerk", "swing_phase_fraction"} <= params


def test_emotion_values_are_lowercase_names():
    assert Emotion("anger") is Emotion.ANGER
    assert all(e.value == e.name.lower() for e in Emotion)
