"""Tests for the free-text emotion prompt parser and its shipped lexicon."""

from __future__ import annotations

import pytest

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.prompt import LEXICON, parse_emotion_prompt


def test_plain_words_map_to_their_emotion():
    r = parse_emotion_prompt("she walks in, furious")
    assert r.intensities[Emotion.ANGER] == 1.0
    assert r.grounded
    assert "furious" in [m[0] for m in r.matched]


def test_intensifiers_and_negation():
    assert parse_emotion_prompt("slightly gloomy").intensities[Emotion.SADNESS] == pytest.approx(
        0.3
    )
    assert (
        parse_emotion_prompt("not angry at all, just tired").intensities.get(Emotion.ANGER, 0.0)
        == 0.0
    )


def test_very_does_not_raise_the_lexicon_weight():
    assert parse_emotion_prompt("very gloomy").intensities[Emotion.SADNESS] == pytest.approx(0.6)


def test_several_terms_take_the_maximum_not_the_sum():
    assert parse_emotion_prompt("annoyed and furious").intensities[Emotion.ANGER] == 1.0


def test_unsourced_emotions_are_flagged():
    r = parse_emotion_prompt("grossed out by the smell")
    assert r.intensities[Emotion.DISGUST] == pytest.approx(0.7)
    assert not r.grounded


def test_unknown_words_are_listed_and_nothing_is_invented():
    r = parse_emotion_prompt("like a lighthouse keeper")
    assert not r.matched
    assert "lighthouse" in r.unmatched
    assert all(v == 0.0 for v in r.intensities.values())


def test_all_six_emotions_are_always_keys():
    assert set(parse_emotion_prompt("").intensities) == set(Emotion)


def test_negated_unsourced_term_stays_grounded():
    r = parse_emotion_prompt("not disgusted")
    assert r.intensities[Emotion.DISGUST] == 0.0
    assert r.grounded


def test_negation_survives_an_intensifier_and_no_longer_is_a_negator():
    assert parse_emotion_prompt("not very angry").intensities[Emotion.ANGER] == 0.0
    assert parse_emotion_prompt("no longer furious").intensities[Emotion.ANGER] == 0.0


def test_every_lexicon_term_names_a_known_emotion_and_a_weight_in_range():
    assert len(LEXICON) >= 140
    for term, (emotion, w) in LEXICON.items():
        Emotion(emotion)
        assert 0.0 < w <= 1.0
        assert term == term.lower() and len(term.split()) <= 2


def test_contracted_negation_is_recognised():
    r = parse_emotion_prompt("she isn't angry")
    assert r.intensities[Emotion.ANGER] == 0.0
    assert ("angry", Emotion.ANGER, 0.0) in r.matched
    assert parse_emotion_prompt("he wasn't happy").intensities[Emotion.HAPPINESS] == 0.0
    assert parse_emotion_prompt("she won't be furious").intensities[Emotion.ANGER] == 0.0
    assert parse_emotion_prompt("she isn't angry").unmatched == ()


def test_contraction_fragments_stay_out_of_unmatched():
    r = parse_emotion_prompt("she's furious")
    assert r.unmatched == () and r.intensities[Emotion.ANGER] == 1.0


@pytest.mark.parametrize("emotion", list(Emotion))
def test_every_emotion_name_parses_to_itself_at_full_intensity(emotion):
    r = parse_emotion_prompt(emotion.value)
    assert r.intensities[emotion] == 1.0


def test_lexicon_validation_fails_loudly(monkeypatch):
    from emotive_gait.emotion import prompt

    class Fake:
        def __init__(self, text):
            self.text = text

        def read_text(self, encoding):
            return self.text

    for text, needle in [
        ("other: {}", "terms"),
        ("terms: {x: [anger]}", "'x'"),
        ("terms: {x: [joy, 0.5]}", "unknown emotion"),
        ("terms: {x: [anger, 1.5]}", "outside"),
    ]:
        monkeypatch.setattr(prompt, "_LEXICON_PATH", Fake(text))
        with pytest.raises(ValueError, match=needle):
            prompt._load_lexicon()
