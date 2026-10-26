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
