from __future__ import annotations

import pytest
import spacy
from spacy.language import Language

from gram2vec import FeatureConfig, Gram2Vec


@pytest.fixture
def mock_pos_pipeline() -> Language:
    """Creates a spaCy pipeline with sentencizer and simulated POS tags."""
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')

    pos_tag_map = {
        'The': 'DET',
        'the': 'DET',
        'quick': 'ADJ',
        'Quick': 'ADJ',
        'brown': 'ADJ',
        'fox': 'NOUN',
        'dog': 'NOUN',
        'jumps': 'VERB',
        'over': 'ADP',
        'lazy': 'ADJ',
        '😂': 'SYM',
        '.': 'PUNCT',
    }

    @Language.component('fixture_pos_tagger')
    def fixture_pos_tagger(doc):
        for token in doc:
            token.pos_ = pos_tag_map.get(token.text, 'NOUN')
        return doc

    nlp.add_pipe('fixture_pos_tagger')
    return nlp


def test_find_feature_spans_func_words():
    """Verify locating function word spans."""
    nlp = spacy.blank('en')
    g2v = Gram2Vec(model=nlp, config=FeatureConfig.all())

    text = 'The quick brown 😂 fox jumps over the lazy dog.'
    spans = g2v.find_feature_spans(text, 'func_words:the')

    assert len(spans) == 2
    assert spans[0].text == 'The'
    assert spans[0].start_char == 0
    assert spans[0].end_char == 3
    assert spans[0].feature_type == 'func_words'
    assert spans[0].feature_name == 'the'

    assert spans[1].text == 'the'
    assert spans[1].start_char == 33
    assert spans[1].end_char == 36

    for s in spans:
        assert text[s.start_char : s.end_char] == s.text


def test_find_feature_spans_emojis():
    """Verify locating emoji spans."""
    nlp = spacy.blank('en')
    g2v = Gram2Vec(model=nlp, config=FeatureConfig.all())

    text = 'The quick brown 😂 fox jumps over the lazy dog.'
    spans = g2v.find_feature_spans(text, 'emojis:😂')

    assert len(spans) == 1
    assert spans[0].text == '😂'
    assert spans[0].start_char == 16
    assert spans[0].end_char == 17
    assert spans[0].feature_type == 'emojis'
    assert spans[0].feature_name == '😂'
    assert text[spans[0].start_char : spans[0].end_char] == '😂'


def test_find_feature_spans_punctuation():
    """Verify locating punctuation spans."""
    nlp = spacy.blank('en')
    g2v = Gram2Vec(model=nlp, config=FeatureConfig.all())

    text = 'The quick brown 😂 fox jumps over the lazy dog.'
    spans = g2v.find_feature_spans(text, 'punctuation:.')

    assert len(spans) == 1
    assert spans[0].text == '.'
    assert spans[0].start_char == 45
    assert spans[0].end_char == 46
    assert text[spans[0].start_char : spans[0].end_char] == '.'


def test_find_feature_spans_pos_unigrams(mock_pos_pipeline: Language):
    """Verify locating POS unigram spans using POS tags."""
    g2v = Gram2Vec(model=mock_pos_pipeline, config=FeatureConfig.all())

    text = 'The quick brown 😂 fox jumps over the lazy dog.'
    noun_spans = g2v.find_feature_spans(text, 'pos_unigrams:NOUN')
    assert len(noun_spans) == 2
    noun_texts = [s.text for s in noun_spans]
    assert 'fox' in noun_texts
    assert 'dog' in noun_texts

    det_spans = g2v.find_feature_spans(text, 'pos_unigrams:DET')
    assert len(det_spans) == 2
    det_texts = [s.text for s in det_spans]
    assert 'The' in det_texts
    assert 'the' in det_texts


def test_find_feature_spans_pos_bigrams(mock_pos_pipeline: Language):
    """Verify locating POS bigram spans."""
    g2v = Gram2Vec(model=mock_pos_pipeline, config=FeatureConfig.all())

    text = 'The quick brown fox jumps over the lazy dog.'
    spans = g2v.find_feature_spans(text, 'pos_bigrams:NOUN VERB')
    assert len(spans) == 1
    assert spans[0].text == 'fox jumps'
    assert text[spans[0].start_char : spans[0].end_char] == 'fox jumps'

    # Beginning of sentence adjective
    bos_spans = g2v.find_feature_spans('Quick brown fox jumps.', 'pos_bigrams:BOS ADJ')
    assert len(bos_spans) == 1
    assert bos_spans[0].text == 'Quick'

    # Punctuation at end of sentence
    eos_spans = g2v.find_feature_spans(
        'Quick brown fox jumps.', 'pos_bigrams:PUNCT EOS'
    )
    assert len(eos_spans) == 1
    assert eos_spans[0].text == '.'


def test_find_feature_spans_no_match():
    """Verify locating spans for a feature not in the text returns empty list."""
    nlp = spacy.blank('en')
    g2v = Gram2Vec(model=nlp, config=FeatureConfig.all())

    text = 'The quick brown fox jumps over the lazy dog.'
    spans = g2v.find_feature_spans(text, 'emojis:🚀')
    assert spans == []
