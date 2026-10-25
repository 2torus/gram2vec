from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import spacy

from gram2vec import FeatureConfig, Gram2Vec

EXAMPLE_SENTENCES = [
    'The quick brown 😂 fox jumps over the lazy dog.',
    'She sells seashells by the seashore.',
    'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
    'Peter Piper picked a peck of pickled peppers.',
    'A journey of a thousand miles begins with a single step.',
    'To be or not to be, that is the question.',
    'All that glitters is not gold.',
    'The early bird catches the worm.',
    'A picture is worth a thousand words.',
    'When in Rome, do as the Romans do.',
]


def test_vectorize_example_sentences_blank_model():
    """Verify vectorization of example sentences with blank pipeline."""
    nlp = spacy.blank('en')
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')

    cfg = FeatureConfig.only(
        'func_words', 'emojis', 'punctuation', 'letters', 'num_tokens'
    )
    g2v = Gram2Vec(model=nlp, config=cfg)

    features = g2v.from_documents(EXAMPLE_SENTENCES)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(EXAMPLE_SENTENCES)
    assert 'num_tokens:num_tokens' in features.columns

    # Verify emoji feature is detected only in sentence 0
    emoji_col = 'emojis:😂'
    assert emoji_col in features.columns
    assert features[emoji_col].iloc[0] > 0
    assert (features[emoji_col].iloc[1:] == 0).all()

    # Verify all feature values are numeric and >= 0
    assert (features >= 0).all().all()


def test_vectorize_csv_export():
    """Verify extracted features can be exported and read back from CSV."""
    nlp = spacy.blank('en')
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')

    cfg = FeatureConfig.only('func_words', 'num_tokens')
    g2v = Gram2Vec(model=nlp, config=cfg)

    features = g2v.from_documents(EXAMPLE_SENTENCES[:3])

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / 'exported_features.csv'
        features.to_csv(csv_path, index=False)

        assert csv_path.exists()
        reloaded = pd.read_csv(csv_path)
        assert len(reloaded) == 3
        assert 'num_tokens:num_tokens' in reloaded.columns


@pytest.mark.skipif(
    not spacy.util.is_package('en_core_web_lg'),
    reason="Requires installed spaCy model package 'en_core_web_lg'",
)
def test_vectorize_full_en_model():
    """Verify vectorization with full en_core_web_lg model when available."""
    cfg = FeatureConfig.all()
    g2v = Gram2Vec(model='en_core_web_lg', config=cfg)
    features = g2v.from_documents(EXAMPLE_SENTENCES)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(EXAMPLE_SENTENCES)
    for prefix in ['pos_unigrams:', 'pos_bigrams:', 'func_words:']:
        cols = [c for c in features.columns if c.startswith(prefix)]
        assert len(cols) > 0
