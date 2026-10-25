from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from gram2vec import (
    FeatureConfig,
    FeatureLocator,
    FeatureSpan,
    Gram2Vec,
    SyntaxRegexMatcher,
    Verbalizer,
    Vocab,
)
from gram2vec.pipeline import load_spacy_model


def test_vocab_explicit_loading():
    # English vocab
    en_vocab = Vocab(language="en")
    assert en_vocab.has("pos_unigrams")
    assert en_vocab.has("func_words")
    assert len(en_vocab.get("pos_unigrams")) > 0
    assert "NOUN" in en_vocab.get("pos_unigrams")

    # Russian vocab loaded in same process without os.environ
    ru_vocab = Vocab(language="ru")
    assert ru_vocab.has("func_words")
    assert len(ru_vocab.get("func_words")) > 0
    # Russian func_words should differ from English
    assert ru_vocab.get("func_words") != en_vocab.get("func_words")


def test_feature_config():
    default_cfg = FeatureConfig.default()
    assert default_cfg.pos_unigrams is True
    assert default_cfg.letters is False

    all_cfg = FeatureConfig.all()
    assert all_cfg.letters is True

    only_cfg = FeatureConfig.only("pos_unigrams", "emojis")
    assert only_cfg.pos_unigrams is True
    assert only_cfg.emojis is True
    assert only_cfg.func_words is False

    dict_cfg = FeatureConfig.from_dict({"pos_unigrams": 1, "func_words": 0})
    assert dict_cfg.pos_unigrams is True
    assert dict_cfg.func_words is False


def test_matcher_isolation():
    en_matcher = SyntaxRegexMatcher(language="en")
    ru_matcher = SyntaxRegexMatcher(language="ru")

    assert "it-cleft" in en_matcher.patterns
    assert "passive_rus" in ru_matcher.patterns
    assert "passive_rus" not in en_matcher.patterns


def test_load_spacy_model_with_language_instance():
    # Pass an existing Language object directly
    mock_nlp = spacy.blank("en")
    loaded = load_spacy_model(mock_nlp, max_length=5000)
    assert loaded is mock_nlp
    assert loaded.max_length == 5000


def test_load_spacy_model_missing_raises_informative_error():
    with pytest.raises(OSError) as exc_info:
        load_spacy_model("non_existent_model_name_xyz", auto_download=False)
    assert "could not be loaded" in str(exc_info.value)
    assert "non_existent_model_name_xyz" in str(exc_info.value)


def test_gram2vec_vectorization_with_blank_model():
    # Create a minimal pipeline for testing without needing heavy downloads
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    cfg = FeatureConfig.only("func_words", "num_tokens")
    g2v = Gram2Vec(model=nlp, config=cfg)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells on the seashore.",
    ]
    df = g2v.from_documents(texts)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "num_tokens:num_tokens" in df.columns
    # Check that func_words columns exist
    func_cols = [c for c in df.columns if c.startswith("func_words:")]
    assert len(func_cols) > 0


def test_verbalizer():
    # Create sample dataframe with author IDs and numerical columns
    df = pd.DataFrame({
        "authorIDs": ["author_1", "author_1", "author_2", "author_2"],
        "documentID": ["d1", "d2", "d3", "d4"],
        "pos_unigrams:NOUN": [0.5, 0.45, 0.1, 0.15],
        "pos_unigrams:VERB": [0.2, 0.25, 0.6, 0.55],
    }).set_index("documentID")

    verbalizer = Verbalizer(df, zscore_threshold=0.5, language="en")
    result = verbalizer.verbalize_author_id("author_1")
    assert isinstance(result, pd.DataFrame)
    assert "feature_name" in result.columns
    assert "zscore" in result.columns
    assert "verbalized" in result.columns
    assert len(result) > 0
