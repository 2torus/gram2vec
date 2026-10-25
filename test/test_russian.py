from __future__ import annotations

import pandas as pd
import pytest
import spacy

from gram2vec import FeatureConfig, Gram2Vec, Vocab


def test_russian_vocab_loading():
    """Verify loading Russian vocabulary and category structure."""
    vocab = Vocab(language='ru')

    dep_labels = vocab.get('dep_labels')
    assert isinstance(dep_labels, list)
    assert len(dep_labels) == 41
    assert 'ROOT' in dep_labels
    assert 'advmod' in dep_labels

    morph_tags = vocab.get('morph_tags')
    assert isinstance(morph_tags, list)
    assert len(morph_tags) == 60
    assert 'Aspect=Perf' in morph_tags
    assert 'Case=Nom' in morph_tags

    func_words = vocab.get('func_words')
    assert isinstance(func_words, list)
    assert len(func_words) == 353
    assert 'в' in func_words
    assert 'и' in func_words
    assert 'он' in func_words

    sentences = vocab.get('sentences')
    assert isinstance(sentences, list)
    assert len(sentences) == 10
    assert 'passive_rus' in sentences
    assert 'parataxis_rus' in sentences


def test_russian_vocab_different_from_english():
    """Verify Russian and English vocabularies are language-isolated."""
    ru_vocab = Vocab(language='ru')
    en_vocab = Vocab(language='en')

    assert ru_vocab.get('func_words') != en_vocab.get('func_words')
    assert ru_vocab.get('morph_tags') != en_vocab.get('morph_tags')
    assert ru_vocab.get('sentences') != en_vocab.get('sentences')


def test_russian_vectorization_blank_model():
    """Verify Russian text vectorization with blank spaCy model and sentencizer."""
    nlp = spacy.blank('ru')
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')

    cfg = FeatureConfig.only('func_words', 'num_tokens')
    g2v = Gram2Vec(model=nlp, config=cfg)

    russian_text = [
        'Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог.',
        'Москва и Петербург - два главных города России, они отличаются по своей истории и архитектуре.',
    ]

    result = g2v.from_documents(russian_text)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'num_tokens:num_tokens' in result.columns
    assert result['num_tokens:num_tokens'].iloc[0] == 22.0
    assert result['num_tokens:num_tokens'].iloc[1] == 17.0

    # Verify extracted Russian function words
    func_cols = [c for c in result.columns if c.startswith('func_words:')]
    assert len(func_cols) == 353

    # Check that known words from sentence 1 are detected
    assert result['func_words:в'].iloc[0] > 0
    assert result['func_words:он'].iloc[0] > 0
    assert result['func_words:когда'].iloc[0] > 0


@pytest.mark.skipif(
    not spacy.util.is_package('ru_core_news_lg'),
    reason="Requires installed spaCy model package 'ru_core_news_lg'",
)
def test_russian_vectorization_full_model():
    """Verify Russian vectorization when full ru_core_news_lg model is available."""
    russian_text = [
        'Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог.',
        'Москва и Петербург - два главных города России, они отличаются по своей истории и архитектуре.',
    ]

    g2v = Gram2Vec(model='ru_core_news_lg')
    result = g2v.from_documents(russian_text)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    for prefix in ['dep_labels:', 'morph_tags:', 'func_words:']:
        cols = [c for c in result.columns if c.startswith(prefix)]
        assert len(cols) > 0
