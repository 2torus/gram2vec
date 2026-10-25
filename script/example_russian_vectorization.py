"""Example script demonstrating Russian text vectorization with the explicit Gram2Vec API."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import spacy

from gram2vec import FeatureConfig, Gram2Vec


def example_russian_vectorization() -> None:
    """Demonstrate Russian text vectorization using the explicit API."""
    print('Russian text vectorization example:')

    # Sample Russian text
    russian_text = [
        'Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог.',
        'Москва и Петербург - два главных города России, они отличаются по своей истории и архитектуре.',
    ]

    print(f'\nInput texts:\n1. {russian_text[0]}\n2. {russian_text[1]}')

    # Explicitly instantiate Gram2Vec with Russian model (or fallback if package not installed)
    try:
        g2v = Gram2Vec(model='ru_core_news_lg')
    except OSError:
        print(
            "\nNote: 'ru_core_news_lg' not installed. "
            "Falling back to blank 'ru' model with sentencizer for demonstration."
        )
        nlp = spacy.blank('ru')
        nlp.add_pipe('sentencizer')
        g2v = Gram2Vec(model=nlp, config=FeatureConfig.only('func_words', 'num_tokens'))

    print('\nVectorizing Russian text...')
    result = g2v.from_documents(russian_text)

    print(f'\nResult shape: {result.shape}')

    # Print some feature samples
    print('\nSample features:')
    for prefix in ['dep_labels:', 'morph_tags:', 'func_words:']:
        sample_cols = [col for col in result.columns if col.startswith(prefix)][:3]
        if sample_cols:
            print(f'\n{prefix[:-1]} samples:')
            print(result[sample_cols])


if __name__ == '__main__':
    example_russian_vectorization()
