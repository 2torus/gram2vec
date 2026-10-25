"""Example script demonstrating Russian vocabulary loading with the explicit Gram2Vec API."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from gram2vec import Vocab


def example_russian_vocab() -> None:
    """Demonstrate Russian vocabulary loading via explicit Vocab instance."""
    print('Russian vocabulary loading example:')

    # Explicitly instantiate Vocab for Russian without touching os.environ
    vocab = Vocab(language='ru')

    # Example dependency labels
    dep_labels = vocab.get('dep_labels')
    print(f'\nDependency labels ({len(dep_labels)} items):')
    print(
        ', '.join(dep_labels[:10]) + '...'
        if len(dep_labels) > 10
        else ', '.join(dep_labels)
    )

    # Example morphological tags
    morph_tags = vocab.get('morph_tags')
    print(f'\nMorphological tags ({len(morph_tags)} items):')
    print(
        ', '.join(morph_tags[:10]) + '...'
        if len(morph_tags) > 10
        else ', '.join(morph_tags)
    )

    # Example function words
    func_words = vocab.get('func_words')
    print(f'\nFunction words ({len(func_words)} items):')
    print(
        ', '.join(func_words[:10]) + '...'
        if len(func_words) > 10
        else ', '.join(func_words)
    )

    # Example sentence patterns (from matcher)
    sentences = vocab.get('sentences')
    print(f'\nSentence patterns ({len(sentences)} items):')
    print(', '.join(sentences))


if __name__ == '__main__':
    example_russian_vocab()
