from __future__ import annotations

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gram2vec import Gram2Vec


def test_russian_vectorization():
    """Test that vectorization works correctly with Russian text using the explicit API."""
    print("Testing Russian text vectorization:")

    # Sample Russian text
    russian_text = [
        "Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог.",
        "Москва и Петербург - два главных города России, они отличаются по своей истории и архитектуре.",
    ]

    print(f"\nInput texts:\n1. {russian_text[0]}\n2. {russian_text[1]}")

    # Explicitly instantiate Gram2Vec with Russian model (no os.environ needed!)
    # Can also pass a local Path: Gram2Vec(model=Path("/path/to/local/model"))
    g2v = Gram2Vec(model="ru_core_news_lg")

    print("\nVectorizing Russian text...")
    result = g2v.from_documents(russian_text)

    print(f"\nResult shape: {result.shape}")

    # Print some feature samples
    print("\nSample features:")
    for prefix in ["dep_labels:", "morph_tags:", "func_words:"]:
        sample_cols = [col for col in result.columns if col.startswith(prefix)][:3]
        if sample_cols:
            print(f"\n{prefix[:-1]} samples:")
            print(result[sample_cols])


if __name__ == "__main__":
    test_russian_vectorization()
