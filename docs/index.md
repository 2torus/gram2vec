# Gram2Vec: Interpretable Document Vectorizer

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2406.12131-b31b1b.svg)](https://arxiv.org/abs/2406.12131)

**Gram2Vec** is an interpretable, stylistic document embedding library that maps texts into high-dimensional vector representations based on the normalized relative frequencies of concrete linguistic, syntactic, and stylistic features.

Unlike black-box neural embeddings, every dimension in a `Gram2Vec` vector directly corresponds to an interpretable grammatical feature—making it ideal for **Authorship Attribution (AA)**, **AI text detection**, **forensic linguistics**, and **stylometric analysis**.

---

## Key Highlights

=== "💡 100% Interpretable"
    Every feature corresponds to an explicit grammatical construct (POS n-grams, morph tags, dependency relations, regex syntax trees, punctuation, emojis). Explain exactly *why* a document was attributed to an author.

=== "⚡ Explicit & Stateless API"
    Clean, predictable, and thread-safe v2.0 API. Load spaCy models explicitly from installed package names, local directory paths, or pre-configured `spacy.Language` objects.

=== "🌍 Multilingual"
    Out-of-the-box support for English (`en_core_web_lg`), Russian (`ru_core_news_lg`), and Chinese (`zh_core_web_sm`).

=== "📊 Extensible"
    Easily configure feature subsets with `FeatureConfig` or implement custom stylistic extractors with regex syntax patterns.

---

## Quick Example

```python
from gram2vec import Gram2Vec, FeatureConfig

# Initialize with English spaCy model and all features enabled
g2v = Gram2Vec(model="en_core_web_lg", config=FeatureConfig.all())

# Extract stylistic feature vectors as a pandas DataFrame
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore."
]
df = g2v.from_documents(documents)
print(df.shape)  # (2, num_features)
```

---

## Paper & Citation

If you use `Gram2Vec` in your research, please cite the original paper:

> Peter Zeng, Hannah Stortz, Eric Sclafani, Alina Shabaeva, Maria Elizabeth Garza, Daniel Greeson, and Owen Rambow.  
> **[Gram2Vec: An Interpretable Document Vectorizer](https://arxiv.org/abs/2406.12131)**. *arXiv preprint arXiv:2406.12131* (2024).

```bibtex
@misc{zeng2024gram2vecinterpretabledocumentvectorizer,
      title={Gram2Vec: An Interpretable Document Vectorizer}, 
      author={Peter Zeng and Hannah Stortz and Eric Sclafani and Alina Shabaeva and Maria Elizabeth Garza and Daniel Greeson and Owen Rambow},
      year={2024},
      eprint={2406.12131},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12131}, 
}
```

---

## Next Steps

- Check out the [Getting Started Guide](getting-started.md) for installation and initial setup.
- Explore the [Interactive Notebooks](notebooks/test_gram2vec.ipynb) to see vectorization in action.
- Browse the [Stylistic Feature Catalog](reference/features.md) for a dictionary of extracted features.
- Inspect the [API Reference](reference/api.md) for comprehensive function and class documentation.
