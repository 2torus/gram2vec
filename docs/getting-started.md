# Getting Started

This guide walks you through installing Gram2Vec, downloading language models, and generating your first stylistic document vector.

---

## Installation

### Using pip

Install Gram2Vec from GitHub:

```bash
pip install git+https://github.com/eric-sclafani/gram2vec.git
```

### From source (development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/eric-sclafani/gram2vec.git
cd gram2vec
uv sync --all-groups
```

---

## Language Models

Gram2Vec uses [spaCy](https://spacy.io) under the hood for syntactic parsing, POS tagging, and morphological analysis.

Download the required model for your target language:

=== "English"
    ```bash
    python -m spacy download en_core_web_lg
    ```

=== "Russian"
    ```bash
    python -m spacy download ru_core_news_lg
    ```

=== "Chinese"
    ```bash
    python -m spacy download zh_core_web_sm
    ```

If you manage dependencies with `poethepoet`, you can download all models at once with:

```bash
poe models
```

---

## Three Ways to Initialize Gram2Vec

The v2.0 API is stateless and explicit:

=== "1. By Installed Package Name"
    ```python
    from gram2vec import Gram2Vec

    g2v = Gram2Vec(model="en_core_web_lg")
    ```

=== "2. From Local Directory Path"
    ```python
    from pathlib import Path
    from gram2vec import Gram2Vec

    g2v = Gram2Vec(model=Path("./models/en_core_web_lg"))
    ```

=== "3. From Pre-configured spacy.Language"
    ```python
    import spacy
    from gram2vec import Gram2Vec

    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    g2v = Gram2Vec(model=nlp)
    ```

---

## Vectorizing Documents

Once initialized, pass strings or a list of documents to `from_documents()`:

```python
from gram2vec import Gram2Vec

g2v = Gram2Vec(model="en_core_web_lg")

texts = [
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago—never mind how long precisely...",
]

# Returns a pandas DataFrame where each row is a document vector
df = g2v.from_documents(texts)
print(df)
```

Each column in `df` represents the normalized relative frequency of a stylistic feature (such as `pos_NOUN`, `dep_nsubj`, `punct_comma`, or `syntax_clause`).
