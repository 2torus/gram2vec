# Gram2Vec

## Description
`Gram2Vec` is a grammatical style embedding algorithm that embeds documents into a higher dimensional space by extracting the normalized relative frequencies of stylistic features present in the text. To learn more, please refer to https://arxiv.org/pdf/2406.12131.

More specifically, Gram2vec vectorizes based off features pertaining to grammar, such as POS tags, punctuation, syntactic constructions, and much more.

## Motivation

Vector representations produced by deep neural networks excel at capturing meaning from natural language text. However, they infamously lack innate explainability. Each position in the vector is learned through a training period of weight optimizations. This means the numbers in these vectors don't have concrete features they pertain to. For certain tasks, this is not ideal.

In `authorship attribution` (AA), the task of using an automated system to identify the author of a document, explainability is an important factor. This task is interested in identifying what stylometric choices authors make that differentiate them from each other. Why was author A chosen over author B? What linguistic choices did author A make that caused my AA algorithm to choose them? 

Additionally, deep embeddings are known to capture semantic content from text. For authorship attribution, this is not ideal since the task is primarily concerned with writing style, such as grammar usage, lexical word choice, etc...

Gram2vec was born with these ideas in mind. Given a document, it produces a vector such that each position corresponds to a tangible stylistic feature. 

Instead of using uninterpretable deep embeddings inside of an AA model, getting high numbers, and calling it a day, using gram2vec vectors will provide you with reasons behind a model's predictions.


## Setup

In your working directory, create an environment by running (I think any version > 3.9 should work, not 100% sure though):
```bash
python3.11 -m venv venv/
source venv/bin/activate
```
which will create a directory called `venv/` to store all the dependencies. 

Next, run:
```bash
pip install git+https://github.com/eric-sclafani/gram2vec
```
which will install gram2vec into your environment, as well as all of its dependencies.

If you'd like to play around with the source code, you can instead clone the repo directly and install with the `-e` flag (editable installation):
```bash
pip install -e gram2vec/
```

## Usage

### Vectorizer (`Gram2Vec`)

To get started, import `Gram2Vec` from `gram2vec` and initialize an instance. The API is completely stateless and explicit—there are no global variables or hidden environment variables.

#### 1. Explicit Model Loading

You can load your spaCy model in three ways:

```python
from pathlib import Path
import spacy
from gram2vec import Gram2Vec

# Option A: By installed package name
g2v = Gram2Vec(model='en_core_web_lg')

# Option B: From a local model directory on disk
g2v = Gram2Vec(model=Path('./models/ru_core_news_lg'))

# Option C: From a pre-configured spaCy pipeline
nlp = spacy.load('en_core_web_sm', exclude=['ner'])
nlp.max_length = 5_000_000
g2v = Gram2Vec(model=nlp)
```

The language (`"en"`, `"ru"`, etc.) is automatically inferred from `model.lang`, or you can specify it explicitly via `language="ru"`. Multiple `Gram2Vec` instances for different languages can run in the same Python process without interference.

#### 2. Vectorizing Documents

There are two primary ways to vectorize:

##### From a list of strings (`.from_documents()`):
```python
documents = [
    'This is a test string 😄!!!',
    'The string below me is false.',
    'The string above me is true 😱!',
]

df = g2v.from_documents(documents)
```

|   pos_unigrams:ADJ   |   pos_unigrams:ADP   |   pos_unigrams:ADV   | ... |   sentences:obj-relcl   |   sentences:tag-question   |   sentences:coordinate-clause   |
|----------------------|----------------------|----------------------|-----|------------------------|---------------------------|-------------------------------|
|           0.000000   |           0.000000   |           0.0        | ... |           0.0          |           0.0             |           0.0                 |
|           0.142857   |           0.142857   |           0.0        | ... |           0.0          |           0.0             |           0.0                 |
|           0.142857   |           0.142857   |           0.0        | ... |           0.0          |           0.0             |           0.0                 |

##### From JSONLines (`.from_jsonlines()`):
Generates a dataframe from either a single `.jsonl` file or a directory of `.jsonl` files. Expects `fullText`, `authorIDs`, and `documentID`.

```python
df = g2v.from_jsonlines('path/to/dataset/data.jsonl')
# Or a folder of .jsonl files:
df = g2v.from_jsonlines('path/to/dataset/directory/')
```

#### 3. Feature Configuration (`FeatureConfig`)

You can configure which feature extractors are active using the typed `FeatureConfig` dataclass or a dictionary:

```python
from gram2vec import FeatureConfig, Gram2Vec

# Default features (letters=False, all others True)
cfg = FeatureConfig.default()

# Enable all features (including letters)
cfg = FeatureConfig.all()

# Enable only specific features
cfg = FeatureConfig.only('pos_unigrams', 'pos_bigrams', 'func_words')

# Pass to Gram2Vec
g2v = Gram2Vec(model='en_core_web_lg', config=cfg)
```

Optionally, you can include document word2vec embeddings for experimental comparison by passing `include_content_embedding=True`:

```python
df = g2v.from_documents(documents, include_content_embedding=True)
```

### Feature Locator

You can inspect exactly where specific grammatical features occur in text spans:

```python
spans = g2v.find_feature_spans('The quick brown fox jumps.', 'pos_unigrams:NOUN')
for span in spans:
    print(f"{span.feature_name} '{span.text}' at [{span.start_char}:{span.end_char}]")
```

### Verbalizer

The `Verbalizer` calculates z-scores from a feature vector dataframe to produce human-readable explanations of the most distinguishing stylistic traits:

```python
from gram2vec import Gram2Vec

g2v = Gram2Vec(model='en_core_web_lg')
docs_df = g2v.from_jsonlines('path/to/dataset/directory/')

verbalizer = g2v.create_verbalizer(docs_df, zscore_threshold=2.0)

# Verbalize an author
author_df = verbalizer.verbalize_author_id('author_112')
print(author_df.head(3))

# Verbalize an unseen document vector
doc_df = verbalizer.verbalize_document_vector(unseen_vector)
print(doc_df.head(3))
```

## Vocab

This section provides more details about how vocabulary works in `gram2vec` and is not needed to understand how to use the software.

In general, each feature is frequency based. A **vocab** is therefore the collection of items that get counted for a feature. Each vocab is stored in a local `vocab/` directory. These files are read by `gram2vec` and used in the feature extractors.

If new vocabularies are added, for the sake of consistency, the vocabulary files should have the same name as the feature function. Examples of this can be seen in `vectorizer.py`.

Some features, in particular the `sentences` feature produced by `SyntaxRegexMatcher`, do not require a vocab (at least, not in the same way the others do).

Some vocabularies require more explanation. The following subsections go into more detail about them

### POS Bigrams

From the list of POS tags from <a href="https://universaldependencies.org/u/pos/">Universal dependencies</a> (18 total tags), I create all possible combinations. 
> So $18^2$ = 324 possible POS bigrams

## Adding more features

This section is for gram2vec development. If you'd like to extend the code and add more countable features, here is a detailed guide on how to do so.

### Step 1

Define what you want to count and why. What is the intuition behind it and what could it tell you about authors' writing styles?

### Step 2

Define a vocabulary. This will be some collection of countable objects. In the [vocab](src/gram2vec/vocab) directory, the vocabularies are just text files of countable items. If using an external file for vocabulary, give it the same name as the feature you're counting. Alternatively, you may want to add a regex matching feature similar to how my other package, [Syntax Regex Matcher](https://github.com/eric-sclafani/syntax-regex-matcher), works. 

After defining your vocab, register it inside of the `Vocab` class in [vocab.py](src/gram2vec/vocab.py). There are two methods for doing so based on how you implement the features. Use what's already there as a guide.

### Step 3

Inside of [vectorizer.py](src/gram2vec/vectorizer.py), define an extractor method on `Gram2Vec` that returns a **Counter** of your countable feature and add it to `_get_extractor_map()`.

### Step 4

Add your feature as a field to `FeatureConfig` in [config.py](src/gram2vec/config.py).

### Step 5

Finally, if you want to also use the `Verbalizer` with your new features, you should add it to `NAME_MAPPING_EN` (and `NAME_MAPPING_RU`) in [verbalizer.py](src/gram2vec/verbalizer.py) to provide a descriptive string representation. 


## Acknowledgements & Credits

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

The v2.0 API refactoring was developed in pair-programming collaboration with Antigravity (AI coding assistant by Google DeepMind). Every line of code was reviewed by a human.
