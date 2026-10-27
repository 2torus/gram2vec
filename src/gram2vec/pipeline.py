"""Pipeline utilities for loading spaCy models and sentence-level processing."""

from __future__ import annotations

from pathlib import Path
from sys import stderr
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

SentenceSpan = Tuple[int, int]
Bigram = Tuple[str, str]


def load_spacy_model(
    model: Union[str, Path, Language],
    exclude: Optional[List[str]] = None,
    max_length: int = 10_000_000,
    auto_download: bool = False,
) -> Language:
    """Explicitly loads a spaCy Language model without global state.

    Args:
        model: A pre-loaded spaCy Language instance, a local filesystem directory Path/str,
               or an installed spaCy model package name (e.g., 'en_core_web_lg').
        exclude: Pipeline components to exclude (default: ['ner']).
        max_length: Maximum document length for the spaCy pipeline.
        auto_download: Whether to attempt automatic downloading if the model package is missing.

    Returns:
        Configured spacy.language.Language instance.

    """
    if exclude is None:
        exclude = ['ner']

    # 1. Pre-loaded spaCy Language instance
    if isinstance(model, Language):
        model.max_length = max_length
        return model

    # 2. Local filesystem path
    model_path = Path(model) if isinstance(model, (str, Path)) else None
    if model_path is not None and model_path.exists():
        nlp = spacy.load(str(model_path), exclude=exclude)
        nlp.max_length = max_length
        return nlp

    # 3. Installed package name
    model_name = str(model)
    try:
        nlp = spacy.load(model_name, exclude=exclude)
    except OSError as err:
        if auto_download:
            print(
                f"Gram2Vec: Downloading spaCy language model '{model_name}'...",
                file=stderr,
            )
            from spacy.cli import download

            download(model_name)
            nlp = spacy.load(model_name, exclude=exclude)
        else:
            raise OSError(
                f"spaCy language model '{model_name}' could not be loaded. "
                f"Please ensure it is installed (e.g. 'python -m spacy download {model_name}'), "
                f'provide a path to a local model directory (Path or str), or pass an already loaded '
                f'spacy.language.Language object. To permit automatic downloading, set auto_download=True.'
            ) from err

    nlp.max_length = max_length
    return nlp


# ~~~ Linguistic feature extraction helpers from spaCy Doc ~~~


def extract_tokens(doc: Doc) -> List[str]:
    """Returns token text for each token in the document."""
    return [token.text for token in doc]


def extract_pos_tags(doc: Doc) -> List[str]:
    """Returns part-of-speech tags for each token in the document."""
    return [token.pos_ for token in doc]


def extract_dep_labels(doc: Doc) -> List[str]:
    """Returns dependency labels for each token in the document."""
    return [token.dep_ for token in doc]


def extract_morph_tags(doc: Doc) -> List[str]:
    """Returns non-empty morphological tags for each token in the document."""
    return [morph for token in doc for morph in token.morph if morph != '']


def extract_pos_bigrams(doc: Doc) -> List[str]:
    """Generates POS bigrams with sentence boundaries (BOS and EOS)."""
    spans: List[SentenceSpan] = [(sent.start, sent.end) for sent in doc.sents]
    pos_tags = [token.pos_ for token in doc]

    new_tokens: List[str] = []
    for i, pos in enumerate(pos_tags):
        for sent_start, sent_end in spans:
            if i == sent_start:
                new_tokens.append('BOS')
            elif i == sent_end:
                new_tokens.append('EOS')
        new_tokens.append(pos)
    new_tokens.append('EOS')

    bigrams_list: List[Bigram] = [
        (new_tokens[i], new_tokens[i + 1]) for i in range(len(new_tokens) - 1)
    ]
    return [f'{b[0]} {b[1]}' for b in bigrams_list]


# ~~~ Biber-style categorical features (fixed, content-agnostic keys) ~~~

PRONOUN_KEYS: Tuple[str, ...] = ('first_person', 'second_person', 'third_person', 'it')
DEGREE_KEYS: Tuple[str, ...] = ('comparative', 'superlative')
AVG_LENGTH_KEYS: Tuple[str, ...] = (
    'chars_per_token',
    'tokens_per_sentence',
    'tokens_per_noun_chunk',
)

_PERSON_TO_KEY = {'1': 'first_person', '2': 'second_person', '3': 'third_person'}
_DEGREE_TO_KEY = {'Cmp': 'comparative', 'Sup': 'superlative'}


def pronoun_category(token: Token) -> Optional[str]:
    """Returns the pronoun category of a token, or None if it is not a personal pronoun.

    Uses the universal ``Person`` morphological feature, so it works for any language
    whose model predicts morphology. English 'it' is split out from third person,
    since it behaves very differently stylistically (Biber).
    """
    if token.pos_ != 'PRON':
        return None
    person = token.morph.get('Person')
    if not person:
        return None
    if token.lang_ == 'en' and token.lower_ == 'it':
        return 'it'
    return _PERSON_TO_KEY.get(person[0])


def degree_category(token: Token) -> Optional[str]:
    """Returns 'comparative'/'superlative' from the universal ``Degree`` feature, else None."""
    degree = token.morph.get('Degree')
    if not degree:
        return None
    return _DEGREE_TO_KEY.get(degree[0])


def extract_pronouns(doc: Doc) -> List[str]:
    """Returns the pronoun category for each personal pronoun in the document."""
    return [c for c in (pronoun_category(t) for t in doc) if c is not None]


def extract_degree(doc: Doc) -> List[str]:
    """Returns the degree category for each comparative/superlative token."""
    return [c for c in (degree_category(t) for t in doc) if c is not None]


def extract_named_entity_labels(doc: Doc) -> List[str]:
    """Returns entity labels (not entity text) for each named entity."""
    return [ent.label_ for ent in doc.ents]


def extract_avg_lengths(doc: Doc) -> Dict[str, float]:
    """Computes average token, sentence and noun-chunk lengths."""
    num_tokens = len(doc)
    num_sents = len(list(doc.sents)) if num_tokens else 0
    try:
        chunks = list(doc.noun_chunks)
    except (NotImplementedError, ValueError):
        # Language has no noun-chunk iterator, or the doc is not parsed
        chunks = []
    return {
        'chars_per_token': (
            sum(len(t.text) for t in doc) / num_tokens if num_tokens else 0.0
        ),
        'tokens_per_sentence': num_tokens / num_sents if num_sents else 0.0,
        'tokens_per_noun_chunk': (
            sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
        ),
    }
