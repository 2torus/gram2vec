from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import demoji
import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc

from .config import FeatureConfig
from .feature_locator import FeatureLocator, FeatureSpan
from .matcher import SyntaxRegexMatcher
from .pipeline import (
    extract_dep_labels,
    extract_morph_tags,
    extract_pos_bigrams,
    extract_pos_tags,
    load_spacy_model,
)
from .verbalizer import Verbalizer
from .vocab import Vocab

# Filter out specific open_text deprecation warning
warnings.filterwarnings(
    "ignore",
    message=r"open_text is deprecated\. Use files\(\) instead\.",
    category=DeprecationWarning,
)


@dataclass
class Gram2VecDocument:
    """Encapsulates raw text and processed spaCy Doc object.
    Preserves raw text for emoji extraction and original statistics.
    """

    raw: str
    doc: Doc
    num_tokens: int
    num_sentences: int


def _remove_emojis(document: str) -> str:
    """Removes emojis from a string and normalizes extra whitespace."""
    new_string = demoji.replace(document, "").split()
    return " ".join(new_string)


def _load_jsonlines(path: Union[str, Path]) -> pd.DataFrame:
    """Loads one or more .jsonl files into a single dataframe."""
    p = Path(path)
    if p.is_file() or str(path).endswith(".jsonl"):
        return pd.read_json(str(p), lines=True)
    if p.is_dir():
        dfs = [pd.read_json(str(f), lines=True) for f in p.glob("*.jsonl")]
        if not dfs:
            raise FileNotFoundError(f"No .jsonl files found in directory: {p}")
        return pd.concat(dfs).reset_index(drop=True)
    raise FileNotFoundError(f"Path does not exist: {path}")


def _get_json_entries(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Retrieves fullText, authorIDs, and documentID fields from a jsonlines dataframe."""
    try:
        documents = df["fullText"]
        author_ids = df["authorIDs"]
        document_ids = df["documentID"]
    except KeyError as err:
        raise KeyError(
            "Specified jsonlines dataframe missing required fields: "
            "'fullText', 'authorIDs', and 'documentID'"
        ) from err
    return documents, author_ids, document_ids


class Gram2Vec:
    """Primary interface for stylistic feature vectorization.
    Fully self-contained, stateless, and supports explicit spaCy model loading.
    """

    def __init__(
        self,
        model: Union[str, Path, Language] = "en_core_web_lg",
        *,
        language: Optional[str] = None,
        config: Optional[Union[FeatureConfig, Dict[str, Any]]] = None,
        vocab: Optional[Vocab] = None,
        vocab_dir: Optional[Union[str, Path]] = None,
        matcher: Optional[SyntaxRegexMatcher] = None,
        spacy_exclude: Optional[List[str]] = None,
        auto_download: bool = False,
        max_length: int = 10_000_000,
        batch_size: int = 1000,
        n_process: int = 1,
    ):
        """Initialize a Gram2Vec vectorizer.

        Args:
            model: spaCy Language instance, local model directory path, or installed package name.
            language: Language code ('en', 'ru', etc.). Inferred from model if omitted.
            config: FeatureConfig instance or dict of feature flags.
            vocab: Custom Vocab instance. Automatically constructed if None.
            vocab_dir: Directory containing vocabulary files if using custom vocabs.
            matcher: Custom SyntaxRegexMatcher instance. Automatically constructed if None.
            spacy_exclude: spaCy pipeline components to exclude (default: ['ner']).
            auto_download: If True, attempts to download the model package if not found.
            max_length: Maximum document length for spaCy.
            batch_size: Document batch size for nlp.pipe.
            n_process: Number of parallel worker processes for nlp.pipe.

        """
        self.nlp = load_spacy_model(
            model=model,
            exclude=spacy_exclude,
            max_length=max_length,
            auto_download=auto_download,
        )

        # Infer language from spaCy model if not explicitly specified
        self.language = language if language is not None else getattr(self.nlp, "lang", "en")

        # Initialize matcher for sentence syntax constructions
        self.matcher = matcher if matcher is not None else SyntaxRegexMatcher(language=self.language)

        # Initialize vocabulary
        self.vocab = vocab if vocab is not None else Vocab(
            language=self.language,
            vocab_dir=vocab_dir,
            matcher=self.matcher,
        )

        # Initialize configuration
        if isinstance(config, FeatureConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = FeatureConfig.from_dict(config)
        else:
            self.config = FeatureConfig.default()

        self.batch_size = batch_size
        self.n_process = n_process

        self._feature_locator: Optional[FeatureLocator] = None

    @property
    def feature_locator(self) -> FeatureLocator:
        """Returns a FeatureLocator configured with this instance's model and vocabs."""
        if self._feature_locator is None:
            self._feature_locator = FeatureLocator(self.nlp, self.vocab, self.matcher)
        return self._feature_locator

    def find_feature_spans(self, text: str, feature_name: str) -> List[FeatureSpan]:
        """Locates spans in text where the specified feature occurs."""
        return self.feature_locator.locate_feature(text, feature_name)

    def create_verbalizer(
        self, docs_df: pd.DataFrame, zscore_threshold: float = 2.0
    ) -> Verbalizer:
        """Creates a Verbalizer instance calibrated to this model's language."""
        return Verbalizer(docs_df, zscore_threshold=zscore_threshold, language=self.language)

    # ~~~ Feature Extractors ~~~

    def _extract_pos_unigrams(self, doc: Gram2VecDocument) -> Counter:
        return Counter(extract_pos_tags(doc.doc))

    def _extract_pos_bigrams(self, doc: Gram2VecDocument) -> Counter:
        return Counter(extract_pos_bigrams(doc.doc))

    def _extract_func_words(self, doc: Gram2VecDocument) -> Counter:
        func_words_vocab = set(self.vocab.get("func_words"))
        return Counter([t.text for t in doc.doc if t.text in func_words_vocab])

    def _extract_punctuation(self, doc: Gram2VecDocument) -> Counter:
        punc_vocab = set(self.vocab.get("punctuation"))
        return Counter([c for t in doc.doc for c in t.text if c in punc_vocab])

    def _extract_letters(self, doc: Gram2VecDocument) -> Counter:
        letters_vocab = set(self.vocab.get("letters"))
        return Counter([c for t in doc.doc for c in t.text if c in letters_vocab])

    def _extract_dep_labels(self, doc: Gram2VecDocument) -> Counter:
        return Counter(extract_dep_labels(doc.doc))

    def _extract_morph_tags(self, doc: Gram2VecDocument) -> Counter:
        return Counter(extract_morph_tags(doc.doc))

    def _extract_sentences(self, doc: Gram2VecDocument) -> Counter:
        matches = self.matcher.match_document(doc.doc)
        return Counter([m.pattern_name for m in matches])

    def _extract_emojis(self, doc: Gram2VecDocument) -> Counter:
        emojis_vocab = set(self.vocab.get("emojis"))
        extracted_emojis = demoji.findall_list(doc.raw, desc=False)
        counted: Counter = Counter()
        for e in extracted_emojis:
            if e in emojis_vocab:
                counted[e] += 1
            else:
                counted["OOV_emoji"] += 1
        return counted

    def _extract_num_tokens(self, doc: Gram2VecDocument) -> Counter:
        return Counter({"num_tokens": doc.num_tokens})

    def _get_extractor_map(self) -> Dict[str, Callable[[Gram2VecDocument], Counter]]:
        return {
            "pos_unigrams": self._extract_pos_unigrams,
            "pos_bigrams": self._extract_pos_bigrams,
            "func_words": self._extract_func_words,
            "punctuation": self._extract_punctuation,
            "letters": self._extract_letters,
            "emojis": self._extract_emojis,
            "dep_labels": self._extract_dep_labels,
            "morph_tags": self._extract_morph_tags,
            "sentences": self._extract_sentences,
            "num_tokens": self._extract_num_tokens,
        }

    # ~~~ Normalization and Vector Construction ~~~

    def _include_zero_vocab_counts(
        self, counted: Counter, vocab_items: Iterable[str]
    ) -> pd.Series:
        """Ensures all vocabulary items are present in order, filling unobserved with 0."""
        count_dict = {item: counted.get(item, 0) for item in vocab_items}
        return pd.Series(count_dict)

    def _normalize_feature_counts(
        self, feat_name: str, counts: pd.Series, num_tokens: int
    ) -> pd.Series:
        """Normalizes counted feature series according to feature type."""
        if feat_name == "num_tokens":
            return counts

        if feat_name in ("emojis", "punctuation", "func_words", "sentences"):
            norm_factor = num_tokens if num_tokens > 0 else 1
            return counts / norm_factor

        total = counts.sum()
        norm_factor = total if total > 0 else 1
        return counts / norm_factor

    def _apply_features_to_doc(
        self, doc: Gram2VecDocument, include_content_embedding: bool
    ) -> pd.Series:
        """Extracts and normalizes all active features for a single document."""
        extractor_map = self._get_extractor_map()
        feature_series: List[pd.Series] = []

        for feat_name in self.config.active_features():
            if feat_name not in extractor_map:
                continue

            raw_counts = extractor_map[feat_name](doc)

            if feat_name == "num_tokens":
                counts_series = pd.Series(raw_counts)
            else:
                vocab_items = self.vocab.get(feat_name)
                counts_series = self._include_zero_vocab_counts(raw_counts, vocab_items)

            normalized = self._normalize_feature_counts(
                feat_name, counts_series, doc.num_tokens
            )
            prefixed = normalized.add_prefix(f"{feat_name}:")
            feature_series.append(prefixed)

        if include_content_embedding:
            emb = pd.Series(doc.doc.vector).add_prefix("Embedding dim: ")
            feature_series.append(emb)

        return pd.concat(feature_series, axis=0)

    def _process_raw_documents(self, documents: Iterable[str]) -> List[Gram2VecDocument]:
        """Converts raw string documents into Gram2VecDocument instances."""
        doc_list = list(documents)
        cleaned_docs = [_remove_emojis(doc) for doc in doc_list]

        nlp_docs = list(
            self.nlp.pipe(
                cleaned_docs,
                batch_size=self.batch_size,
                n_process=self.n_process,
            )
        )

        token_counts = [len(d) for d in nlp_docs]
        sentence_counts = [len(list(d.sents)) for d in nlp_docs]

        return [
            Gram2VecDocument(raw, d, t_cnt, s_cnt)
            for raw, d, t_cnt, s_cnt in zip(doc_list, nlp_docs, token_counts, sentence_counts, strict=False)
        ]

    # ~~~ Public Vectorization API ~~~

    def from_documents(
        self,
        documents: Iterable[str],
        include_content_embedding: bool = False,
    ) -> pd.DataFrame:
        """Creates a stylistic feature vector matrix from an iterable of strings.

        Args:
            documents: Iterable of raw text strings.
            include_content_embedding: Whether to append the spaCy document embedding.

        Returns:
            pd.DataFrame where each row is a document and each column is a stylistic feature.

        """
        processed_docs = self._process_raw_documents(documents)
        feature_vectors = [
            self._apply_features_to_doc(doc, include_content_embedding)
            for doc in processed_docs
        ]
        if not feature_vectors:
            return pd.DataFrame()
        return pd.concat(feature_vectors, axis=1).T

    def from_jsonlines(
        self,
        path: Union[str, Path],
        include_content_embedding: bool = False,
    ) -> pd.DataFrame:
        """Creates a stylistic feature vector matrix from a .jsonl file or directory of .jsonl files.
        Requires 'fullText', 'authorIDs', and 'documentID' fields.

        Args:
            path: Path to a .jsonl file or directory containing .jsonl files.
            include_content_embedding: Whether to append the spaCy document embedding.

        Returns:
            pd.DataFrame with 'authorIDs' column and 'documentID' as index.

        """
        df = _load_jsonlines(path)
        documents, author_ids, document_ids = _get_json_entries(df)

        processed_docs = self._process_raw_documents(documents)
        feature_vectors = [
            self._apply_features_to_doc(doc, include_content_embedding)
            for doc in processed_docs
        ]
        if not feature_vectors:
            return pd.DataFrame()

        vector_df = pd.concat(feature_vectors, axis=1).T
        vector_df.insert(0, "authorIDs", author_ids.values)
        vector_df.set_index(document_ids.values, inplace=True)
        return vector_df

    def from_spacy_docs(
        self,
        docs: Iterable[Doc],
        raw_texts: Optional[Iterable[str]] = None,
        include_content_embedding: bool = False,
    ) -> pd.DataFrame:
        """Creates a stylistic feature vector matrix from pre-processed spaCy Doc objects.

        Args:
            docs: Iterable of spaCy Doc objects.
            raw_texts: Optional iterable of original raw strings (for emoji extraction).
            include_content_embedding: Whether to append the spaCy document embedding.

        """
        doc_list = list(docs)
        if raw_texts is not None:
            raw_list = list(raw_texts)
        else:
            raw_list = [d.text for d in doc_list]

        processed_docs = [
            Gram2VecDocument(raw, d, len(d), len(list(d.sents)))
            for raw, d in zip(raw_list, doc_list, strict=False)
        ]

        feature_vectors = [
            self._apply_features_to_doc(doc, include_content_embedding)
            for doc in processed_docs
        ]
        if not feature_vectors:
            return pd.DataFrame()
        return pd.concat(feature_vectors, axis=1).T


# Class alias for flexibility
Vectorizer = Gram2Vec
