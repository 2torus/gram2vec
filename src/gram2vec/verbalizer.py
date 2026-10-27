"""Verbalizer for translating stylometric z-score deviations into natural language."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import zscore

NAME_MAPPING_EN: Dict[str, str] = {
    'pos_unigrams': 'part of speech unigram',
    'pos_bigrams': 'part of speech bigram',
    'morph_tags': 'morphological tag',
    'dep_labels': 'dependency parse label',
    'func_words': 'function word',
    'punctuation': 'punctuation mark',
    'letters': 'letter',
    'sentences': 'sentence type',
    'emojis': 'emoji',
    'num_tokens': 'token count',
}

NAME_MAPPING_RU: Dict[str, str] = {
    'pos_unigrams': 'часть речи (униграмма)',
    'pos_bigrams': 'часть речи (биграмма)',
    'morph_tags': 'морфологический признак',
    'dep_labels': 'синтаксическая связь',
    'func_words': 'служебное слово',
    'punctuation': 'знак препинания',
    'letters': 'буква',
    'sentences': 'тип предложения',
    'emojis': 'эмодзи',
    'num_tokens': 'количество токенов',
}


class Verbalizer:
    """Encapsulates the z-score verbalization functionality of Gram2Vec.

    Computes salient stylometric features on document and author levels.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'authorIDs': ['a1', 'a1', 'a2', 'a2'],
        ...     'pos_unigrams:NOUN': [0.10, 0.12, 0.35, 0.38],
        ... }, index=['d1', 'd2', 'd3', 'd4'])
        >>> v = Verbalizer(df, zscore_threshold=0.5)
        >>> v.language
        'en'
        >>> auth_df = v.verbalize_author_id('a2')
        >>> not auth_df.empty
        True

    """

    def __init__(
        self,
        docs_df: pd.DataFrame,
        zscore_threshold: float = 2.0,
        language: str = 'en',
        custom_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize a Verbalizer instance.

        Args:
            docs_df: DataFrame containing document feature vectors and optional 'authorIDs'.
            zscore_threshold: Absolute z-score threshold for identifying salient features.
            language: Language code ('en' or 'ru') for verbal template messages.
            custom_mapping: Optional dictionary mapping feature prefixes to readable descriptions.

        """
        self.threshold = zscore_threshold
        self.language = language
        self.docs_df = docs_df
        self.name_mapping = (
            custom_mapping
            if custom_mapping is not None
            else (NAME_MAPPING_RU if language == 'ru' else NAME_MAPPING_EN)
        )

        if 'authorIDs' in docs_df.columns:
            self.author_df = self._make_author_df(self.docs_df)
        else:
            self.author_df = pd.DataFrame()

    def _exclude_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Exclude given columns from numerical operations.

        Args:
            df: Input DataFrame.
            cols: List of column names to exclude.

        Returns:
            DataFrame with specified columns excluded.

        """
        return df.loc[:, ~df.columns.isin(cols)]

    def _get_author_docs(self, author_id: str) -> pd.DataFrame:
        """Retrieve an author's documents from self.docs_df.

        Args:
            author_id: Identifier of the author.

        Returns:
            Filtered DataFrame containing only the author's rows.

        """
        return self.docs_df.loc[self.docs_df['authorIDs'] == author_id]

    def _make_author_df(self, docs_df: pd.DataFrame) -> pd.DataFrame:
        """Average document vectors by author.

        Args:
            docs_df: DataFrame containing 'authorIDs' and feature columns.

        Returns:
            DataFrame of averaged feature vectors indexed by author ID.

        """
        author_ids = docs_df['authorIDs'].unique()
        author_ids_to_avs = {}

        for author_id in author_ids:
            author_doc_entries = self._get_author_docs(author_id)
            author_doc_vectors = self._exclude_columns(
                author_doc_entries, cols=['documentID', 'authorIDs']
            )
            author_ids_to_avs[author_id] = author_doc_vectors.mean(axis=0)

        return pd.DataFrame(author_ids_to_avs).T

    def _get_threshold_zscores_idxs(
        self, zscores: Union[np.ndarray, pd.Series, Any]
    ) -> np.ndarray:
        """Get column indices for zscores exceeding threshold.

        Args:
            zscores: 1D array or Series of z-score values.

        Returns:
            NumPy array of indices where absolute z-score >= threshold.

        """
        arr = np.asarray(zscores)
        return np.where(np.abs(arr) >= self.threshold)[0]

    def _get_record_from_id(self, df: pd.DataFrame, id_val: str) -> int:
        """Retrieve index of record in dataframe by index label.

        Args:
            df: DataFrame to search.
            id_val: Index label value to locate.

        Returns:
            Integer row position.

        Raises:
            ValueError: If id_val is not found in dataframe index.

        """
        if id_val in df.index:
            return df.index.to_list().index(id_val)
        raise ValueError(f"id '{id_val}' not found in dataframe index")

    def _get_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate z-scores for numerical columns.

        Args:
            df: DataFrame containing numeric feature columns.

        Returns:
            DataFrame of standardized z-scores with matching columns and index.

        """
        numeric_df = self._exclude_columns(df, cols=['authorIDs']).select_dtypes(
            include=np.number
        )
        scores = zscore(numeric_df.to_numpy(), axis=0, nan_policy='omit')
        return pd.DataFrame(scores, index=numeric_df.index, columns=numeric_df.columns)

    def _get_identifying_features(self, author_id: str, df: pd.DataFrame) -> pd.Series:
        """Calculate feature zscores for an author ID and keep those exceeding threshold.

        Args:
            author_id: Author ID to analyze.
            df: DataFrame of author-level feature vectors.

        Returns:
            Series containing salient feature names and their z-scores.

        """
        all_zscores = self._get_zscores(df)
        idx = self._get_record_from_id(df, author_id)
        extracted_zscores = all_zscores.iloc[idx]
        chosen_indices = self._get_threshold_zscores_idxs(extracted_zscores.to_numpy())
        return extracted_zscores.iloc[chosen_indices]

    def _template(self, doc_or_author: str, feat_name: str, direction: str) -> str:
        """Generate a readable verbal explanation for a feature.

        Args:
            doc_or_author: Either 'document' or 'author'.
            feat_name: Full feature name (e.g. 'pos_unigrams:NOUN').
            direction: Direction of deviation ('more' or 'less').

        Returns:
            Formatted explanation string in configured language.

        """
        parts = feat_name.split(':', 1)
        feat_type = parts[0]
        feat_val = parts[1] if len(parts) > 1 else feat_name
        type_desc = self.name_mapping.get(feat_type, feat_type)

        if self.language == 'ru':
            dir_str = 'реже' if direction == 'less' else 'чаще'
            entity_str = (
                'автор использует' if doc_or_author == 'author' else 'документ содержит'
            )
            avg_str = (
                'среднего автора' if doc_or_author == 'author' else 'среднего документа'
            )
            return f"Этот {entity_str} признак ({type_desc}) '{feat_val}' {dir_str}, чем в среднем для {avg_str}."

        return f"This {doc_or_author} uses the {type_desc} '{feat_val}' {direction} than the average {doc_or_author}"

    def _verbalize_zscores(self, zscores: pd.Series, to_verbalize: str) -> List[str]:
        """Create a list of verbalized zscore sentences.

        Args:
            zscores: Series mapping feature names to z-scores.
            to_verbalize: Entity type ('document' or 'author').

        Returns:
            List of generated natural language sentences.

        """
        converted = []
        for feat_name, score in zscores.items():
            direction = 'less' if score < 0 else 'more'
            converted.append(self._template(to_verbalize, str(feat_name), direction))
        return converted

    def verbalize_document_vector(
        self, document_vector: Union[np.ndarray, pd.Series]
    ) -> pd.DataFrame:
        """Retrieve a document's most distinguishing features given a single vector.

        Args:
            document_vector: 1D array or Series representing document feature vector.

        Returns:
            pd.DataFrame with 'feature_name', 'zscore', and 'verbalized' columns.

        """
        numeric_df = self.docs_df.select_dtypes(include=np.number)
        feature_names = numeric_df.columns.to_list()

        vec = (
            document_vector.to_numpy()
            if isinstance(document_vector, pd.Series)
            else np.asarray(document_vector)
        )

        fit_matrix = np.vstack([numeric_df.to_numpy(), vec])
        scores = zscore(fit_matrix, axis=0, nan_policy='omit')
        unseen_doc_zscores = pd.Series(scores[-1], index=feature_names)

        chosen_indices = self._get_threshold_zscores_idxs(unseen_doc_zscores.to_numpy())
        selected_zscores = unseen_doc_zscores.iloc[chosen_indices]
        templated_strings = self._verbalize_zscores(selected_zscores, 'document')

        return pd.DataFrame(
            {
                'feature_name': [feature_names[int(i)] for i in chosen_indices],
                'zscore': selected_zscores.to_numpy(),
                'verbalized': templated_strings,
            }
        )

    def verbalize_author_id(self, author_id: str) -> pd.DataFrame:
        """Retrieve an author's most distinguishing features given an author ID.

        Args:
            author_id: Identifier of the author.

        Returns:
            pd.DataFrame with 'feature_name', 'zscore', and 'verbalized' columns.

        Raises:
            ValueError: If author_df is empty because 'authorIDs' was missing.

        """
        selected_id_zscores = self._get_identifying_features(author_id, self.author_df)
        templated_strings = self._verbalize_zscores(selected_id_zscores, 'author')

        return pd.DataFrame(
            {
                'feature_name': selected_id_zscores.index.to_list(),
                'zscore': selected_id_zscores.values,
                'verbalized': templated_strings,
            }
        )
