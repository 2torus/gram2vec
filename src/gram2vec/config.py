"""Feature configuration options for Gram2Vec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional


@dataclass
class FeatureConfig:
    """Configuration for enabling/disabling specific grammatical features in Gram2Vec.

    Attributes:
        pos_unigrams: Coarse part-of-speech unigram frequencies.
        pos_bigrams: Part-of-speech bigram frequencies with sentence boundary markers.
        func_words: Function word frequencies.
        punctuation: Punctuation mark frequencies.
        letters: Individual character/letter frequencies (disabled by default).
        emojis: Emoji frequencies including out-of-vocabulary grouping.
        dep_labels: Universal dependency label frequencies.
        morph_tags: Morphological tag frequencies.
        sentences: Syntactic sentence pattern frequencies.
        num_tokens: Total token count of the document.
        pronouns: Personal pronoun frequencies grouped by person.
        degree: Comparative and superlative degree frequencies.
        avg_lengths: Average token, sentence, and noun-chunk lengths.
        named_entities: Named entity label frequencies (requires 'ner' spaCy component).

    Examples:
        >>> config = FeatureConfig.default()
        >>> config.pos_unigrams
        True
        >>> config.letters
        False

    """

    pos_unigrams: bool = True
    pos_bigrams: bool = True
    func_words: bool = True
    punctuation: bool = True
    letters: bool = False
    emojis: bool = True
    dep_labels: bool = True
    morph_tags: bool = True
    sentences: bool = True
    num_tokens: bool = True
    pronouns: bool = True
    degree: bool = True
    avg_lengths: bool = True
    # Requires the spaCy 'ner' component, which is excluded by default
    named_entities: bool = False

    @classmethod
    def default(cls) -> FeatureConfig:
        """Return the default configuration with standard stylometric features enabled.

        Returns:
            A FeatureConfig instance with default feature flags.

        Examples:
            >>> config = FeatureConfig.default()
            >>> config.pos_unigrams
            True
            >>> config.letters
            False

        """
        return cls()

    @classmethod
    def all(cls) -> FeatureConfig:
        """Enable all available features, including letters and named entities.

        Returns:
            A FeatureConfig instance with every feature enabled.

        Examples:
            >>> config = FeatureConfig.all()
            >>> config.letters and config.named_entities
            True

        """
        return cls(letters=True, named_entities=True)

    @classmethod
    def only(cls, *features: str) -> FeatureConfig:
        """Enable only the explicitly specified features, disabling all others.

        Args:
            *features: Names of the features to activate.

        Returns:
            A FeatureConfig instance with only the given features enabled.

        Raises:
            ValueError: If an unknown feature name is provided.

        Examples:
            >>> config = FeatureConfig.only('pos_unigrams', 'punctuation')
            >>> config.pos_unigrams
            True
            >>> config.pos_bigrams
            False

        """
        valid_fields = {f.name for f in fields(cls)}
        init_kwargs = dict.fromkeys(valid_fields, False)
        for feat in features:
            if feat in init_kwargs:
                init_kwargs[feat] = True
            else:
                raise ValueError(
                    f"Unknown feature '{feat}'. Valid features are: {sorted(valid_fields)}"
                )
        return cls(**init_kwargs)

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> FeatureConfig:
        """Create a FeatureConfig from a dictionary (e.g. {'pos_unigrams': 1, 'letters': 0}).

        Args:
            config_dict: Dictionary mapping feature names to boolean/truthy values.
                If None or empty, returns the default configuration.

        Returns:
            A FeatureConfig instance with configured flags.

        Examples:
            >>> config = FeatureConfig.from_dict({'letters': True, 'pos_unigrams': False})
            >>> config.letters
            True
            >>> config.pos_unigrams
            False

        """
        if not config_dict:
            return cls.default()

        valid_fields = {f.name for f in fields(cls)}
        kwargs: Dict[str, bool] = {}
        for key, val in config_dict.items():
            if key in valid_fields:
                kwargs[key] = bool(val)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, bool]:
        """Convert the configuration to a dictionary mapping feature names to booleans.

        Returns:
            A dictionary containing all feature names and their active states.

        Examples:
            >>> d = FeatureConfig.only('emojis').to_dict()
            >>> d['emojis']
            True
            >>> d['pos_unigrams']
            False

        """
        return asdict(self)

    def is_active(self, feature_name: str) -> bool:
        """Check if a given feature is activated.

        Args:
            feature_name: The name of the feature to check.

        Returns:
            True if the feature is active, False otherwise.

        Examples:
            >>> config = FeatureConfig.only('func_words')
            >>> config.is_active('func_words')
            True
            >>> config.is_active('letters')
            False

        """
        return getattr(self, feature_name, False)

    def active_features(self) -> List[str]:
        """Return a list of feature names that are activated in this configuration.

        Returns:
            List of active feature names.

        Examples:
            >>> config = FeatureConfig.only('pos_unigrams', 'punctuation')
            >>> config.active_features()
            ['pos_unigrams', 'punctuation']

        """
        return [f.name for f in fields(self) if getattr(self, f.name)]
