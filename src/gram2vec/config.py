"""Feature configuration options for Gram2Vec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional


@dataclass
class FeatureConfig:
    """Configuration for enabling/disabling specific grammatical features in Gram2Vec."""

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
        """Returns the default configuration."""
        return cls()

    @classmethod
    def all(cls) -> FeatureConfig:
        """Enables all available features, including letters and named entities."""
        return cls(letters=True, named_entities=True)

    @classmethod
    def only(cls, *features: str) -> FeatureConfig:
        """Enables only the explicitly specified features, disabling all others."""
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
        """Creates a FeatureConfig from a dictionary (e.g. {'pos_unigrams': 1, 'letters': 0})."""
        if not config_dict:
            return cls.default()

        valid_fields = {f.name for f in fields(cls)}
        kwargs: Dict[str, bool] = {}
        for key, val in config_dict.items():
            if key in valid_fields:
                kwargs[key] = bool(val)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, bool]:
        """Converts the configuration to a dictionary."""
        return asdict(self)

    def is_active(self, feature_name: str) -> bool:
        """Checks if a given feature is activated."""
        return getattr(self, feature_name, False)

    def active_features(self) -> List[str]:
        """Returns a list of feature names that are activated in this configuration."""
        return [f.name for f in fields(self) if getattr(self, f.name)]
