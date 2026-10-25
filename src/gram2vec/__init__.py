"""Gram2Vec: Document embedding based on grammatical and stylistic features."""

__version__ = '2.0.0'
__author__ = 'Eric Sclafani'

from .config import FeatureConfig
from .feature_locator import FeatureLocator, FeatureSpan
from .matcher import Match, SyntaxRegexMatcher
from .vectorizer import Gram2Vec, Gram2VecDocument, Vectorizer
from .verbalizer import Verbalizer
from .vocab import Vocab

__all__ = [
    '__version__',
    '__author__',
    'Gram2Vec',
    'Vectorizer',
    'FeatureConfig',
    'FeatureLocator',
    'FeatureSpan',
    'Gram2VecDocument',
    'Match',
    'SyntaxRegexMatcher',
    'Verbalizer',
    'Vocab',
]
