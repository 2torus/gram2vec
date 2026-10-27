"""Syntax regex matcher for syntactic patterns on dependency linearized trees."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from spacy.tokens import Doc, Span


@dataclass
class Match:
    """Represents a regex match on a linearized sentence.

    Attributes:
        pattern_name: Identifier of the matched syntactic pattern.
        matched: Substring from the linearized dependency tree that matched.
        sentence: Raw text of the sentence that contained the match.

    Examples:
        >>> match = Match('passive', '(was-be-VBD-auxpass)', 'The apple was eaten.')
        >>> match.pattern_name
        'passive'
        >>> repr(match)
        'passive : The apple was eaten.'

    """

    pattern_name: str
    matched: str
    sentence: str

    def __repr__(self) -> str:
        return f'{self.pattern_name} : {self.sentence}'


DEFAULT_ENGLISH_PATTERNS: Dict[str, str] = {
    'it-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-(relcl|advcl)',
    'pseudo-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-(WP|WRB)-(dobj|advmod)',
    'all-cleft': r'(\([^-]*-be-[^-]*-[^-]*.*\([^-]*-all-(P)?DT)|(\([^-]*-all-(P)?DT-[^-]*.*\([^-]*-be-[^-]*)',
    'there-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([^-]*-there-EX-expl.*\([^-]*-[^-]*-[^-]*-attr.*\([^-]*-[^-]*-[^-]*-(relcl|acl)',
    'if-because-cleft': r'\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-[^-]*-advcl\([^-*]*-if-IN-mark',
    'passive': r'\([^-]*-[^-]*-(NN[^-]*|PRP|WDT)-nsubjpass.*\([^-]*-be-[^-]*-auxpass',
    'subj-relcl': r'\([^-]*-[^-]*-[^-]*-relcl.*\([^-]*-[^-]*-(WP|WDT)-nsubj',
    'obj-relcl': r'\([^-]*-[^-]*-NN[^-]*-(nsubj|attr).*\([^-]*-[^-]*-[^-]*-(relcl|ccomp).*\([^-]*-[^-]*-(WP|WDT|IN)-(pobj|dobj)',
    'tag-question': r'\([^-]*-(do|be|could|can|have)-[^-]*-ROOT.*\(\?-\?-\.-punct',
    'coordinate-clause': r'\([^-]*-[^-]*-CC-cc\).*\([^-]*-[^-]*-(VB[^-]*|JJ)-conj.*\([^-]*-[^-]*-[^-]*-nsubj',
}

DEFAULT_RUSSIAN_PATTERNS: Dict[str, str] = {
    'passive_rus': r'\([^-]*-[^-]*-VERB-ROOT.*?Voice=Pass.*?\)+$',
    'parataxis_rus': r'\([^-]*-[^-]*-[^-]*-parataxis.*\)',
    'participle_rus': r'\([^-]*-[^-]*-(?:VERB|ADJ)-ROOT.*?(?:ADJ|VERB)-(?:amod|acl)(?!:relcl).*?(?:VerbForm=Part|[а-я]+(?:ющ|ащ|ящ|вш|ем|им|нн|т)[а-я]+).*?\)+$',
    'gerund_rus': r'\([^-]*-[^-]*-(?:VERB|PROPN)-ROOT.*?(?:[А-Яа-я]+(?<!ющ|[ая]щ|вш)[яв(сь|ся)]-[^-]*-(?:VERB|NOUN|ADJ|PROPN)-(?:advcl|amod).*?(?:VerbForm=Conv|[^|]*)).*?\)+$',
    'conj_rus': r'\([^-]*-[^-]*-(?!SCONJ)\w+-ROOT.*?-conj.*?\)+$',
    'nested_structure_rus': r'\([^-]*-[^-]*-(?:VERB|ADV|NOUN)-ROOT.*?(?:VERB-(?:xcomp|advcl|ccomp|acl:relcl|acl)|AUX-aux:pass|ADV-advcl).*?(?:SCONJ-(?:mark|fixed|obj)|ADV-(?:mark|advmod)|который-PRON|что-PRON|чтобы-SCONJ|хотя-SCONJ|такой-DET).*?\)+$',
    'one_word_sent_rus': r'\([^-]*-[^-]*-(NOUN|VERB|PROPN)-ROOT-[^()]*\([^-]*-[^-]*-PUNCT-punct-\)\)',
    'diminutive_rus': r'.*\([а-яё]*(?:ик|[её]к|[её]нок|очк|ечк|ышк|оньк|еньк)[а-яё]*-[а-яё]*-NOUN-.*?\).*',
    'multiple_punct_rus': r'\([^-]*-[^-]*-[^-]*-ROOT.*?(?:(?:\(([!?])\1*-\1\1*-PUNCT-punct-\)){2,}|\(\.{2,}-\.{2,}-PUNCT-punct-\))\)+$',
    'additional_info_rus': r'\([^-]*-[^-]*-[^-]*-ROOT.*?(?:\([^-]*\([^)]+\)[^-]*-[^-]*-PUNCT-punct-[^)]*\)|(?:\([^-]*-[^-]*-PUNCT-punct-[^)]*\))){2,}\)+$',
}


class SyntaxRegexMatcher:
    r"""Encapsulates sentence regex patterns and methods to apply them to target documents.

    Supports syntactic pattern matching across linearized dependency parse trees
    for English ('en') and Russian ('ru').

    Examples:
        >>> matcher = SyntaxRegexMatcher(language='en')
        >>> 'passive' in matcher.patterns
        True
        >>> matcher.add_patterns({'custom_rule': r'\(.*\)'})
        >>> 'custom_rule' in matcher.patterns
        True
        >>> matcher.remove_patterns(['custom_rule'])
        >>> 'custom_rule' in matcher.patterns
        False

    """

    def __init__(
        self,
        language: str = 'en',
        patterns: Optional[Dict[str, str]] = None,
    ):
        """Initialize a SyntaxRegexMatcher instance.

        Args:
            language: Language code ('en' or 'ru') to load default syntactic patterns.
            patterns: Optional custom dictionary mapping pattern names to regular expression strings.

        """
        self.language = language

        if patterns is not None:
            self.patterns = dict(patterns)
        elif language == 'en':
            self.patterns = dict(DEFAULT_ENGLISH_PATTERNS)
        elif language == 'ru':
            self.patterns = dict(DEFAULT_RUSSIAN_PATTERNS)
        else:
            self.patterns = {}

    def print_patterns(self) -> None:
        """Print all registered patterns and their regular expressions to stdout."""
        for pattern_name, pattern in self.patterns.items():
            print(f'{pattern_name} : {pattern}\n')

    def _find_treegex_matches(self, doc: Doc) -> Tuple[Match, ...]:
        """Iterate through a document's sentences, applying every regex to each sentence.

        Args:
            doc: spaCy Doc to search for pattern matches.

        Returns:
            A tuple of Match objects found in the document.

        """
        matches: List[Match] = []
        for sent in doc.sents:
            tree_string = self.linearize_tree(sent)
            for name, pattern in self.patterns.items():
                match = re.search(pattern, tree_string)
                if match:
                    matches.append(Match(name, match.group(), sent.text))
        return tuple(matches)

    def add_patterns(self, patterns: Dict[str, str]) -> None:
        r"""Update patterns with a user supplied dictionary of {pattern_name: regex} pairs.

        Args:
            patterns: Dictionary mapping pattern names to regular expression strings.

        Examples:
            >>> matcher = SyntaxRegexMatcher()
            >>> matcher.add_patterns({'test_pat': r'\(.*-ROOT.*\)'})
            >>> 'test_pat' in matcher.patterns
            True

        """
        self.patterns.update(patterns)

    def remove_patterns(self, to_remove: Iterable[str]) -> None:
        """Remove the given pattern names from registered patterns.

        Args:
            to_remove: Iterable of pattern names to remove.

        Raises:
            KeyError: If a pattern name is not found in registered patterns.

        Examples:
            >>> matcher = SyntaxRegexMatcher()
            >>> matcher.remove_patterns(['tag-question'])
            >>> 'tag-question' in matcher.patterns
            False

        """
        for pattern_name in to_remove:
            if pattern_name not in self.patterns:
                raise KeyError(f"Pattern '{pattern_name}' not in registered patterns.")
            del self.patterns[pattern_name]

    def match_document(self, document: Doc) -> Tuple[Match, ...]:
        """Apply all registered patterns to a single spaCy document.

        Args:
            document: Processed spaCy Doc to analyze.

        Returns:
            Tuple of Match instances found across all sentences.

        """
        return self._find_treegex_matches(document)

    def match_documents(self, documents: Iterable[Doc]) -> List[Tuple[Match, ...]]:
        """Apply all registered patterns to a collection of spaCy documents.

        Args:
            documents: Iterable of spaCy Doc objects.

        Returns:
            List of tuples of Match instances, one tuple per document.

        """
        return [self._find_treegex_matches(doc) for doc in documents]

    def linearize_tree(self, sentence: Span) -> str:
        """Convert a spaCy dependency-parsed sentence into a linear tree string while preserving relations.

        Args:
            sentence: A spaCy Span representing a sentence.

        Returns:
            Linearized tree string representation.

        """

        def get_nt_count(sent: Span) -> int:
            return sum(1 for token in sent if list(token.children))

        def ending_parenthesis(n: int) -> str:
            return ')' * n

        def parse_dependency_parse(sent: Span) -> str:
            stack = [sent.root]
            result = ''
            while stack:
                token = stack.pop()
                if self.language == 'en':
                    result += f'({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}'
                else:
                    # Russian and general morphological representation
                    result += f'({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}-{token.morph}'

                stack.extend(reversed(list(token.children)))

                if not list(token.children):
                    result += ')'
            return result

        parse = parse_dependency_parse(sentence)
        nt_count = get_nt_count(sentence)
        return f'{parse}{ending_parenthesis(nt_count)}'
